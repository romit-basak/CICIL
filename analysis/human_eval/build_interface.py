"""Build the self-contained human-eval annotation page (human_eval.html).

Embeds the sample data as a JSON blob inside one HTML file -- fetch() of local
CSVs fails under file:// (CORS), and a single file keeps annotator setup at
"open it in a browser". No backend, no dependencies: vanilla JS, autosave to
localStorage, results exported as a CSV download.

BLINDING: this script reads sample_spanish.csv, sample_english.csv, and
sample_target.csv only. It must never read sample_key.csv -- the A/B -> arm
mapping stays out of the page so annotators can't unblind themselves.

    uv run python -m analysis.human_eval.build_interface   # -> human_eval.html

Annotators need this repo checked out with the dataset at
data/americasnlp2026/ (images are referenced by relative path, not embedded).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "human_eval.html"  # reassigned by --suffix in main()

# image-id prefix -> dataset language directory
LANG_DIR = {"grn": "guarani", "yua": "maya", "hch": "wixarika",
            "nlv": "nahuatl", "bzd": "bribri"}


def load_items(suffix: str = "", split: str = "dev") -> list[dict]:
    def by_id(path: Path) -> dict[str, dict]:
        with path.open(encoding="utf-8") as f:
            return {r["sample_id"]: r for r in csv.DictReader(f)}

    spanish = by_id(HERE / f"sample_spanish{suffix}.csv")
    english = by_id(HERE / f"sample_english{suffix}.csv")
    target = by_id(HERE / f"sample_target{suffix}.csv")

    missing = sorted(set(spanish) - set(english))
    if missing:
        raise SystemExit(
            f"sample_english.csv is missing rows for {missing} -- run "
            f"translate_english.py first."
        )

    # Optional: this round's CBIR retrieval reference (build_cbir_refs.py).
    # Present only for rounds comparing RAG arms; absent (e.g. round 1) is
    # fine -- the reference block and its rating question are simply omitted.
    cbir_path = HERE / f"sample_cbir_ref{suffix}.csv"
    cbir = by_id(cbir_path) if cbir_path.exists() else {}

    # Resolve image paths via the same basename-lookup data_io.py uses
    # everywhere else, rather than hand-building a path template: dev and
    # pilot use DIFFERENT on-disk layouts (dev: <lang>/images/, pilot:
    # images/<lang>/) and a hardcoded template silently pointed at the wrong
    # one for round 3 (pilot). One shared resolver means this can't drift again.
    from src.stage1.data_io import load_split
    paths_cache: dict[tuple[str, str], dict[str, Path]] = {}

    def resolve_image_src(fname: str, lang_dir: str, row_split: str) -> str:
        key = (lang_dir, row_split)
        if key not in paths_cache:
            paths_cache[key] = {ex.image_path.name: ex.image_path
                                for ex in load_split(lang_dir, row_split)}
        abs_path = paths_cache[key].get(fname)
        if abs_path is None:
            # Fall back to a guessed path so the browser's onerror message is
            # still informative about what was looked for.
            return f"../../data/americasnlp2026/data/{row_split}/{lang_dir}/images/{fname}"
        return os.path.relpath(abs_path, start=HERE)

    items = []
    for sid, row in spanish.items():
        fname = row["image_filename"]  # keep exact case (bzd_042.JPG)
        prefix = fname.split("_")[0]
        lang_dir = LANG_DIR[prefix]
        # Rounds mixing splits (round 4: wixarika pilot + cross-culture dev)
        # carry a per-row split column; earlier rounds fall back to --split.
        row_split = row.get("split") or split
        item = {
            "sample_id": sid,
            "language": row["language"],
            "image_filename": fname,
            "image_src": resolve_image_src(fname, lang_dir, row_split),
            "caption_A": row["caption_A"],
            "caption_B": row["caption_B"],
            "english_A": english[sid]["english_A"],
            "english_B": english[sid]["english_B"],
            "target_A": target[sid]["caption_A"],
            "target_B": target[sid]["caption_B"],
        }
        c = cbir.get(sid)
        if c and c.get("cbir_title"):
            item["cbir_title"] = c["cbir_title"]
            item["cbir_description_en"] = english[sid].get("cbir_description_en", "")
            item["cbir_image_url"] = c.get("cbir_image_url", "")
            item["cbir_score"] = c.get("cbir_score", "")
            item["cbir_band"] = c.get("cbir_band", "")
            item["cbir_page_url"] = c.get("cbir_page_url", "")
        gold_en = english[sid].get("gold_english", "")
        if gold_en:
            item["gold_english"] = gold_en
        items.append(item)
    items.sort(key=lambda r: r["sample_id"])
    return items


DIMENSIONS = [
    ("cultural_accuracy", "Cultural accuracy",
     ["0 — the informative cultural claims are fabricated or wrong "
      "(invented sites/regions/artifact identities); merely echoing the "
      "given culture name doesn't rescue it",
      "1 — no false cultural claims but no informative ones either (honest "
      "silence, bare hedged attribution), or genuinely correct specifics "
      "mixed with wrong ones",
      "2 — makes at least one correct, informative cultural claim beyond "
      "the given culture name"]),
    ("faithfulness", "Image faithfulness",
     ["0 — contradicts the image / hallucinates major content (incl. "
      "confident claims about things not visible, e.g. invented locations)",
      "1 — mostly right, one notable wrong or missing element",
      "2 — faithful to the salient content; a grounded hedge about a "
      "visible object's identity is not a fault"]),
    ("fluency", "Fluency (content coherence — see rubric's pivot caveat)",
     ["0 — broken: repetition loops, truncation, incoherent",
      "1 — understandable but awkward or confused",
      "2 — natural and coherent"]),
]

# Scored ONCE per item, not per A/B slot: retrieval happens once per image
# regardless of which two captions are being compared (both RAG arms in a
# round see the identical CBIR neighbor). Only shown when sample_cbir_ref*.csv
# exists (see build_cbir_refs.py) -- absent for rounds with no RAG arm.
CBIR_DIMENSION = ("cbir_relevance", "Is the retrieved reference actually related to this image?",
                  ["0 — unrelated: wrong subject/culture, retrieval clearly failed",
                   "1 — partially related: same culture or general theme, wrong specific subject",
                   "2 — clearly related: same or closely matching subject"])

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CICIL Human Evaluation — Stage 1 (English-assisted)</title>
<style>
  :root {{
    --bg: #f5f5f4; --fg: #1c1917; --card: #fff; --shadow: rgba(0,0,0,.08);
    --border: #d6d3d1; --muted: #57534e; --lbl: #64748b; --arm-bd: #e7e5e4;
    --es-bg: #f8fafc; --es-bd: #e2e8f0; --en-bg: #eff6ff; --en-bd: #bfdbfe;
    --tg-bg: #fafaf9; --tg-fg: #78716c;
    --gold-bg: #fefce8; --gold-bd: #fde047;
    --warn-bg: #fffbeb; --warn-bd: #fde68a;
    --err-bg: #fef2f2; --err-bd: #fca5a5; --err-fg: #991b1b;
    --btn-bg: #fff; --btn-bd: #a8a29e; --primary: #1d4ed8;
    --done: #15803d; --todo: #b45309;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #121214; --fg: #e7e5e4; --card: #1d1d20; --shadow: rgba(0,0,0,.5);
      --border: #3f3f46; --muted: #a1a1aa; --lbl: #94a3b8; --arm-bd: #3f3f46;
      --es-bg: #1e242a; --es-bd: #334155; --en-bg: #17203a; --en-bd: #1e3a5f;
      --tg-bg: #1a1a1c; --tg-fg: #a8a29e;
      --gold-bg: #2a2410; --gold-bd: #a16207;
      --warn-bg: #2a2410; --warn-bd: #a16207;
      --err-bg: #2a1215; --err-bd: #7f1d1d; --err-fg: #fca5a5;
      --btn-bg: #2a2a2e; --btn-bd: #52525b; --primary: #2563eb;
      --done: #4ade80; --todo: #fbbf24;
    }}
  }}
  body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background: var(--bg); color: var(--fg); }}
  .wrap {{ max-width: 880px; margin: 0 auto; padding: 16px; }}
  .card {{ background: var(--card); border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 1px 3px var(--shadow); }}
  img.eval {{ max-width: 100%; max-height: 420px; display: block; margin: 0 auto; border-radius: 6px; }}
  .imgerr {{ background: var(--err-bg); border: 1px solid var(--err-bd); color: var(--err-fg); padding: 14px; border-radius: 6px; }}
  h1 {{ font-size: 1.15rem; }} h2 {{ font-size: 1rem; margin: 4px 0 8px; }}
  .cap {{ margin: 6px 0; padding: 10px 12px; border-radius: 6px; line-height: 1.45; }}
  .cap-es {{ background: var(--es-bg); border: 1px solid var(--es-bd); }}
  .cap-en {{ background: var(--en-bg); border: 1px solid var(--en-bd); }}
  .cap-tg {{ background: var(--tg-bg); border: 1px dashed var(--border); color: var(--tg-fg); font-size: .9rem; }}
  .cap-gold {{ background: var(--gold-bg); border: 1px solid var(--gold-bd); font-weight: 500; }}
  .lbl {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: var(--lbl); display: block; margin-bottom: 3px; }}
  .dim {{ margin: 8px 0 12px; }}
  .dim b {{ display: block; margin-bottom: 4px; }}
  .dim label {{ display: block; font-size: .88rem; margin: 2px 0 2px 6px; cursor: pointer; }}
  .arm {{ border-top: 2px solid var(--arm-bd); padding-top: 10px; margin-top: 14px; }}
  textarea {{ width: 100%; min-height: 70px; box-sizing: border-box; font: inherit; padding: 8px; border-radius: 6px; border: 1px solid var(--border); background: var(--btn-bg); color: var(--fg); }}
  .nav {{ display: flex; gap: 10px; align-items: center; justify-content: space-between; }}
  /* Sticky top nav: Prev/Next stay reachable while the long form scrolls.
     (Requested fix: the buttons scrolled away and Export sat where Next was
     expected, inviting misclicks.) */
  .nav-top {{ position: sticky; top: 8px; z-index: 10; }}
  button {{ font: inherit; padding: 8px 18px; border-radius: 6px; border: 1px solid var(--btn-bd); background: var(--btn-bg); color: var(--fg); cursor: pointer; }}
  button.primary {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
  button:disabled {{ opacity: .45; cursor: default; }}
  .progress {{ font-variant-numeric: tabular-nums; color: var(--muted); }}
  .warn {{ background: var(--warn-bg); border: 1px solid var(--warn-bd); padding: 10px 12px; border-radius: 6px; margin-top: 10px; }}
  .done {{ color: var(--done); }} .todo {{ color: var(--todo); }}
  #gate input {{ font: inherit; padding: 8px; border-radius: 6px; border: 1px solid var(--border); background: var(--btn-bg); color: var(--fg); width: 240px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card" id="gate">
    <h1>CICIL Human Evaluation — Stage 1 captions (English-assisted)</h1>
    <p>Score captions <b>A</b> and <b>B</b> independently on all three dimensions,
    <i>then</i> pick a preference. Judge against the <b>image</b>, reading the Spanish
    via its English translation. The Indigenous-language captions are shown for
    reference only — do not score them. See <code>RUBRIC.md</code> for anchors and
    protocol.</p>
    <p><label>Your name (required): <input id="annotator" placeholder="e.g. Tisha"></label>
    <button class="primary" onclick="start()">Start / resume</button></p>
  </div>
  <div id="app" style="display:none"></div>
</div>
<script>
const ITEMS = {items_json};
const DIMS = {dims_json};
const CBIR_DIM = {cbir_dim_json};
let annotator = "";
let idx = 0;

const key = (sid) => `cicil_eval::${{annotator}}::${{sid}}`;
const load = (sid) => JSON.parse(localStorage.getItem(key(sid)) || "{{}}");
const save = (sid, obj) => localStorage.setItem(key(sid), JSON.stringify(obj));

function start() {{
  annotator = document.getElementById("annotator").value.trim();
  if (!annotator) {{ alert("Please enter your name first."); return; }}
  document.getElementById("gate").style.display = "none";
  document.getElementById("app").style.display = "";
  render();
}}

function setField(sid, field, value) {{
  const cur = load(sid); cur[field] = value; save(sid, cur);
  document.getElementById("status").innerHTML = statusLine();
}}

function isComplete(sid) {{
  const it = ITEMS.find(i => i.sample_id === sid);
  const r = load(sid);
  const dimsDone = DIMS.every(d => r["A_" + d[0]] !== undefined && r["B_" + d[0]] !== undefined)
                   && r["preference_A_B_tie"] !== undefined;
  const cbirDone = !it.cbir_title || r[CBIR_DIM[0]] !== undefined;
  return dimsDone && cbirDone;
}}

function statusLine() {{
  const done = ITEMS.filter(it => isComplete(it.sample_id)).length;
  return `<span class="${{done === ITEMS.length ? "done" : "todo"}}">${{done}}/${{ITEMS.length}} items fully scored</span>`;
}}

function dimBlock(slot, sid, saved) {{
  return DIMS.map(([dkey, dname, anchors]) => {{
    const field = slot + "_" + dkey;
    const opts = anchors.map((a, v) =>
      `<label><input type="radio" name="${{field}}" ${{saved[field] == v ? "checked" : ""}}
        onchange="setField('${{sid}}','${{field}}',${{v}})"> ${{a}}</label>`).join("");
    return `<div class="dim"><b>${{dname}}</b>${{opts}}</div>`;
  }}).join("");
}}

function esc(s) {{
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

function refBlock(it, sid, saved) {{
  if (!it.cbir_title) return "";
  const [dkey, dname, anchors] = CBIR_DIM;
  const opts = anchors.map((a, v) =>
    `<label><input type="radio" name="${{dkey}}" ${{saved[dkey] == v ? "checked" : ""}}
      onchange="setField('${{sid}}','${{dkey}}',${{v}})"> ${{a}}</label>`).join("");
  const link = it.cbir_page_url
    ? `<a href="${{it.cbir_page_url}}" target="_blank" rel="noopener">open on Wikimedia Commons &rarr;</a>`
    : "(no source link recorded)";
  // Live hotlink to Commons' own CDN (Special:FilePath) -- not a copy stored
  // in this repo, same as how Wikipedia itself embeds Commons media. Needs an
  // internet connection to render; falls back to the text+link if it 404s.
  const img = it.cbir_image_url
    ? `<img class="eval" style="max-height:280px" src="${{it.cbir_image_url}}"
         onerror="this.outerHTML='<p class=warn>Preview failed to load (offline, or the file moved on Commons) — use the link below instead.</p>'">`
    : "";
  return `
    <div class="arm">
      <h2>Retrieved reference (what the RAG arms were given as context)</h2>
      ${{img}}
      <div class="cap cap-tg">
        <span class="lbl">Commons title &middot; match strength: ${{it.cbir_band}} (score ${{it.cbir_score}})</span>
        ${{esc(it.cbir_title)}}<br>${{esc(it.cbir_description_en || "")}}
      </div>
      <p>${{link}} — compare it to the image above.</p>
      <div class="dim"><b>${{dname}}</b>${{opts}}</div>
    </div>`;
}}

function render() {{
  const it = ITEMS[idx];
  const sid = it.sample_id;
  const saved = load(sid);
  const capBlock = (slot) => `
    <div class="arm"><h2>Caption ${{slot}}</h2>
      <div class="cap cap-en"><span class="lbl">English (judge this meaning)</span>${{esc(it["english_" + slot])}}</div>
      <div class="cap cap-es"><span class="lbl">Spanish (Stage 1 output)</span>${{esc(it["caption_" + slot])}}</div>
      <div class="cap cap-tg"><span class="lbl">${{it.language}} caption (Stage 2 output — display only, not judged)</span>${{esc(it["target_" + slot])}}</div>
      ${{dimBlock(slot, sid, saved)}}
    </div>`;
  const pref = ["A", "B", "tie"].map(v =>
    `<label style="display:inline-block;margin-right:16px"><input type="radio" name="pref"
      ${{saved["preference_A_B_tie"] === v ? "checked" : ""}}
      onchange="setField('${{sid}}','preference_A_B_tie','${{v}}')"> ${{v}}</label>`).join("");
  const navBar = (statusSpan) => `
      <button onclick="go(-1)" ${{idx === 0 ? "disabled" : ""}}>&larr; Prev</button>
      <span class="progress">Item ${{idx + 1}}/${{ITEMS.length}} &nbsp;·&nbsp; ${{sid}} (${{it.language}}) &nbsp;·&nbsp; ${{statusSpan}}</span>
      <button onclick="go(1)" ${{idx === ITEMS.length - 1 ? "disabled" : ""}}>Next &rarr;</button>`;
  document.getElementById("app").innerHTML = `
    <div class="card nav nav-top">
      ${{navBar(`<span id="status">${{statusLine()}}</span>`)}}
    </div>
    <div class="card">
      <img class="eval" src="${{it.image_src}}"
           onerror="this.outerHTML='<div class=imgerr>Image not found at <code>${{it.image_src}}</code>.<br>The dataset must be checked out at <code>data/americasnlp2026/</code> in this repo (it is gitignored — download it separately).</div>'">
      ${{it.gold_english ? `<div class="cap cap-gold"><span class="lbl">Gold reference (real human-written caption — context, not the answer key: it may name events/places NOT visible in the image; judge captions against the image)</span>${{esc(it.gold_english)}}</div>` : ""}}
    </div>
    <div class="card">
      ${{refBlock(it, sid, saved)}}
      ${{capBlock("A")}}
      ${{capBlock("B")}}
      <div class="arm"><h2>Preference — better overall description of this image</h2>${{pref}}</div>
      <div class="arm"><h2>Notes (cultural errors, hallucinations, category hits/misses)</h2>
        <textarea onchange="setField('${{sid}}','notes',this.value)">${{esc(saved["notes"] || "")}}</textarea></div>
    </div>
    <div class="card nav">
      ${{navBar(`<span>${{statusLine()}}</span>`)}}
    </div>
    <div class="card nav">
      <span>Autosaves locally as you go (per annotator name).</span>
      <span>
        <button onclick="showCSV()">Show CSV (copy manually)</button>
        <button class="primary" onclick="exportCSV()">Export results CSV</button>
      </span>
    </div>
    <div class="card" id="csvbox" style="display:none">
      <h2>Results CSV — select all &amp; copy, then save as <code>results/human_eval_results_&lt;name&gt;{round_suffix}.csv</code></h2>
      <p>Use this if the download button does nothing (e.g. VSCode's integrated browser
      can't download files).</p>
      <textarea id="csvtext" style="min-height:180px" onclick="this.select()"></textarea>
    </div>`;
}}

function go(delta) {{
  idx = Math.max(0, Math.min(ITEMS.length - 1, idx + delta));
  render();
  window.scrollTo(0, 0);
}}

function csvQuote(v) {{
  v = String(v ?? "");
  return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}}

function buildCSV() {{
  const cols = ["annotator", "sample_id", "language",
                ...DIMS.flatMap(d => ["A_" + d[0], "B_" + d[0]]),
                "preference_A_B_tie", CBIR_DIM[0], "notes"];
  const lines = [cols.join(",")];
  for (const it of ITEMS) {{
    const r = load(it.sample_id);
    lines.push(cols.map(c =>
      csvQuote(c === "annotator" ? annotator :
               c === "sample_id" ? it.sample_id :
               c === "language" ? it.language : r[c])).join(","));
  }}
  return lines.join("\\n") + "\\n";
}}

function confirmComplete() {{
  const incomplete = ITEMS.filter(it => !isComplete(it.sample_id)).map(it => it.sample_id);
  return !incomplete.length ||
         confirm("Not fully scored yet: " + incomplete.join(", ") + "\\nExport anyway?");
}}

function showCSV() {{
  if (!confirmComplete()) return;
  const box = document.getElementById("csvbox");
  box.style.display = "";
  document.getElementById("csvtext").value = buildCSV();
  box.scrollIntoView();
}}

function exportCSV() {{
  if (!confirmComplete()) return;
  const blob = new Blob([buildCSV()], {{ type: "text/csv;charset=utf-8" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `human_eval_results_${{annotator.replace(/\\s+/g, "_")}}{round_suffix}.csv`;
  a.click();
  alert("If no file downloaded (VSCode's browser can't), use 'Show CSV' and copy it " +
        "manually. Either way the file belongs in analysis/human_eval/results/.");
}}
</script>
</body>
</html>
"""


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build the annotation HTML.")
    ap.add_argument("--suffix", default="",
                    help="round suffix (e.g. _round2): reads sample_*{suffix}.csv, "
                         "writes human_eval{suffix}.html")
    ap.add_argument("--split", default="dev", choices=["pilot", "dev", "test"],
                    help="dataset split the images live under (pilot for round 3)")
    args = ap.parse_args()
    items = load_items(args.suffix, args.split)
    html = HTML_TEMPLATE.format(
        items_json=json.dumps(items, ensure_ascii=False),
        dims_json=json.dumps(DIMENSIONS, ensure_ascii=False),
        cbir_dim_json=json.dumps(CBIR_DIMENSION, ensure_ascii=False),
        round_suffix=args.suffix,
    )
    out_file = HERE / f"human_eval{args.suffix}.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Wrote {out_file} ({len(items)} items). Open it in a browser to annotate.")


if __name__ == "__main__":
    main()
