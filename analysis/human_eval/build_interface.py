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
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_FILE = HERE / "human_eval.html"

# image-id prefix -> dataset language directory
LANG_DIR = {"grn": "guarani", "yua": "maya", "hch": "wixarika",
            "nlv": "nahuatl", "bzd": "bribri"}


def load_items() -> list[dict]:
    def by_id(path: Path) -> dict[str, dict]:
        with path.open(encoding="utf-8") as f:
            return {r["sample_id"]: r for r in csv.DictReader(f)}

    spanish = by_id(HERE / "sample_spanish.csv")
    english = by_id(HERE / "sample_english.csv")
    target = by_id(HERE / "sample_target.csv")

    missing = sorted(set(spanish) - set(english))
    if missing:
        raise SystemExit(
            f"sample_english.csv is missing rows for {missing} -- run "
            f"translate_english.py first."
        )

    items = []
    for sid, row in spanish.items():
        fname = row["image_filename"]  # keep exact case (bzd_042.JPG)
        prefix = fname.split("_")[0]
        lang_dir = LANG_DIR[prefix]
        items.append({
            "sample_id": sid,
            "language": row["language"],
            "image_filename": fname,
            "image_src": f"../../data/americasnlp2026/data/dev/{lang_dir}/images/{fname}",
            "caption_A": row["caption_A"],
            "caption_B": row["caption_B"],
            "english_A": english[sid]["english_A"],
            "english_B": english[sid]["english_B"],
            "target_A": target[sid]["caption_A"],
            "target_B": target[sid]["caption_B"],
        })
    items.sort(key=lambda r: r["sample_id"])
    return items


DIMENSIONS = [
    ("cultural_accuracy", "Cultural accuracy",
     ["0 — wrong/absent: culturally generic, or names the wrong culture",
      "1 — partial: right gist but vague, or mixes correct + incorrect detail",
      "2 — accurate: correct culturally specific content, right terms"]),
    ("faithfulness", "Image faithfulness",
     ["0 — contradicts the image / hallucinates major content",
      "1 — mostly right, one notable wrong or missing element",
      "2 — faithful to the salient content"]),
    ("fluency", "Fluency (content coherence — see rubric's pivot caveat)",
     ["0 — broken: repetition loops, truncation, incoherent",
      "1 — understandable but awkward or confused",
      "2 — natural and coherent"]),
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CICIL Human Evaluation — Stage 1 (English-assisted)</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background: #f5f5f4; color: #1c1917; }}
  .wrap {{ max-width: 880px; margin: 0 auto; padding: 16px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  img.eval {{ max-width: 100%; max-height: 420px; display: block; margin: 0 auto; border-radius: 6px; }}
  .imgerr {{ background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 14px; border-radius: 6px; }}
  h1 {{ font-size: 1.15rem; }} h2 {{ font-size: 1rem; margin: 4px 0 8px; }}
  .cap {{ margin: 6px 0; padding: 10px 12px; border-radius: 6px; line-height: 1.45; }}
  .cap-es {{ background: #f8fafc; border: 1px solid #e2e8f0; }}
  .cap-en {{ background: #eff6ff; border: 1px solid #bfdbfe; }}
  .cap-tg {{ background: #fafaf9; border: 1px dashed #d6d3d1; color: #78716c; font-size: .9rem; }}
  .lbl {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: #64748b; display: block; margin-bottom: 3px; }}
  .dim {{ margin: 8px 0 12px; }}
  .dim b {{ display: block; margin-bottom: 4px; }}
  .dim label {{ display: block; font-size: .88rem; margin: 2px 0 2px 6px; cursor: pointer; }}
  .arm {{ border-top: 2px solid #e7e5e4; padding-top: 10px; margin-top: 14px; }}
  textarea {{ width: 100%; min-height: 70px; box-sizing: border-box; font: inherit; padding: 8px; border-radius: 6px; border: 1px solid #d6d3d1; }}
  .nav {{ display: flex; gap: 10px; align-items: center; justify-content: space-between; }}
  button {{ font: inherit; padding: 8px 18px; border-radius: 6px; border: 1px solid #a8a29e; background: #fff; cursor: pointer; }}
  button.primary {{ background: #1d4ed8; border-color: #1d4ed8; color: #fff; }}
  .progress {{ font-variant-numeric: tabular-nums; color: #57534e; }}
  .warn {{ background: #fffbeb; border: 1px solid #fde68a; padding: 10px 12px; border-radius: 6px; margin-top: 10px; }}
  .done {{ color: #15803d; }} .todo {{ color: #b45309; }}
  #gate input {{ font: inherit; padding: 8px; border-radius: 6px; border: 1px solid #d6d3d1; width: 240px; }}
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
  const r = load(sid);
  return DIMS.every(d => r["A_" + d[0]] !== undefined && r["B_" + d[0]] !== undefined)
         && r["preference_A_B_tie"] !== undefined;
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
  document.getElementById("app").innerHTML = `
    <div class="card nav">
      <button onclick="go(-1)" ${{idx === 0 ? "disabled" : ""}}>&larr; Prev</button>
      <span class="progress">Item ${{idx + 1}}/${{ITEMS.length}} &nbsp;·&nbsp; ${{sid}} (${{it.language}}) &nbsp;·&nbsp; <span id="status">${{statusLine()}}</span></span>
      <button onclick="go(1)" ${{idx === ITEMS.length - 1 ? "disabled" : ""}}>Next &rarr;</button>
    </div>
    <div class="card">
      <img class="eval" src="${{it.image_src}}"
           onerror="this.outerHTML='<div class=imgerr>Image not found at <code>${{it.image_src}}</code>.<br>The dataset must be checked out at <code>data/americasnlp2026/</code> in this repo (it is gitignored — download it separately).</div>'">
    </div>
    <div class="card">
      ${{capBlock("A")}}
      ${{capBlock("B")}}
      <div class="arm"><h2>Preference — better overall description of this image</h2>${{pref}}</div>
      <div class="arm"><h2>Notes (cultural errors, hallucinations, category hits/misses)</h2>
        <textarea onchange="setField('${{sid}}','notes',this.value)">${{esc(saved["notes"] || "")}}</textarea></div>
    </div>
    <div class="card nav">
      <span>Autosaves locally as you go (per annotator name).</span>
      <button class="primary" onclick="exportCSV()">Export results CSV</button>
    </div>`;
}}

function go(delta) {{ idx = Math.max(0, Math.min(ITEMS.length - 1, idx + delta)); render(); }}

function csvQuote(v) {{
  v = String(v ?? "");
  return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}}

function exportCSV() {{
  const incomplete = ITEMS.filter(it => !isComplete(it.sample_id)).map(it => it.sample_id);
  if (incomplete.length &&
      !confirm("Not fully scored yet: " + incomplete.join(", ") + "\\nExport anyway?")) return;
  const cols = ["annotator", "sample_id", "language",
                ...DIMS.flatMap(d => ["A_" + d[0], "B_" + d[0]]),
                "preference_A_B_tie", "notes"];
  const lines = [cols.join(",")];
  for (const it of ITEMS) {{
    const r = load(it.sample_id);
    lines.push(cols.map(c =>
      csvQuote(c === "annotator" ? annotator :
               c === "sample_id" ? it.sample_id :
               c === "language" ? it.language : r[c])).join(","));
  }}
  const blob = new Blob([lines.join("\\n") + "\\n"], {{ type: "text/csv;charset=utf-8" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `human_eval_results_${{annotator.replace(/\\s+/g, "_")}}.csv`;
  a.click();
  alert("Saved. Move the downloaded file into analysis/human_eval/results/ and commit it.");
}}
</script>
</body>
</html>
"""


def main() -> None:
    items = load_items()
    html = HTML_TEMPLATE.format(
        items_json=json.dumps(items, ensure_ascii=False),
        dims_json=json.dumps(DIMENSIONS, ensure_ascii=False),
    )
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_FILE} ({len(items)} items). Open it in a browser to annotate.")


if __name__ == "__main__":
    main()
