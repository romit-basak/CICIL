"""RQ3 harness: ChrF++ broken down by cultural category.

RQ3 asks *which cultural categories are hardest to ground*. Stage 1's cultural-VQA
module answers four category questions per image (ceremony, material culture,
landscape, kinship) and stores them in each record's ``cultural_annotations``. This
module (1) labels, per image, which categories are actually *present* (the VQA often
answers "no evidence of a ceremony"), (2) scores each dev caption with ChrF++, and
(3) reports mean ChrF++ per category — so we can see where cultural grounding helps.

Because a category label is a judgement call, the automatic labels are a **first
pass** (a negation-cue heuristic) meant to be audited: the harness writes a
per-image labeling CSV, and if a human-corrected override CSV is supplied it takes
precedence. Nandita owns the taxonomy, so this is deliberately a scaffold she can
correct rather than a black box.

Two-mode operation, so it is useful *now* even though the gold references live under
``data/`` (gitignored):
  * **Labeling + distribution** — always runs (needs only ``outputs/``): writes the
    per-image labels and a category-presence figure.
  * **Per-category ChrF++** — runs when ``data/`` is mounted (gold refs) and
    ``sacrebleu`` is installed: adds the score columns and the category×language
    heatmap. Reuses ``src.stage1.evaluate.chrfpp`` (the official scorer) unchanged.

Run:
  python -m analysis.rq3_category --lang guarani
  python -m analysis.rq3_category --all
  python -m analysis.rq3_category --lang guarani --override analysis/rq3_labels_guarani.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PREDICTIONS = ROOT / "predictions"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"

CATEGORIES = ["ceremony", "material_culture", "landscape", "kinship"]
CATEGORY_LABEL = {
    "ceremony": "Ceremony",
    "material_culture": "Material culture",
    "landscape": "Landscape",
    "kinship": "Kinship",
}

# lang stem -> display name (matches results_data / the paper)
LANGS = {
    "guarani": "Guaraní",
    "maya": "Yucatec Maya",
    "wixarika": "Wixárika",
    "nahuatl": "Nahuatl",
    "bribri": "Bribri",
}

# Spanish phrases the VQA uses when it finds NO evidence for a category. If any of
# these appears in an annotation, we treat that category as *absent* for the image.
# Intentionally conservative (strong "no evidence" phrasings only) to avoid marking
# a hedged-but-descriptive answer as absent; edge cases are what the override is for.
_ABSENCE_PATTERNS = [
    r"no hay evidencia",
    r"no hay ninguna",
    r"no hay (personas|vestiment\w*|instrumento\w*|elementos|informaci\w*)",
    r"no se (observ\w+|aprecia\w*|puede determinar|identifica\w*|ve\b|ven\b)",
    r"no son visibles",
    r"no (es|son) visible\w*",
    r"sin evidencia",
    r"no proporciona informaci",
    r"no hay indicios",
]
_ABSENCE_RE = re.compile("|".join(_ABSENCE_PATTERNS))


def _norm(text: str) -> str:
    """Lowercase + strip accents so patterns match regardless of accenting."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def category_present(annotation: str) -> tuple[bool, list[str]]:
    """Return (present?, matched-absence-cues) for one category's annotation text."""
    if not annotation or len(annotation.strip()) < 15:
        return False, ["<empty/too-short>"]
    cues = _ABSENCE_RE.findall(_norm(annotation))
    # findall on an alternation returns tuple groups; flatten to the matched text.
    flat = [c if isinstance(c, str) else next((g for g in c if g), "") for c in cues]
    matched = sorted({m for m in flat if m})
    return (len(matched) == 0), matched


# --- IO ----------------------------------------------------------------------

def load_cultural_records(lang: str) -> list[dict]:
    path = OUTPUTS / f"{lang}_dev_cultural-vqa_ollama.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"cultural-VQA outputs not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_predictions(lang: str) -> list[str]:
    path = PREDICTIONS / f"{lang}_cultural-vqa_k5_predictions.txt"
    if not path.exists():
        raise FileNotFoundError(f"predictions not found: {path}")
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def load_references(lang: str) -> list[str] | None:
    """Gold dev captions via the shared loader; None if data/ is not mounted."""
    try:
        from src.stage1.data_io import load_split  # noqa: PLC0415
        return [ex.target_caption or "" for ex in load_split(lang, "dev")]
    except Exception as exc:  # FileNotFoundError (no data/) or import error
        print(f"  [refs unavailable for {lang}: {type(exc).__name__}] "
              "-> labeling only, ChrF++ skipped. Mount data/ to score.")
        return None


def load_override(path: Path | None) -> dict[str, dict[str, bool]]:
    """Optional human-corrected labels: CSV with id + one 0/1 column per category."""
    if not path:
        return {}
    labels: dict[str, dict[str, bool]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = {c: str(row.get(c, "")).strip() in {"1", "true", "True"}
                                 for c in CATEGORIES}
    print(f"  applied override labels for {len(labels)} images from {path.name}")
    return labels


# --- Core --------------------------------------------------------------------

def analyze_language(lang: str, override: Path | None = None) -> dict:
    records = load_cultural_records(lang)
    preds = load_predictions(lang)
    refs = load_references(lang)
    overrides = load_override(override)

    if len(preds) != len(records):
        raise ValueError(f"{lang}: {len(preds)} preds vs {len(records)} records")
    if refs is not None and len(refs) != len(records):
        raise ValueError(f"{lang}: {len(refs)} refs vs {len(records)} records")

    chrfpp = None
    if refs is not None:
        from src.stage1.evaluate import chrfpp as _chrfpp  # official scorer
        chrfpp = _chrfpp

    rows = []
    for i, rec in enumerate(records):
        annots = rec.get("cultural_annotations", {}) or {}
        present, cues = {}, {}
        for cat in CATEGORIES:
            if rec["id"] in overrides:
                present[cat] = overrides[rec["id"]][cat]
                cues[cat] = ["<override>"]
            else:
                p, c = category_present(annots.get(cat, ""))
                present[cat], cues[cat] = p, c
        score = chrfpp(preds[i], refs[i]) if chrfpp else None
        rows.append({"id": rec["id"], "present": present, "cues": cues, "chrfpp": score})

    return {"lang": lang, "rows": rows, "scored": refs is not None}


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Per-category counts and (if scored) mean ChrF++ over present-category images."""
    summary = {}
    for cat in CATEGORIES:
        present_rows = [r for r in rows if r["present"][cat]]
        scores = [r["chrfpp"] for r in present_rows if r["chrfpp"] is not None]
        summary[cat] = {
            "n_present": len(present_rows),
            "n_total": len(rows),
            "mean_chrfpp": round(sum(scores) / len(scores), 2) if scores else None,
        }
    return summary


# --- Outputs -----------------------------------------------------------------

def write_labels_csv(lang: str, rows: list[dict]) -> Path:
    """Auditable per-image labels — edit the 0/1 cells and feed back via --override."""
    path = OUT_DIR / f"rq3_labels_{lang}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", *CATEGORIES, "chrfpp",
                    *[f"{c}_absence_cues" for c in CATEGORIES]])
        for r in rows:
            w.writerow([
                r["id"],
                *[int(r["present"][c]) for c in CATEGORIES],
                "" if r["chrfpp"] is None else f"{r['chrfpp']:.2f}",
                *["; ".join(r["cues"][c]) for c in CATEGORIES],
            ])
    return path


def write_summary_csv(results: list[dict]) -> Path:
    path = OUT_DIR / "rq3_category_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "category", "n_present", "n_total", "mean_chrfpp"])
        for res in results:
            summ = summarize(res["rows"])
            for cat in CATEGORIES:
                s = summ[cat]
                w.writerow([LANGS[res["lang"]], CATEGORY_LABEL[cat],
                            s["n_present"], s["n_total"],
                            "" if s["mean_chrfpp"] is None else s["mean_chrfpp"]])
    return path


def plot_presence(results: list[dict]) -> Path:
    """Grouped bars: per category, how many images have it present, by language."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # dataviz categorical slots 1-5 (validated order), one hue per language.
    hues = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#e34948"]
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})

    langs = [r["lang"] for r in results]
    x = np.arange(len(CATEGORIES))
    width = 0.8 / max(len(langs), 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for j, res in enumerate(results):
        summ = summarize(res["rows"])
        vals = [summ[c]["n_present"] for c in CATEGORIES]
        xpos = x + (j - (len(langs) - 1) / 2) * width
        ax.bar(xpos, vals, width, color=hues[j % len(hues)], zorder=3,
               label=LANGS[res["lang"]])
        for xi, v in zip(xpos, vals):
            ax.text(xi, v + 0.4, str(v), ha="center", va="bottom",
                    fontsize=6.2, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABEL[c] for c in CATEGORIES], color=INK)
    ax.set_ylabel("# dev images with category present", color=INK)
    ax.set_title("Cultural-category presence by language (Stage 1 VQA)",
                 fontsize=10.5, color=INK, loc="left", pad=22)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=5,
              bbox_to_anchor=(1.0, 1.09), columnspacing=1.0, handlelength=1.0)
    fig.subplots_adjust(bottom=0.12, top=0.85, left=0.08, right=0.98)

    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "rq3_category_presence.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scores_heatmap(results: list[dict]) -> Path | None:
    """category×language heatmap of mean ChrF++ (only when scored)."""
    scored = [r for r in results if r["scored"]]
    if not scored:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    langs = [LANGS[r["lang"]] for r in scored]
    data = np.array([[summarize(r["rows"])[c]["mean_chrfpp"] or np.nan
                      for r in scored] for c in CATEGORIES])
    fig, ax = plt.subplots(figsize=(1.4 + 1.1 * len(langs), 3.4))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(langs)), langs)
    ax.set_yticks(range(len(CATEGORIES)), [CATEGORY_LABEL[c] for c in CATEGORIES])
    for i in range(len(CATEGORIES)):
        for j in range(len(langs)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="#0b0b0b")
    ax.set_title("Mean ChrF++ by cultural category", fontsize=10.5,
                 color="#0b0b0b", loc="left", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ChrF++")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "rq3_category_chrf_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="RQ3 per-category ChrF++ harness.")
    ap.add_argument("--lang", choices=list(LANGS))
    ap.add_argument("--all", action="store_true", help="run every language")
    ap.add_argument("--override", type=Path, help="human-corrected labels CSV (one language)")
    args = ap.parse_args()

    if not args.lang and not args.all:
        ap.error("provide --lang <lang> or --all")
    targets = list(LANGS) if args.all else [args.lang]

    results = []
    for lang in targets:
        print(f"\n== {LANGS[lang]} ({lang}) ==")
        res = analyze_language(lang, override=args.override if lang == args.lang else None)
        results.append(res)
        labels_path = write_labels_csv(lang, res["rows"])
        summ = summarize(res["rows"])
        for cat in CATEGORIES:
            s = summ[cat]
            score = "—" if s["mean_chrfpp"] is None else f"{s['mean_chrfpp']:.2f} ChrF++"
            print(f"  {CATEGORY_LABEL[cat]:<16} present in {s['n_present']:>2}/{s['n_total']}"
                  f" images   {score}")
        print(f"  labels -> {labels_path.relative_to(ROOT)}")

    summary_path = write_summary_csv(results)
    presence_fig = plot_presence(results)
    heat = plot_scores_heatmap(results)
    print(f"\nsummary -> {summary_path.relative_to(ROOT)}")
    print(f"figure  -> {presence_fig.relative_to(ROOT)}")
    if heat:
        print(f"figure  -> {heat.relative_to(ROOT)}")
    else:
        print("heatmap -> skipped (no ChrF++ scores; mount data/ to enable)")


if __name__ == "__main__":
    main()
