"""Stage 2 — Step 3: score all predictions and print comparison tables.

Runs ``src.stage1.evaluate`` for every prediction file in ``predictions/`` and
prints tables:

  Table 1:  Official baseline vs generic vs cultural-vqa           (k=5, full pipelines)
  Table 2:  RETRIEVAL-ARM ABLATION (RQ1 headline): cultural-query vs
            text-query, both under cultural-vqa mode                (k=5)
  Table 3+: k ablation for each (mode, query-arm) run config        (k=3,5,8)

    uv run python -m src.stage2.run_ablations

Reads files named ``{lang}_{mode}[_{backend}][_{query_arm}query]_k{k}_predictions.txt``
(``translate.pred_filename`` is the single source of truth for this naming scheme),
written by translate.py / run_sweep.py.

Table 1 and its k-ablation tables read the bare, backend/query-arm-default
filename (``backend="ollama"``, ``query_arm="auto"``) -- this is the same
lookup the original prelim results used, kept byte-identical on purpose:
analysis/rq3_category.py and analysis/human_eval/build_sample.py both hardcode
this exact filename with no fallback. Table 2 and its k-ablation tables are a
SEPARATE lookup keyed on the new query-arm-tagged files, so an ablation run
that hasn't been re-run yet shows "--" there without ever touching (or
appearing to invalidate) the already-reported Table 1 numbers.
"""

from __future__ import annotations

from src.stage1.evaluate import score_translations
from .paths import PRED_DIR
from .translate import pred_filename

# Official MT baselines (README / eval.py sentence-mean numbers; see the handoff doc).
OFFICIAL_BASELINES = {
    "guarani": 20.82,
    "wixarika": 17.77,
    "nahuatl": 11.53,
    "bribri": 7.57,
    "maya": None,  # no MT baseline exists
}

LANGUAGES = ["guarani", "bribri", "maya", "wixarika", "nahuatl"]
MODES = ["generic", "cultural-vqa"]
K_VALUES = [3, 5, 8]

# (mode, query_arm) run configs actually produced by translate.py / run_sweep.py's
# query-arm ablation. Keep in sync with run_sweep.RUN_CONFIGS. generic+cultural is
# intentionally absent -- generic-mode records carry no cultural_annotations, so
# it would just be a duplicate of generic+text.
RUN_CONFIGS: list[tuple[str, str]] = [
    ("generic", "text"),
    ("cultural-vqa", "cultural"),
    ("cultural-vqa", "text"),
]
CONFIG_LABELS = {
    ("generic", "text"): "Generic",
    ("cultural-vqa", "cultural"): "Cultural-VQA (cultural query)",
    ("cultural-vqa", "text"): "Cultural-VQA (text query)",
}


def pred_path(lang: str, mode: str, k: int, *, backend: str = "ollama",
              query_arm: str = "auto"):
    return PRED_DIR / pred_filename(lang, mode, k, backend=backend, query_arm=query_arm)


def score_file(lang: str, pred_file) -> float | None:
    """Return the mean ChrF++ for a predictions file, or None on failure.

    Calls the official scorer directly (src.stage1.evaluate.score_translations)
    rather than parsing stdout — avoids scraping the wrong number.
    """
    if not pred_file.exists():
        return None
    try:
        mean, _per_line = score_translations(lang, pred_file, split="dev")
        return mean
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: scoring failed for {pred_file.name}: {e}")
        return None


def fmt(score) -> str:
    return "  --  " if score is None else f"{score:.2f}"


def main():
    print("\nScoring all prediction files in predictions/ ...\n")

    # Legacy lookup (bare filenames, backend="ollama"/query_arm="auto") -- feeds
    # Table 1 and the original two k-ablation tables. Unchanged from before this
    # merge so today's already-reported numbers keep reading from the exact same
    # files.
    legacy_scores = {lang: {mode: {} for mode in MODES} for lang in LANGUAGES}
    for lang in LANGUAGES:
        for mode in MODES:
            for k in K_VALUES:
                pred_file = pred_path(lang, mode, k)
                if not pred_file.exists():
                    continue
                print(f"  Scoring {pred_file.name} ...")
                legacy_scores[lang][mode][k] = score_file(lang, pred_file)
                print(f"    ChrF++ = {fmt(legacy_scores[lang][mode][k])}")

    # New query-arm lookup -- feeds Table 2 (headline) and its own k-ablation
    # tables. Separate dict/files on purpose: see module docstring.
    arm_scores: dict = {lang: {cfg: {} for cfg in RUN_CONFIGS} for lang in LANGUAGES}
    for lang in LANGUAGES:
        for cfg in RUN_CONFIGS:
            mode, query_arm = cfg
            for k in K_VALUES:
                pred_file = pred_path(lang, mode, k, query_arm=query_arm)
                if not pred_file.exists():
                    continue
                print(f"  Scoring {pred_file.name} ...")
                arm_scores[lang][cfg][k] = score_file(lang, pred_file)
                print(f"    ChrF++ = {fmt(arm_scores[lang][cfg][k])}")

    print("\n\n" + "=" * 70)
    print("TABLE 1: Official baseline vs Generic vs Cultural-VQA  (k=5)")
    print("=" * 70)
    header = f"{'Language':<14}  {'Official':>10}  {'Generic k=5':>12}  {'Cultural k=5':>13}  {'Delta':>7}"
    print(header)
    print("-" * len(header))
    for lang in LANGUAGES:
        generic = legacy_scores[lang]["generic"].get(5)
        cultural = legacy_scores[lang]["cultural-vqa"].get(5)
        delta = f"{cultural - generic:+.2f}" if generic is not None and cultural is not None else ""
        print(f"{lang:<14}  {fmt(OFFICIAL_BASELINES.get(lang)):>10}  "
              f"{fmt(generic):>12}  {fmt(cultural):>13}  {delta:>7}")

    generic_cfg = ("generic", "text")
    cultural_cfg = ("cultural-vqa", "cultural")
    text_query_cfg = ("cultural-vqa", "text")

    print("\n\n" + "=" * 78)
    print("TABLE 2: RETRIEVAL-ARM ABLATION (RQ1 headline), k=5")
    print("Does culturally-indexed retrieval beat vanilla text retrieval?")
    print("Mode fixed at cultural-vqa; only the retrieval QUERY changes.")
    print("=" * 78)
    header2 = f"{'Language':<14}  {'Cultural query':>15}  {'Text query':>12}  {'Delta':>7}"
    print(header2)
    print("-" * len(header2))
    for lang in LANGUAGES:
        cultural_q = arm_scores[lang][cultural_cfg].get(5)
        text_q = arm_scores[lang][text_query_cfg].get(5)
        delta = f"{cultural_q - text_q:+.2f}" if cultural_q is not None and text_q is not None else ""
        print(f"{lang:<14}  {fmt(cultural_q):>15}  {fmt(text_q):>12}  {delta:>7}")
    print(
        "\n(Positive delta = culturally-indexed retrieval beats vanilla text "
        "retrieval, holding prompt mode and k fixed.)"
    )

    for title, mode in [("Generic arm", "generic"), ("Cultural-VQA arm", "cultural-vqa")]:
        print("\n\n" + "=" * 53)
        print(f"TABLE: Retrieval depth ablation -- {title}")
        print("=" * 53)
        h = f"{'Language':<14}  {'k=3':>8}  {'k=5':>8}  {'k=8':>8}"
        print(h)
        print("-" * len(h))
        for lang in LANGUAGES:
            row = f"{lang:<14}"
            for k in K_VALUES:
                row += f"  {fmt(legacy_scores[lang][mode].get(k)):>8}"
            print(row)

    for cfg in RUN_CONFIGS:
        label = CONFIG_LABELS[cfg]
        print("\n\n" + "=" * 53)
        print(f"TABLE: Retrieval depth ablation -- {label}")
        print("=" * 53)
        h = f"{'Language':<14}  {'k=3':>8}  {'k=5':>8}  {'k=8':>8}"
        print(h)
        print("-" * len(h))
        for lang in LANGUAGES:
            row = f"{lang:<14}"
            for k in K_VALUES:
                row += f"  {fmt(arm_scores[lang][cfg].get(k)):>8}"
            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
