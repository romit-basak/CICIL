"""Stage 2 — Step 3: score all predictions and print comparison tables.

Runs ``src.stage1.evaluate`` for every prediction file in ``predictions/`` and
prints three tables:

  Table 1: Official baseline vs generic vs cultural-vqa at k=5 (RQ1 headline)
  Table 2: k ablation for the generic arm (k=3, 5, 8)
  Table 3: k ablation for the cultural-vqa arm (k=3, 5, 8)

    uv run python -m src.stage2.run_ablations
"""

from __future__ import annotations

from src.stage1.evaluate import score_translations
from .paths import PRED_DIR

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
    scores = {lang: {mode: {} for mode in MODES} for lang in LANGUAGES}

    for lang in LANGUAGES:
        for mode in MODES:
            for k in K_VALUES:
                pred_file = PRED_DIR / f"{lang}_{mode}_k{k}_predictions.txt"
                if not pred_file.exists():
                    continue
                print(f"  Scoring {pred_file.name} ...")
                scores[lang][mode][k] = score_file(lang, pred_file)
                print(f"    ChrF++ = {fmt(scores[lang][mode][k])}")

    print("\n\n" + "=" * 70)
    print("TABLE 1: Official baseline vs Generic vs Cultural-VQA  (k=5, RQ1)")
    print("=" * 70)
    header = f"{'Language':<14}  {'Official':>10}  {'Generic k=5':>12}  {'Cultural k=5':>13}  {'Delta':>7}"
    print(header)
    print("-" * len(header))
    for lang in LANGUAGES:
        generic = scores[lang]["generic"].get(5)
        cultural = scores[lang]["cultural-vqa"].get(5)
        delta = f"{cultural - generic:+.2f}" if generic is not None and cultural is not None else ""
        print(f"{lang:<14}  {fmt(OFFICIAL_BASELINES.get(lang)):>10}  "
              f"{fmt(generic):>12}  {fmt(cultural):>13}  {delta:>7}")

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
                row += f"  {fmt(scores[lang][mode].get(k)):>8}"
            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
