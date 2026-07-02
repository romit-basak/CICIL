"""ChrF++ evaluation, matching the official AmericasNLP 2026 scorer.

Reproduces ``baseline/eval.py`` exactly: sacrebleu ``CHRF(word_order=2)``
scored per sentence and averaged, with predictions aligned to references by
line order. Also surfaces the official baseline / leaderboard numbers from the
shared-task results CSVs.

This is a shared utility: point ``score_translations`` at any file of
target-language predictions (one per line, in dev-JSONL order) — including
Stage 2 outputs — to get a number comparable to the official leaderboard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sacrebleu.metrics import CHRF

from . import config
from .data_io import load_split

# word_order=2 == ChrF++ (character F-score + word bigrams), the official config.
_CHRF = CHRF(word_order=2)

RESULTS_DIR = config.DATASET_ROOT.parent / "results" / "automatic_metric"


def chrfpp(hypothesis: str, reference: str) -> float:
    """Sentence-level ChrF++ score (identical call to baseline/eval.py)."""
    return _CHRF.sentence_score(hypothesis, [reference]).score


def _read_lines(path: Path, expected: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").split("\n")
    # Drop a single trailing empty line from a final newline (off-by-one guard).
    if lines and lines[-1] == "" and len(lines) == expected + 1:
        lines = lines[:-1]
    return lines


def score_translations(lang: str, translations_txt: str | Path,
                        split: str = "dev") -> tuple[float, list[float]]:
    """Mean ChrF++ of a predictions file vs. dev references (line-aligned)."""
    refs = [ex.target_caption or "" for ex in load_split(lang, split)]
    hyps = _read_lines(Path(translations_txt), len(refs))
    if len(hyps) != len(refs):
        raise ValueError(
            f"Line/reference mismatch for {lang}/{split}: "
            f"{len(hyps)} predictions vs {len(refs)} references."
        )
    scores = [chrfpp(h, r) for h, r in zip(hyps, refs)]
    return sum(scores) / len(scores), scores


def official_baseline_table() -> tuple[dict[str, float | None], dict[str, tuple[str, float]]]:
    """Return {lang: baseline_chrf} and {lang: (top_team, top_chrf)} from the CSVs."""
    import pandas as pd

    baseline: dict[str, float | None] = {}
    top: dict[str, tuple[str, float]] = {}
    for lang in config.LANGUAGES:
        csv = RESULTS_DIR / f"{lang}_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        base_rows = df[df["team"] == "baseline"]
        baseline[lang] = float(base_rows["chrf"].iloc[0]) if len(base_rows) else None
        best = df.loc[df["chrf"].idxmax()]
        top[lang] = (str(best["team"]), float(best["chrf"]))
    return baseline, top


def _print_official() -> None:
    baseline, top = official_baseline_table()
    print(f"{'language':<10} {'baseline chrF++':>16} {'best system':>28}")
    print("-" * 56)
    for lang in config.LANGUAGES:
        b = baseline.get(lang)
        b_str = f"{b:.2f}" if b is not None else "—"
        if lang in top:
            team, score = top[lang]
            top_str = f"{team} ({score:.2f})"
        else:
            top_str = "—"
        print(f"{lang:<10} {b_str:>16} {top_str:>28}")
    print("\nSource: data/americasnlp2026/results/automatic_metric/*_results.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="ChrF++ evaluation (official-compatible).")
    ap.add_argument("--official", action="store_true",
                    help="Print the official baseline + best-system table and exit.")
    ap.add_argument("--lang", choices=config.LANGUAGES)
    ap.add_argument("--split", default="dev", choices=["pilot", "dev", "test"])
    ap.add_argument("--translations", help="File of target-language predictions (one per line).")
    args = ap.parse_args()

    if args.official:
        _print_official()
        return
    if not (args.lang and args.translations):
        ap.error("provide --official, or both --lang and --translations")

    mean, scores = score_translations(args.lang, args.translations, args.split)
    print(f"{args.lang}/{args.split}: {len(scores)} items, mean chrF++ = {mean:.2f}")


if __name__ == "__main__":
    main()
