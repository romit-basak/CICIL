"""Score the v4.1 pilot quality-vs-rounds curve.

Input: the --out-jsonl records from prototype_agent_rag.py --pilot
(one per image: base, ocr, per-round captions, questioner stop reason, final).
Gold: the wixarika pilot's Spanish reference captions.

Reports, per config file:
  - ChrF++ of the BASE captions (round 0), each round's captions, and the
    finals. Round k uses the round-k caption where the interrogation was
    still running, else the last available caption (an image that stopped
    at round 2 contributes its round-2 caption to rounds 3..10) -- so the
    curve reads as "quality if you had capped at k rounds."
  - Where the questioner chose to stop vs. where the curve plateaus
    (self-stop calibration -- the deployable stopping rule).

Run:
    uv run python -m analysis.human_eval.score_agent_curve \
        outputs/agent_rag_pilot_curve_gemini.jsonl \
        outputs/agent_rag_pilot_curve_local7b.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def caption_at_round(rec: dict, k: int) -> str:
    """Caption if the run had been capped at k rounds (k=0 -> base)."""
    if k == 0:
        return rec["base"]
    available = [r["caption"] for r in rec["rounds"] if r["round"] <= k]
    return available[-1] if available else rec["base"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()

    from sacrebleu.metrics import CHRF

    from src.stage1.data_io import load_split

    gold = {e.id: e.spanish_caption for e in load_split("wixarika", "pilot")
            if e.spanish_caption}
    chrf = CHRF(word_order=2)

    for path in args.files:
        recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        recs = [r for r in recs if r["id"] in gold]
        refs = [[gold[r["id"]] for r in recs]]
        print(f"\n=== {path.name} (n={len(recs)}) ===")

        print(f"{'capped at':<12} {'ChrF++':>7}")
        for k in range(0, args.max_rounds + 1):
            score = chrf.corpus_score([caption_at_round(r, k) for r in recs], refs).score
            label = "base (0)" if k == 0 else f"round {k}"
            print(f"{label:<12} {score:7.2f}")
        final_score = chrf.corpus_score([r["final"] for r in recs], refs).score
        print(f"{'final':<12} {final_score:7.2f}")

        stops = Counter(len(r["rounds"]) if r["stop_reason"] else "cap"
                        for r in recs)
        print("questioner rounds used (\"cap\" = never stopped voluntarily):")
        for key in sorted(stops, key=str):
            print(f"  {key}: {stops[key]} image(s)")

        n_ocr = sum(1 for r in recs if r["ocr"] not in ("(ninguno)", "(no extraído)"))
        print(f"images with OCR text extracted: {n_ocr}/{len(recs)}")


if __name__ == "__main__":
    main()
