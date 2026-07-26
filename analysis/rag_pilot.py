"""RAG pilot comparison: per lang x arm metrics for the Stage-1 retrieval pilot.

Prints, for each (lang, arm):
  * end-to-end ChrF++ (official scorer, needs data/ mounted)
  * culture/artifact-term rate in the Spanish Stage-1 outputs
  * hedged-naming rate ("posiblemente" + a concept) in Spanish outputs
  * degeneration rate in the target-language predictions
plus spot-checks for grn_019 / grn_025 / hch_021.

    uv run python -m analysis.rag_pilot
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PREDICTIONS = ROOT / "predictions"

LANGS = ["guarani", "wixarika"]
# (label, stage1 backend tag). Baseline = distilled smolvlm without RAG.
ARMS = [("smolvlm (no RAG)", "smolvlm"),
        ("smolvlm-rag", "smolvlm-rag"),
        ("ollama-rag (teacher)", "ollama-rag")]

CULTURE_TERMS = re.compile(
    r"guaran[ií]|wix[aá]rika|huichol|paraguay|ñandut[íi]|ao po'?i|mate\b|bombilla|"
    r"terer[eé]|chamam[eé]|chipa|wirikuta|nierika|chaquira|peyote|venado|"
    r"tejido|bordado|encaje|artesan[ií]a", re.I)
HEDGE = re.compile(r"posiblemente|podr[ií]a ser", re.I)
SPOT_IDS = {"grn_019", "grn_025", "hch_021"}


def degen_rate(pred_file: Path) -> str:
    if not pred_file.exists():
        return "--"
    lines = pred_file.read_text(encoding="utf-8").splitlines()
    bad = sum(1 for l in lines if len(l.split()) > 10
              and len(set(l.split())) / len(l.split()) < 0.3)
    return f"{bad}/{len(lines)}"


def chrf(lang: str, pred_file: Path) -> str:
    if not pred_file.exists():
        return "--"
    from src.stage1.evaluate import score_translations
    try:
        mean, _ = score_translations(lang, pred_file, split="dev")
        return f"{mean:.2f}"
    except Exception:  # noqa: BLE001 - data/ may be unmounted
        return "n/a"


def main() -> None:
    spots = []
    print(f"{'lang':<10} {'arm':<22} {'ChrF++':>7} {'cult-term':>10} "
          f"{'hedged':>7} {'degen':>7}")
    for lang in LANGS:
        for label, tag in ARMS:
            jsonl = OUTPUTS / f"{lang}_dev_cultural-vqa_{tag}.jsonl"
            pred = PREDICTIONS / f"{lang}_cultural-vqa_{tag}_k5_predictions.txt"
            if not jsonl.exists():
                print(f"{lang:<10} {label:<22} (no stage-1 output yet)")
                continue
            rows = [json.loads(l) for l in jsonl.open(encoding="utf-8")]
            n = len(rows)
            n_cult = sum(bool(CULTURE_TERMS.search(r["generated_spanish"])) for r in rows)
            n_hedge = sum(bool(HEDGE.search(r["generated_spanish"])) for r in rows)
            print(f"{lang:<10} {label:<22} {chrf(lang, pred):>7} "
                  f"{n_cult:>4}/{n:<3} {n_hedge:>5}/{n:<3} {degen_rate(pred):>7}")
            for r in rows:
                if r["id"] in SPOT_IDS:
                    spots.append((r["id"], label, r["generated_spanish"]))

    print("\n== Spot checks ==")
    for sid, label, text in sorted(spots):
        print(f"\n[{sid}] {label}:\n  {text[:280]}")


if __name__ == "__main__":
    main()
