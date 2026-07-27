"""RAG comparison: per lang x arm metrics for the Stage-1 retrieval work.

Prints, for each (lang, arm):
  * end-to-end ChrF++ (official scorer, needs data/ mounted)
  * culture-NAME rate — output names the culture/region (guaraní, maya, ...)
  * CONCEPT rate — output names a specific cultural concept (ñandutí, cenote,
    Wirikuta, ...) EXCLUDING the bare culture name. This split matters for
    honesty: the v2 RAG prompt states the culture as given, so echoing the
    culture name back is partly trivial; naming the *artifact/site* is not.
  * hedged rate ("posiblemente"/"podría ser") in Spanish outputs
  * degeneration rate in the target-language predictions
plus spot-checks for a fixed id set.

    uv run python -m analysis.rag_pilot
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PREDICTIONS = ROOT / "predictions"

LANGS = ["guarani", "wixarika", "maya", "nahuatl", "bribri"]
# (label, stage1 backend tag). Baseline = distilled smolvlm without RAG.
ARMS = [("smolvlm (no RAG)", "smolvlm"),
        ("smolvlm-rag", "smolvlm-rag"),
        ("smolvlm-ragdistill", "smolvlm-ragdistill"),
        ("vllm-rag (teacher)", "vllm-rag"),
        ("ollama-rag (pilot)", "ollama-rag")]

# Culture/region names — with the v2 culture-as-given prompt, matching these is
# partly prompt echo; report separately from concepts.
CULTURE_NAME = {
    "guarani": r"guaran[ií]|paraguay",
    "wixarika": r"wix[aá]rika|huichol",
    "maya": r"\bmayas?\b|yucatec[oa]?s?\b|yucat[aá]n",
    "nahuatl": r"n[aá]huatl|nahuas?\b|mexicas?\b|aztecas?\b",
    "bribri": r"bribri|talamanca",
}
# Specific cultural concepts/sites/artifacts (never in the prompt).
CONCEPT = {
    "guarani": r"ñandut[íi]|ao po'?i|mate\b|bombilla|terer[eé]|chamam[eé]|chipa|"
               r"tipo[íi]|typ[oó]i|encaje|misiones jesu[ií]ticas",
    "wixarika": r"wirikuta|nierika|chaquira|peyote|h[ií]kuri|venado|xukuri|"
                r"tepari|real de catorce|san luis potos[ií]",
    "maya": r"cenote|huipil|milpa|sacb[eé]|chich[eé]n|itz[aá]|uxmal|"
            r"henequ[eé]n|hanal pixan|cochinita|d[ií]a de muertos",
    "nahuatl": r"tenochtitlan|milpa|temazcal|voladores|amate|nixtamal|"
               r"molcajete|d[ií]a de muertos|huipil|chinampa",
    "bribri": r"cacao|s[ii̠]b[oö]|us[eé]k[aö]l|kek[oö]ldi|k[eè]k[oö]ldi|"
              r"yorkin|york[ií]n|cahuita|puerto viejo",
}
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
    print(f"{'lang':<9} {'arm':<20} {'ChrF++':>7} {'cult-name':>10} "
          f"{'concept':>8} {'hedged':>7} {'degen':>7}")
    for lang in LANGS:
        name_re = re.compile(CULTURE_NAME[lang], re.I)
        conc_re = re.compile(CONCEPT[lang], re.I)
        for label, tag in ARMS:
            jsonl = OUTPUTS / f"{lang}_dev_cultural-vqa_{tag}.jsonl"
            pred = PREDICTIONS / f"{lang}_cultural-vqa_{tag}_k5_predictions.txt"
            if not jsonl.exists():
                continue
            rows = [json.loads(l) for l in jsonl.open(encoding="utf-8")]
            n = len(rows)
            n_name = sum(bool(name_re.search(r["generated_spanish"])) for r in rows)
            n_conc = sum(bool(conc_re.search(r["generated_spanish"])) for r in rows)
            n_hedge = sum(bool(HEDGE.search(r["generated_spanish"])) for r in rows)
            print(f"{lang:<9} {label:<20} {chrf(lang, pred):>7} "
                  f"{n_name:>4}/{n:<3} {n_conc:>4}/{n:<3} "
                  f"{n_hedge:>4}/{n:<3} {degen_rate(pred):>7}")
            for r in rows:
                if r["id"] in SPOT_IDS:
                    spots.append((r["id"], label, r["generated_spanish"]))

    print("\n== Spot checks ==")
    for sid, label, text in sorted(spots):
        print(f"\n[{sid}] {label}:\n  {text[:280]}")


if __name__ == "__main__":
    main()
