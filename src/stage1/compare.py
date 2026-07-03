"""Compare generic vs cultural-VQA Stage 1 outputs — does the contribution help?

Stage 1's novel contribution is cultural-VQA prompting. But its output is Spanish,
which we *can* audit even though nobody on the team reads the target languages. This
module quantifies whether cultural-VQA actually beats the generic baseline, so we
learn it before the end-to-end ChrF++ (and before the paper claims a win).

Two kinds of signal:
  * Reference-free diagnostics (always): length bloat, self-contradiction on object
    material, cultural leakage (right culture vs wrong culture keywords), and how far
    cultural-VQA rewrites the plain baseline. Runs on any split.
  * Reference-based ChrF++ (pilot only): mean ChrF++ of each mode's Spanish against the
    pilot's gold Spanish caption, plus the cultural-minus-generic delta. This is the
    decision-relevant number; it needs Spanish references, which exist only in pilot.

Usage:
  python -m src.stage1.compare --lang wixarika --split pilot
  python -m src.stage1.compare --lang guarani  --split dev
  python -m src.stage1.compare --lang guarani  --split dev \
      --generic outputs/a.jsonl --cultural outputs/b.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

from . import config
from .data_io import load_split
from .evaluate import chrfpp

# Distinct material stems; >1 present in one description flags self-contradiction
# (grn_001's cultural-VQA answer claimed ceramic + clay + leather + wood at once).
MATERIAL_STEMS = [
    "ceramic", "barro", "arcilla", "cuero", "mader", "metal",
    "plastic", "mimbre", "tela", "textil", "vidrio",
]

# Per-language cultural keyword stems: on-target (the actual indigenous culture) vs
# off-target (neighbouring mestizo/national cultures the VLM tends to default to).
# An off-target hit with zero on-target hits is a cultural mis-grounding flag.
CULTURE_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "guarani": {"on": ["guarani", "paraguay"],
                "off": ["argentin", "uruguay", "brasil", "brazil"]},
    "wixarika": {"on": ["wixarika", "huichol", "mexic", "nayarit", "jalisco"],
                 "off": ["argentin", "uruguay", "brasil", "brazil", "paraguay"]},
    "bribri": {"on": ["bribri", "costa rica", "costarric", "talamanca"],
               "off": ["argentin", "uruguay", "brasil", "brazil", "mexic"]},
    "maya": {"on": ["maya", "yucatec", "yucatan", "mexic"],
             "off": ["argentin", "uruguay", "brasil", "brazil", "paraguay"]},
    "nahuatl": {"on": ["nahuatl", "nahua", "azteca", "mexic"],
                "off": ["argentin", "uruguay", "brasil", "brazil", "paraguay"]},
}


def _norm(text: str) -> str:
    """Lowercase and strip diacritics so keyword matching is accent-insensitive."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _tokens(text: str) -> int:
    return len(text.split())


def _distinct_materials(text: str) -> int:
    norm = _norm(text)
    return sum(1 for stem in MATERIAL_STEMS if re.search(rf"\b{stem}", norm))


def _culture_hits(text: str, lang: str) -> tuple[int, int]:
    """Return (on_target_hits, off_target_hits) for the language's keyword sets."""
    kw = CULTURE_KEYWORDS.get(lang, {"on": [], "off": []})
    norm = _norm(text)
    on = sum(1 for k in kw["on"] if k in norm)
    off = sum(1 for k in kw["off"] if k in norm)
    return on, off


def _annotation_text(record: dict) -> str:
    """Concatenate all cultural_annotations values (empty for generic mode)."""
    ann = record.get("cultural_annotations") or {}
    return " ".join(v for v in ann.values() if v)


def _locate(lang: str, split: str, mode: str) -> Path | None:
    """Pick the {lang}_{split}_{mode}*.jsonl file with the most records.

    Glob (not exact) because older cultural-vqa files lack the _<backend> suffix;
    most-records prefers the real run (e.g. generic_ollama's 50) over a 1-line smoke file.
    """
    matches = sorted(config.OUTPUT_DIR.glob(f"{lang}_{split}_{mode}*.jsonl"))
    if not matches:
        return None
    return max(matches, key=lambda p: sum(1 for _ in p.open(encoding="utf-8")))


def _load_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def _reference_free(lang: str, generic: dict[str, dict],
                    cultural: dict[str, dict]) -> list[dict]:
    """Per-id diagnostics over ids present in both modes."""
    ids = sorted(set(generic) & set(cultural))
    rows: list[dict] = []
    for _id in ids:
        g_text = generic[_id].get("generated_spanish", "")
        c_text = cultural[_id].get("generated_spanish", "")
        g_on, g_off = _culture_hits(g_text, lang)
        c_on, c_off = _culture_hits(c_text, lang)
        rows.append({
            "id": _id,
            "gen_tokens": _tokens(g_text),
            "cvqa_tokens": _tokens(c_text),
            "annot_tokens": _tokens(_annotation_text(cultural[_id])),
            "gen_materials": _distinct_materials(g_text),
            "cvqa_materials": _distinct_materials(c_text),
            "gen_culture_on": g_on, "gen_culture_off": g_off,
            "cvqa_culture_on": c_on, "cvqa_culture_off": c_off,
            "divergence_chrfpp": round(chrfpp(g_text, c_text), 1),
        })
    return rows


def _print_reference_free(rows: list[dict]) -> None:
    print("\n=== Reference-free diagnostics (ids in both modes) ===")
    if not rows:
        print("  No overlapping ids between generic and cultural-vqa outputs.")
        return
    hdr = (f"{'id':<10} {'tok g→c':>10} {'annot':>6} {'mat g/c':>8} "
           f"{'cult g(on/off)':>15} {'cult c(on/off)':>15} {'diverg':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['id']:<10} "
              f"{r['gen_tokens']:>4}→{r['cvqa_tokens']:<5} "
              f"{r['annot_tokens']:>6} "
              f"{r['gen_materials']}/{r['cvqa_materials']:>6} "
              f"{r['gen_culture_on']}/{r['gen_culture_off']:>13} "
              f"{r['cvqa_culture_on']}/{r['cvqa_culture_off']:>13} "
              f"{r['divergence_chrfpp']:>7}")

    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    cvqa_conflicts = sum(1 for r in rows if r["cvqa_materials"] > 1)
    gen_conflicts = sum(1 for r in rows if r["gen_materials"] > 1)
    cvqa_misground = sum(1 for r in rows
                         if r["cvqa_culture_off"] > 0 and r["cvqa_culture_on"] == 0)
    gen_misground = sum(1 for r in rows
                        if r["gen_culture_off"] > 0 and r["gen_culture_on"] == 0)
    print("-" * len(hdr))
    print(f"  n={n}")
    print(f"  mean tokens        generic={mean('gen_tokens'):.0f}  "
          f"cultural={mean('cvqa_tokens'):.0f}  "
          f"(+ {mean('annot_tokens'):.0f} annotation tokens per image)")
    print(f"  material conflicts (>1 material)   generic={gen_conflicts}/{n}  "
          f"cultural={cvqa_conflicts}/{n}")
    print(f"  cultural mis-grounding (off, no on) generic={gen_misground}/{n}  "
          f"cultural={cvqa_misground}/{n}")
    print(f"  mean divergence (generic vs cultural chrF++) = {mean('divergence_chrfpp'):.1f}"
          "   (lower = larger rewrite)")


def _chrfpp_vs_gold(lang: str, split: str, generic: dict[str, dict],
                    cultural: dict[str, dict]) -> dict[str, dict] | None:
    """Mean ChrF++ of each mode's Spanish vs gold Spanish, if references exist."""
    gold = {ex.id: ex.spanish_caption
            for ex in load_split(lang, split) if ex.spanish_caption}
    if not gold:
        print(f"\n=== Reference-based ChrF++ ===\n"
              f"  Skipped: no Spanish gold captions for {lang}/{split} "
              f"(only the pilot split has them).")
        return None

    print("\n=== Reference-based ChrF++ vs gold Spanish ===")
    results: dict[str, dict] = {}
    for mode, records in (("generic", generic), ("cultural-vqa", cultural)):
        ids = sorted(set(gold) & set(records))
        scores = {_id: chrfpp(records[_id].get("generated_spanish", ""), gold[_id])
                  for _id in ids}
        empty = sum(1 for _id in ids if not records[_id].get("generated_spanish", "").strip())
        mean = sum(scores.values()) / len(scores) if scores else 0.0
        results[mode] = {"mean": mean, "n": len(scores), "empty": empty, "scores": scores}
        flag = f"  [!] {empty}/{len(scores)} EMPTY outputs" if empty else ""
        print(f"  {mode:<12} n={len(scores):<3} mean chrF++ = {mean:.2f}{flag}")

    # A verdict is only meaningful if both modes actually produced text. Empty
    # outputs (e.g. a generation crash) score ~0 and would fake a large delta.
    max_empty_frac = max(
        (r["empty"] / r["n"]) if r["n"] else 1.0 for r in results.values()
    )
    if results["generic"]["scores"] and results["cultural-vqa"]["scores"]:
        delta = results["cultural-vqa"]["mean"] - results["generic"]["mean"]
        if max_empty_frac > 0.1:
            print(f"  delta (cultural − generic) = {delta:+.2f}   → NO VERDICT: "
                  f"a mode has too many empty outputs ({max_empty_frac:.0%}); "
                  "fix generation and re-run before trusting this number.")
        else:
            # A dead-band avoids reporting a sub-point difference as a win/loss;
            # ChrF++ swings this much from noise even under deterministic decoding.
            if abs(delta) < 0.5:
                verdict = "no meaningful difference (parity)"
            elif delta > 0:
                verdict = "cultural-VQA helps"
            else:
                verdict = "cultural-VQA hurts"
            print(f"  delta (cultural − generic) = {delta:+.2f}   → {verdict}")
    return results


def _write_csv(path: Path, rows: list[dict], chrf: dict[str, dict] | None) -> None:
    if not rows:
        return
    gen_scores = chrf["generic"]["scores"] if chrf else {}
    cvqa_scores = chrf["cultural-vqa"]["scores"] if chrf else {}
    fields = list(rows[0].keys()) + ["gen_chrfpp_gold", "cvqa_chrfpp_gold"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["gen_chrfpp_gold"] = round(gen_scores[r["id"]], 1) if r["id"] in gen_scores else ""
            r["cvqa_chrfpp_gold"] = round(cvqa_scores[r["id"]], 1) if r["id"] in cvqa_scores else ""
            w.writerow(r)
    print(f"\nWrote per-id CSV -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare generic vs cultural-VQA Stage 1 outputs.")
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--split", default="dev", choices=["pilot", "dev", "test"])
    ap.add_argument("--generic", help="Explicit generic JSONL (default: auto-locate).")
    ap.add_argument("--cultural", help="Explicit cultural-vqa JSONL (default: auto-locate).")
    args = ap.parse_args()

    gen_path = Path(args.generic) if args.generic else _locate(args.lang, args.split, "generic")
    cvqa_path = Path(args.cultural) if args.cultural else _locate(args.lang, args.split, "cultural-vqa")
    if gen_path is None or cvqa_path is None:
        ap.error(f"Missing outputs for {args.lang}/{args.split}: "
                 f"generic={gen_path}, cultural-vqa={cvqa_path}. "
                 "Generate them first or pass --generic/--cultural.")

    print(f"generic     : {gen_path}")
    print(f"cultural-vqa: {cvqa_path}")
    generic = _load_records(gen_path)
    cultural = _load_records(cvqa_path)

    rows = _reference_free(args.lang, generic, cultural)
    _print_reference_free(rows)
    chrf = _chrfpp_vs_gold(args.lang, args.split, generic, cultural)

    out_csv = config.OUTPUT_DIR / f"stage1_comparison_{args.lang}_{args.split}.csv"
    _write_csv(out_csv, rows, chrf)


if __name__ == "__main__":
    main()
