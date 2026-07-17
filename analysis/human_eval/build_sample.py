"""Draw a stratified, blinded sample for human evaluation.

Produces a **30-caption** sample per sheet: 5 languages x 3 images x 2 arms
(generic vs cultural-VQA). For each image both captions are shown as a blinded
**A/B** pair (order randomized, arm hidden) so annotators can't favour an arm they
recognise; the A/B->arm mapping is written to a separate key file used only at
analysis time.

Stratification reuses the RQ3 category labels (``analysis.rq3_category``): within
each language the 3 images are chosen greedily to cover as many cultural categories
(ceremony / material culture / landscape / kinship) as possible, so the sample
isn't all easy "landscape" images. Selection is deterministic (fixed seed).

Two sheets, because the team reads Spanish but not the target languages:
  * ``sample_spanish.csv`` — Stage-1 Spanish descriptions (team can annotate now).
  * ``sample_target.csv``  — final target-language captions (native speakers).

Run:  python -m analysis.human_eval.build_sample
      python -m analysis.human_eval.build_sample --per-lang 3 --seed 20260717
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from analysis.rq3_category import (
    CATEGORIES,
    CATEGORY_LABEL,
    LANGS,
    OUTPUTS,
    PREDICTIONS,
    category_present,
)

OUT_DIR = Path(__file__).resolve().parent

# Blank score columns the annotators fill in (see RUBRIC.md for anchors).
SCORE_COLS = [
    "A_cultural_accuracy", "A_faithfulness", "A_fluency",
    "B_cultural_accuracy", "B_faithfulness", "B_fluency",
    "preference_A_B_tie", "notes",
]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_preds(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def _present_set(record: dict) -> set[str]:
    annots = record.get("cultural_annotations", {}) or {}
    return {c for c in CATEGORIES if category_present(annots.get(c, ""))[0]}


def stratified_pick(records: list[dict], n: int, rng: random.Random) -> list[int]:
    """Greedily pick n image indices maximizing cultural-category coverage.

    Each step picks the image covering the most still-uncovered categories (rarer
    categories therefore get in first); ties broken by the seeded RNG. Once all
    categories are covered (or no image adds coverage) the remainder is filled by a
    seeded random draw from the unpicked images.
    """
    present = {i: _present_set(r) for i, r in enumerate(records)}
    picked: list[int] = []
    covered: set[str] = set()
    candidates = list(range(len(records)))
    rng.shuffle(candidates)  # seeded tie-break ordering

    while len(picked) < n and candidates:
        # gain = how many uncovered categories this image would add
        best = max(candidates, key=lambda i: len(present[i] - covered))
        if not (present[best] - covered):
            break  # nothing left to improve coverage; fall through to random fill
        picked.append(best)
        covered |= present[best]
        candidates.remove(best)

    while len(picked) < n and candidates:
        picked.append(candidates.pop())

    return sorted(picked)


def build(per_lang: int, seed: int) -> None:
    rng = random.Random(seed)
    spanish_rows, target_rows, key_rows = [], [], []
    sample_n = 0

    for lang in LANGS:
        gen_recs = _load_jsonl(OUTPUTS / f"{lang}_dev_generic_ollama.jsonl")
        cult_recs = _load_jsonl(OUTPUTS / f"{lang}_dev_cultural-vqa_ollama.jsonl")
        gen_tgt = _load_preds(PREDICTIONS / f"{lang}_generic_k5_predictions.txt")
        cult_tgt = _load_preds(PREDICTIONS / f"{lang}_cultural-vqa_k5_predictions.txt")

        n = min(len(gen_recs), len(cult_recs), len(gen_tgt), len(cult_tgt))
        assert all(len(x) >= n for x in (gen_recs, cult_recs, gen_tgt, cult_tgt))
        # align by id to be safe (files share dev order, but don't assume it)
        gen_by_id = {r["id"]: (i, r) for i, r in enumerate(gen_recs)}

        idxs = stratified_pick(cult_recs[:n], per_lang, rng)
        for ci in idxs:
            crec = cult_recs[ci]
            gi, grec = gen_by_id[crec["id"]]
            sample_n += 1
            sid = f"S{sample_n:03d}"

            # blind: randomly assign which arm is slot A
            gen_is_A = rng.random() < 0.5
            arm_A, arm_B = ("generic", "cultural") if gen_is_A else ("cultural", "generic")

            es = {"generic": grec.get("generated_spanish", ""),
                  "cultural": crec.get("generated_spanish", "")}
            tg = {"generic": gen_tgt[gi], "cultural": cult_tgt[ci]}

            common = {"sample_id": sid, "language": LANGS[lang],
                      "image_filename": crec.get("filename", "")}
            spanish_rows.append({**common, "caption_A": es[arm_A],
                                 "caption_B": es[arm_B], **{c: "" for c in SCORE_COLS}})
            target_rows.append({**common, "caption_A": tg[arm_A],
                                "caption_B": tg[arm_B], **{c: "" for c in SCORE_COLS}})
            cats = sorted(CATEGORY_LABEL[c] for c in _present_set(crec))
            key_rows.append({"sample_id": sid, "language": LANGS[lang],
                             "image_id": crec["id"], "image_filename": crec.get("filename", ""),
                             "slot_A_arm": arm_A, "slot_B_arm": arm_B,
                             "categories_present": "; ".join(cats) or "(none detected)"})

    _write(OUT_DIR / "sample_spanish.csv",
           ["sample_id", "language", "image_filename", "caption_A", "caption_B", *SCORE_COLS],
           spanish_rows)
    _write(OUT_DIR / "sample_target.csv",
           ["sample_id", "language", "image_filename", "caption_A", "caption_B", *SCORE_COLS],
           target_rows)
    _write(OUT_DIR / "sample_key.csv",
           ["sample_id", "language", "image_id", "image_filename",
            "slot_A_arm", "slot_B_arm", "categories_present"],
           key_rows)

    print(f"Sampled {sample_n} images x 2 arms = {2 * sample_n} captions per sheet "
          f"({len(LANGS)} langs x {per_lang}), seed={seed}.")
    # coverage report
    from collections import Counter
    cov = Counter()
    for k in key_rows:
        for c in k["categories_present"].split("; "):
            cov[c] += 1
    print("Category coverage across the sample:")
    for cat in [CATEGORY_LABEL[c] for c in CATEGORIES]:
        print(f"  {cat:<16} {cov.get(cat, 0)} images")
    print(f"\nWrote: sample_spanish.csv, sample_target.csv, sample_key.csv in {OUT_DIR}")
    print("Images referenced by 'image_filename' live under data/dev/<lang>/images/.")


def _write(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the human-eval sample.")
    ap.add_argument("--per-lang", type=int, default=3, help="images per language (x2 arms)")
    ap.add_argument("--seed", type=int, default=20260717, help="deterministic sampling seed")
    args = ap.parse_args()
    build(args.per_lang, args.seed)


if __name__ == "__main__":
    main()
