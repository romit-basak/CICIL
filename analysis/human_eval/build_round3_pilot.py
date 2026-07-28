"""Build the round-3 human-eval sample: the 20-image Wixárika PILOT set.

Unlike dev (no ground truth at all), every pilot image has a real,
human-written Spanish gold caption (``spanish_caption``) -- this round lets
annotators fact-check captions against actual ground truth instead of
guessing context from a language they don't read (the problem that motivated
this round). Scope is necessarily Wixárika-only: it's the only culture with a
pilot split (see DATA_LICENSES.md / STAGE1_HANDOFF.md).

Compares the same two arms as round 2 (smolvlm-rag vs smolvlm-ragdistill) on
ALL 20 pilot images -- not a subset, since 20 is already small. The gold
caption is carried as its own (unblinded, clearly-labeled) column, never
scored, never mixed into the A/B slots.

Run (after generating both arms' Stage 1 pilot outputs and running
translate_pilot_arms.py):
    uv run python -m analysis.human_eval.build_round3_pilot
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from analysis.human_eval.build_sample import SCORE_COLS, _load_jsonl, _load_preds
from analysis.rq3_category import CATEGORY_LABEL, category_present, CATEGORIES

OUT_DIR = Path(__file__).resolve().parent
OUTPUTS = OUT_DIR.parents[1] / "outputs"
PREDICTIONS = OUT_DIR.parents[1] / "predictions"
PILOT_JSONL = OUT_DIR.parents[1] / "data/americasnlp2026/data/pilot/wixarika.jsonl"

LANG = "wixarika"
LANG_DISPLAY = "Wixárika"
ARM_A, ARM_B = "smolvlm-rag", "smolvlm-ragdistill"
SEED = 20260727


def _present_set(record: dict) -> set[str]:
    annots = record.get("cultural_annotations", {}) or {}
    return {c for c in CATEGORIES if category_present(annots.get(c, ""))[0]}


def build() -> None:
    rng = random.Random(SEED)
    gold = {r["id"]: r["spanish_caption"] for r in
            (json.loads(l) for l in PILOT_JSONL.open(encoding="utf-8"))}

    a_recs = _load_jsonl(OUTPUTS / f"{LANG}_pilot_cultural-vqa_{ARM_A}.jsonl")
    b_recs = _load_jsonl(OUTPUTS / f"{LANG}_pilot_cultural-vqa_{ARM_B}.jsonl")
    a_tgt = _load_preds(PREDICTIONS / f"{LANG}_pilot_cultural-vqa_{ARM_A}_k5_predictions.txt")
    b_tgt = _load_preds(PREDICTIONS / f"{LANG}_pilot_cultural-vqa_{ARM_B}_k5_predictions.txt")
    a_by_id = {r["id"]: (i, r) for i, r in enumerate(a_recs)}
    b_by_id = {r["id"]: (i, r) for i, r in enumerate(b_recs)}

    ids = sorted(set(a_by_id) & set(b_by_id) & set(gold))
    assert len(ids) == 20, f"expected 20 pilot images, got {len(ids)}: {ids}"

    spanish_rows, target_rows, key_rows = [], [], []
    for n, iid in enumerate(ids, 1):
        ai, arec = a_by_id[iid]
        bi, brec = b_by_id[iid]
        sid = f"S{n:03d}"
        a_is_A = rng.random() < 0.5
        arm_A, arm_B = (ARM_A, ARM_B) if a_is_A else (ARM_B, ARM_A)
        es = {ARM_A: arec.get("generated_spanish", ""), ARM_B: brec.get("generated_spanish", "")}
        tg = {ARM_A: a_tgt[ai], ARM_B: b_tgt[bi]}
        common = {"sample_id": sid, "language": LANG_DISPLAY,
                  "image_filename": arec.get("filename", "")}
        spanish_rows.append({**common, "caption_A": es[arm_A], "caption_B": es[arm_B],
                             "gold_spanish": gold[iid], **{c: "" for c in SCORE_COLS}})
        target_rows.append({**common, "caption_A": tg[arm_A], "caption_B": tg[arm_B],
                            **{c: "" for c in SCORE_COLS}})
        cats = sorted(CATEGORY_LABEL[c] for c in _present_set(arec))
        key_rows.append({"sample_id": sid, "language": LANG_DISPLAY, "image_id": iid,
                         "image_filename": arec.get("filename", ""),
                         "slot_A_arm": arm_A, "slot_B_arm": arm_B,
                         "categories_present": "; ".join(cats) or "(none detected)"})

    def write(path: Path, header: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    write(OUT_DIR / "sample_spanish_round3.csv",
          ["sample_id", "language", "image_filename", "caption_A", "caption_B",
           "gold_spanish", *SCORE_COLS],
          spanish_rows)
    write(OUT_DIR / "sample_target_round3.csv",
          ["sample_id", "language", "image_filename", "caption_A", "caption_B", *SCORE_COLS],
          target_rows)
    write(OUT_DIR / "sample_key_round3.csv",
          ["sample_id", "language", "image_id", "image_filename",
           "slot_A_arm", "slot_B_arm", "categories_present"],
          key_rows)
    print(f"Wrote {len(ids)} pilot images ({ARM_A} vs {ARM_B}) -> "
          f"sample_{{spanish,target,key}}_round3.csv")


if __name__ == "__main__":
    build()
