"""Build the round-4 human-eval sample: old Stage 1 vs the multi-agent arm.

Round 3 compared two variants of the OLD Stage 1 (smolvlm-rag vs
smolvlm-ragdistill). Round 4 asks the project's central question directly:
does the v4.3 multi-agent interrogation arm (fully local: 7B questioner +
7B answerer) beat the original smolvlm-rag arm, judged by humans against the
pilot's real gold Spanish captions?

Spanish-side only by design — Stage 2 is NOT run for this round; the target
sheet is written with placeholders so build_interface.py needs no changes.
Blinding as always: A/B slots shuffled per image, mapping only in
sample_key_round4.csv.

Run:
    uv run python -m analysis.human_eval.build_round4_pilot
    uv run python -m analysis.human_eval.translate_english --suffix _round4
    uv run python -m analysis.human_eval.build_interface --suffix _round4 --split pilot
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from analysis.human_eval.build_sample import SCORE_COLS, _load_jsonl

OUT_DIR = Path(__file__).resolve().parent
OUTPUTS = OUT_DIR.parents[1] / "outputs"
PILOT_JSONL = OUT_DIR.parents[1] / "data/americasnlp2026/data/pilot/wixarika.jsonl"

LANG_DISPLAY = "Wixárika"
ARM_A = "smolvlm-rag"                       # standard-named Stage 1 output
ARM_B = "agent-local"                       # v4.3 multi-agent, fully local
AGENT_FILE = OUTPUTS / "agent_rag_pilot_curve_local7b_v43.jsonl"
SEED = 20260731


def build() -> None:
    rng = random.Random(SEED)
    gold = {r["id"]: r["spanish_caption"] for r in
            (json.loads(l) for l in PILOT_JSONL.open(encoding="utf-8"))}

    a_recs = _load_jsonl(OUTPUTS / f"wixarika_pilot_cultural-vqa_{ARM_A}.jsonl")
    a_by_id = {r["id"]: r for r in a_recs}
    b_by_id = {r["id"]: r for r in
               (json.loads(l) for l in AGENT_FILE.open(encoding="utf-8"))}

    ids = sorted(set(a_by_id) & set(b_by_id) & set(gold))
    assert len(ids) == 20, f"expected 20 pilot images, got {len(ids)}"

    spanish_rows, target_rows, key_rows = [], [], []
    for n, iid in enumerate(ids, 1):
        arec, brec = a_by_id[iid], b_by_id[iid]
        sid = f"S{n:03d}"
        a_is_A = rng.random() < 0.5
        arm_A, arm_B = (ARM_A, ARM_B) if a_is_A else (ARM_B, ARM_A)
        es = {ARM_A: arec.get("generated_spanish", ""), ARM_B: brec["final"]}
        common = {"sample_id": sid, "language": LANG_DISPLAY,
                  "image_filename": arec.get("filename", f"{iid}.jpg")}
        spanish_rows.append({**common, "caption_A": es[arm_A], "caption_B": es[arm_B],
                             "gold_spanish": gold[iid], **{c: "" for c in SCORE_COLS}})
        target_rows.append({**common,
                            "caption_A": "(Stage 2 not run for round 4)",
                            "caption_B": "(Stage 2 not run for round 4)",
                            **{c: "" for c in SCORE_COLS}})
        key_rows.append({"sample_id": sid, "language": LANG_DISPLAY, "image_id": iid,
                         "image_filename": common["image_filename"],
                         "slot_A_arm": arm_A, "slot_B_arm": arm_B,
                         "categories_present": ""})

    def write(path: Path, header: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    write(OUT_DIR / "sample_spanish_round4.csv",
          ["sample_id", "language", "image_filename", "caption_A", "caption_B",
           "gold_spanish", *SCORE_COLS], spanish_rows)
    write(OUT_DIR / "sample_target_round4.csv",
          ["sample_id", "language", "image_filename", "caption_A", "caption_B",
           *SCORE_COLS], target_rows)
    write(OUT_DIR / "sample_key_round4.csv",
          ["sample_id", "language", "image_id", "image_filename",
           "slot_A_arm", "slot_B_arm", "categories_present"], key_rows)
    print(f"Wrote {len(ids)} pilot images ({ARM_A} vs {ARM_B}) -> "
          f"sample_{{spanish,target,key}}_round4.csv")


if __name__ == "__main__":
    build()
