"""Build the round-4 human-eval sample: old Stage 1 vs the multi-agent arm.

Round 3 compared two variants of the OLD Stage 1 (smolvlm-rag vs
smolvlm-ragdistill). Round 4 asks the project's central question directly:
does the multi-agent interrogation arm (fully local: 7B questioner + 7B
answerer, v4.3.1 transcripts re-assembled under the v4.5 whitelist policy)
beat the original smolvlm-rag arm, judged by humans?

Two blocks:
  1. the 20 wixarika PILOT images -- these have gold Spanish captions,
     shown unblinded for reference;
  2. a seeded dev sample from the other four cultures (same seed as
     prototype_agent_rag --dev-sample) -- dev has no Spanish gold, so
     gold_spanish is empty and annotators judge against the image alone.

NOTE on the gold references (block 1): several gold captions encode
CONTEXT that is not visually discernible (hch_002: "toros de reparo
descansando antes de que comience el jaripeo" -- nothing in the frame
identifies a rodeo). The interface instructs annotators to judge captions
against the IMAGE; gold is context, not the answer key.

Spanish-side only by design -- Stage 2 is NOT run for this round; the target
sheet is written with placeholders so build_interface.py needs no changes.
Blinding as always: A/B slots shuffled per image, mapping only in
sample_key_round4.csv.

Run (after the v4.5-local reassemblies exist):
    uv run python -m analysis.human_eval.build_round4_pilot
    uv run python -m analysis.human_eval.translate_english --suffix _round4
    uv run python -m analysis.human_eval.build_interface --suffix _round4
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

LANG_DISPLAY = {"wixarika": "Wixárika", "guarani": "Guaraní", "maya": "Maya",
                "bribri": "Bribri", "nahuatl": "Nahuatl"}
ARM_A = "smolvlm-rag"                       # standard-named Stage 1 output
ARM_B = "agent-local"                       # multi-agent, fully local, v4.5 finals
AGENT_PILOT = OUTPUTS / "agent_rag_pilot_curve_local7b_v431_v45-local.jsonl"
AGENT_DEV = OUTPUTS / "agent_rag_dev_round4_local7b_v431_v45-local.jsonl"
SEED = 20260731


def build() -> None:
    rng = random.Random(SEED)
    gold = {r["id"]: r["spanish_caption"] for r in
            (json.loads(l) for l in PILOT_JSONL.open(encoding="utf-8"))}

    # (culture, split, image_id, baseline_rec, agent_rec, gold_spanish)
    entries = []

    a_pilot = {r["id"]: r for r in
               _load_jsonl(OUTPUTS / f"wixarika_pilot_cultural-vqa_{ARM_A}.jsonl")}
    b_pilot = {r["id"]: r for r in
               (json.loads(l) for l in AGENT_PILOT.open(encoding="utf-8"))}
    pilot_ids = sorted(set(a_pilot) & set(b_pilot) & set(gold))
    assert len(pilot_ids) == 20, f"expected 20 pilot images, got {len(pilot_ids)}"
    entries += [("wixarika", "pilot", i, a_pilot[i], b_pilot[i], gold[i])
                for i in pilot_ids]

    if AGENT_DEV.exists():
        b_dev = [json.loads(l) for l in AGENT_DEV.open(encoding="utf-8") if l.strip()]
        a_dev: dict[str, dict[str, dict]] = {}
        for rec in sorted(b_dev, key=lambda r: (r["culture"], r["id"])):
            culture = rec["culture"]
            if culture not in a_dev:
                a_dev[culture] = {r["id"]: r for r in _load_jsonl(
                    OUTPUTS / f"{culture}_dev_cultural-vqa_{ARM_A}.jsonl")}
            entries.append((culture, "dev", rec["id"],
                            a_dev[culture][rec["id"]], rec, ""))
    else:
        print(f"WARNING: {AGENT_DEV.name} not found -- building the "
              f"wixarika-pilot block only (rerun after the dev arm lands).")

    spanish_rows, target_rows, key_rows = [], [], []
    for n, (culture, split, iid, arec, brec, gold_es) in enumerate(entries, 1):
        sid = f"S{n:03d}"
        a_is_A = rng.random() < 0.5
        arm_A, arm_B = (ARM_A, ARM_B) if a_is_A else (ARM_B, ARM_A)
        es = {ARM_A: arec.get("generated_spanish", ""), ARM_B: brec["final"]}
        common = {"sample_id": sid, "language": LANG_DISPLAY[culture],
                  "split": split,
                  "image_filename": Path(arec.get("filename", f"{iid}.jpg")).name}
        spanish_rows.append({**common, "caption_A": es[arm_A], "caption_B": es[arm_B],
                             "gold_spanish": gold_es, **{c: "" for c in SCORE_COLS}})
        target_rows.append({**common,
                            "caption_A": "(Stage 2 not run for round 4)",
                            "caption_B": "(Stage 2 not run for round 4)",
                            **{c: "" for c in SCORE_COLS}})
        key_rows.append({"sample_id": sid, "language": LANG_DISPLAY[culture],
                         "split": split, "image_id": iid,
                         "image_filename": common["image_filename"],
                         "slot_A_arm": arm_A, "slot_B_arm": arm_B,
                         "categories_present": ""})

    def write(path: Path, header: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    write(OUT_DIR / "sample_spanish_round4.csv",
          ["sample_id", "language", "split", "image_filename", "caption_A",
           "caption_B", "gold_spanish", *SCORE_COLS], spanish_rows)
    write(OUT_DIR / "sample_target_round4.csv",
          ["sample_id", "language", "split", "image_filename", "caption_A",
           "caption_B", *SCORE_COLS], target_rows)
    write(OUT_DIR / "sample_key_round4.csv",
          ["sample_id", "language", "split", "image_id", "image_filename",
           "slot_A_arm", "slot_B_arm", "categories_present"], key_rows)
    n_dev = len(entries) - len(pilot_ids)
    print(f"Wrote {len(entries)} images ({len(pilot_ids)} wixarika pilot w/ gold "
          f"+ {n_dev} cross-culture dev) ({ARM_A} vs {ARM_B}) -> "
          f"sample_{{spanish,target,key}}_round4.csv")


if __name__ == "__main__":
    build()
