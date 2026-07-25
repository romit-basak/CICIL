"""
build_corpora.py
Converts AmericasNLP parallel corpora into JSONL format for FAISS indexing.
Run from inside the americasnlp2026-main folder:
    python build_corpora.py
"""

import json
import os

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Paths to the downloaded repos (edit these if yours are in a different location)
REPO_2021 = "/Users/nanditamahendra/Downloads/americasnlp2021-main/data"
REPO_2023 = "/Users/nanditamahendra/Downloads/americasnlp2023-main/data"

# Where to save the JSONL files (inside your 2026 repo)
OUT_DIR = "data/corpora"
os.makedirs(OUT_DIR, exist_ok=True)

# ── CORPUS DEFINITIONS ───────────────────────────────────────────────────────
# Each entry: (language display name, output filename, list of (es_file, target_file) pairs to combine)
CORPORA = [
    (
        "Guaraní",
        "guarani.jsonl",
        [
            # 2021 train (larger)
            (f"{REPO_2021}/guarani-spanish/train.es",
             f"{REPO_2021}/guarani-spanish/train.gn"),
            # 2023 train (updated)
            (f"{REPO_2023}/guarani-spanish/train.es",
             f"{REPO_2023}/guarani-spanish/train.gn"),
            # dev sets from both
            (f"{REPO_2021}/guarani-spanish/dev.es",
             f"{REPO_2021}/guarani-spanish/dev.gn"),
            (f"{REPO_2023}/guarani-spanish/dev.es",
             f"{REPO_2023}/guarani-spanish/dev.gn"),
        ]
    ),
    (
        "Bribri",
        "bribri.jsonl",
        [
            (f"{REPO_2021}/bribri-spanish/train.es",
             f"{REPO_2021}/bribri-spanish/train.bzd"),
            (f"{REPO_2021}/bribri-spanish/dev.es",
             f"{REPO_2021}/bribri-spanish/dev.bzd"),
            # 2023 bribri
            (f"{REPO_2023}/bribri-spanish/train.es",
             f"{REPO_2023}/bribri-spanish/train.bzd"),
            (f"{REPO_2023}/bribri-spanish/dev.es",
             f"{REPO_2023}/bribri-spanish/dev.bzd"),
        ]
    ),
    (
        "Wixárika",
        "wixarika.jsonl",
        [
            (f"{REPO_2021}/wixarika-spanish/train.es",
             f"{REPO_2021}/wixarika-spanish/train.hch"),
            (f"{REPO_2021}/wixarika-spanish/dev.es",
             f"{REPO_2021}/wixarika-spanish/dev.hch"),
            (f"{REPO_2023}/wixarika-spanish/train.es",
             f"{REPO_2023}/wixarika-spanish/train.hch"),
            (f"{REPO_2023}/wixarika-spanish/dev.es",
             f"{REPO_2023}/wixarika-spanish/dev.hch"),
        ]
    ),
    (
        "Nahuatl",
        "nahuatl.jsonl",
        [
            (f"{REPO_2021}/nahuatl-spanish/train.es",
             f"{REPO_2021}/nahuatl-spanish/train.nah"),
            (f"{REPO_2021}/nahuatl-spanish/dev.es",
             f"{REPO_2021}/nahuatl-spanish/dev.nah"),
            (f"{REPO_2023}/nahuatl-spanish/train.es",
             f"{REPO_2023}/nahuatl-spanish/train.nah"),
            (f"{REPO_2023}/nahuatl-spanish/dev.es",
             f"{REPO_2023}/nahuatl-spanish/dev.nah"),
        ]
    ),
]

# ── CONVERSION FUNCTION ───────────────────────────────────────────────────────
def convert_to_jsonl(lang_name, out_filename, file_pairs):
    out_path = os.path.join(OUT_DIR, out_filename)
    total = 0
    skipped = 0
    seen = set()  # deduplicate across files

    with open(out_path, "w", encoding="utf-8") as out_f:
        for es_path, tgt_path in file_pairs:
            if not os.path.exists(es_path):
                print(f"  SKIP (not found): {es_path}")
                continue
            if not os.path.exists(tgt_path):
                print(f"  SKIP (not found): {tgt_path}")
                continue

            with open(es_path, encoding="utf-8") as es_f, \
                 open(tgt_path, encoding="utf-8") as tgt_f:

                for es_line, tgt_line in zip(es_f, tgt_f):
                    es   = es_line.strip()
                    tgt  = tgt_line.strip()

                    # skip empty or misaligned lines
                    if not es or not tgt:
                        skipped += 1
                        continue

                    # skip duplicates
                    key = (es, tgt)
                    if key in seen:
                        skipped += 1
                        continue
                    seen.add(key)

                    record = {
                        "spanish_caption": es,
                        "target_caption":  tgt
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1

    print(f"✓ {lang_name:<15} → {out_path}  ({total:,} pairs, {skipped} skipped)")
    return total

# ── MAIN ─────────────────────────────────────────────────────────────────────
print("Building parallel corpora JSONL files...\n")
summary = {}
for lang_name, out_filename, file_pairs in CORPORA:
    n = convert_to_jsonl(lang_name, out_filename, file_pairs)
    summary[lang_name] = n

print("\n── SUMMARY ─────────────────────────────────────────────────────────")
print(f"{'Language':<20} {'# Pairs':>10}")
print("-" * 32)
for lang, n in summary.items():
    print(f"{lang:<20} {n:>10,}")
print(f"\nAll files saved to: {OUT_DIR}/")
print("\nNext step: update src/stage2/build_index.py to point at these files.")