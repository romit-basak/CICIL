#!/usr/bin/env bash
# Stage 1: generate distilled-SmolVLM dev descriptions for the 4 languages that
# don't have them yet (Wixárika already exists). Both modes per language.
#
# Adapter choice is a positional arg so the Part-B adapter-selection result can
# be swapped in without editing this file:
#     ./scripts/run_stage1_smolvlm_all_langs.sh outputs/adapters/distill_full smolvlm
#     ./scripts/run_stage1_smolvlm_all_langs.sh outputs/adapters/distill_noctx smolvlm-noctx
#
# Sequential on purpose (one SmolVLM instance at a time on the M4). Each run's
# JSONL flushes per record, so progress is inspectable mid-run and a crash only
# costs the current run. generate_descriptions.py refuses to overwrite existing
# files, so re-running the script after a partial night only redoes the
# unfinished language if you first delete its partial file.

set -uo pipefail

ADAPTER="${1:?usage: $0 <adapter-dir> <backend-tag>}"
BACKEND="${2:?usage: $0 <adapter-dir> <backend-tag>}"
LANGS=(guarani bribri maya nahuatl)
LOG="outputs/stage1_smolvlm_all_langs_$(date +%Y%m%d_%H%M%S).log"

echo "adapter=$ADAPTER backend=$BACKEND log=$LOG"
for lang in "${LANGS[@]}"; do
    for mode in generic cultural-vqa; do
        echo "=== $(date '+%F %T') $lang / $mode ===" | tee -a "$LOG"
        uv run python -m src.stage1.generate_descriptions \
            --lang "$lang" --split dev --mode "$mode" \
            --backend "$BACKEND" --adapter "$ADAPTER" --joint-questions \
            2>&1 | tee -a "$LOG"
        echo "exit=$? for $lang/$mode" | tee -a "$LOG"
    done
done
echo "=== $(date '+%F %T') all done ===" | tee -a "$LOG"
