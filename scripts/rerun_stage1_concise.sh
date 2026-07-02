#!/usr/bin/env bash
# Re-run the Stage 1 generic vs cultural-VQA comparison on the Wixárika pilot
# with the v2 concise synthesis prompt (see src/stage1/vqa_prompts.py).
#
# The Wixárika pilot (20 images) is the ONLY split with Spanish gold captions, so
# it is the only place ChrF++ can be measured. Expect ~20-40 min with no other
# ollama jobs running. v1 (verbose) outputs are frozen under
# outputs/prelim_v1_verbose/ so this run can safely overwrite the live files.
#
# Usage:  bash scripts/rerun_stage1_concise.sh
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="outputs/rerun_concise_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

{
  echo "=== $(date) : regenerating GENERIC (Wixárika pilot) ==="
  uv run python -m src.stage1.generate_descriptions \
      --lang wixarika --split pilot --mode generic --backend ollama

  echo "=== $(date) : regenerating CULTURAL-VQA (Wixárika pilot, v2 concise) ==="
  uv run python -m src.stage1.generate_descriptions \
      --lang wixarika --split pilot --mode cultural-vqa --backend ollama

  echo "=== $(date) : comparison (v2) ==="
  uv run python -m src.stage1.compare --lang wixarika --split pilot
} 2>&1 | tee "$LOG"

echo
echo "Done. v2 verdict is above; v1 (verbose) baseline is in outputs/prelim_v1_verbose/"
echo "The compare tool prints NO VERDICT if any mode has >10% empty outputs — check that first."
