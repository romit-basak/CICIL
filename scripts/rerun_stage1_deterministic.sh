#!/usr/bin/env bash
# Deterministic Stage 1 comparison on the Wixárika pilot for a CITABLE paper number.
#
# The preliminary runs used temperature=0.2 (stochastic), so the observed
# cultural-VQA vs generic margin (~+1.0 ChrF++) sat within run-to-run noise. This
# run pins temperature=0 with a fixed seed so both modes are reproducible and the
# delta is attributable to the method, not sampling.
#
# NOT run automatically — launch it yourself when the ollama daemon is otherwise
# idle. Expect ~40 min. v1 (verbose) outputs are frozen at outputs/prelim_v1_verbose/;
# this overwrites the live outputs/wixarika_pilot_*_ollama.jsonl, so back those up
# first if you want to keep the temp=0.2 v2 concise numbers.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=0
LOG="outputs/rerun_deterministic_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG (temperature=0, seed=$SEED)"

{
  echo "=== $(date) : GENERIC (Wixárika pilot, temp=0) ==="
  uv run python -m src.stage1.generate_descriptions \
      --lang wixarika --split pilot --mode generic --backend ollama \
      --temperature 0 --seed "$SEED"

  echo "=== $(date) : CULTURAL-VQA (Wixárika pilot, temp=0, v2 concise) ==="
  uv run python -m src.stage1.generate_descriptions \
      --lang wixarika --split pilot --mode cultural-vqa --backend ollama \
      --temperature 0 --seed "$SEED"

  echo "=== $(date) : comparison ==="
  uv run python -m src.stage1.compare --lang wixarika --split pilot
} 2>&1 | tee "$LOG"

echo
echo "Done. Deterministic verdict is above."
echo "The compare tool prints NO VERDICT if any mode has >10% empty outputs — check that first."
