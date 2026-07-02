#!/usr/bin/env bash
# Generate Stage 1 Spanish descriptions for all dev languages.
# Output: outputs/<lang>_dev_<mode>_ollama.jsonl (the Stage 2 hand-off).
# Run from the project root.
#   MODE=generic       ~20s/image  => ~1.5h for 5x50 images   (default)
#   MODE=cultural-vqa  ~100s/image => ~7h for 5x50 images     (overnight)
# Usage: [MODE=cultural-vqa] scripts/run_generic_baseline.sh [lang ...]
set -uo pipefail   # NOT -e: one language's failure must not abort the rest

MODE=${MODE:-generic}
cd "$(dirname "$0")/.."
mkdir -p outputs

# Languages can be overridden as args, e.g. `run_generic_baseline.sh bribri maya`.
langs=("$@")
[ ${#langs[@]} -eq 0 ] && langs=(guarani bribri maya wixarika nahuatl)

for lang in "${langs[@]}"; do
  echo "=== $(date '+%H:%M:%S') generating ($MODE): $lang ==="
  uv run python -m src.stage1.generate_descriptions \
    --lang "$lang" --split dev --mode "$MODE" --backend ollama \
    || echo "!!! FAILED: $lang (continuing)"
done

echo "=== $(date '+%H:%M:%S') done ==="
