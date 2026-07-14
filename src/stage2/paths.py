"""Shared Stage 2 paths, derived from ``src.stage1.config`` (single source of truth).

Using ``config`` here means Stage 2 is robust to the invocation directory (paths
are absolute, anchored at the repo root) and stays consistent with Stage 1's
``outputs/`` location instead of re-hardcoding CWD-relative literals.
"""

from __future__ import annotations

from src.stage1 import config

# FAISS indices are a regenerable build artifact (gitignored).
INDEX_DIR = config.ROOT / "indices"
# Stage 1 hand-off JSONLs live here.
INPUT_DIR = config.OUTPUT_DIR
# Stage 2 prediction .txt files (scored by src.stage1.evaluate).
PRED_DIR = config.ROOT / "predictions"

# The multilingual sentence encoder for retrieval (must match across build/query).
ENCODER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
