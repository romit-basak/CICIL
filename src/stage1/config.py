"""Central paths, model ids, and hyperparameters for Stage 1.

Keeping these in one module means the fine-tuning script and the inference
scripts agree on the LoRA config and dataset locations without duplication.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# Project root is two levels up from this file: <root>/src/stage1/config.py
ROOT = Path(__file__).resolve().parents[2]

# The AmericasNLP 2026 CICIL repo is cloned (gitignored) under data/.
DATASET_ROOT = ROOT / "data" / "americasnlp2026" / "data"
PILOT_DIR = DATASET_ROOT / "pilot"
DEV_DIR = DATASET_ROOT / "dev"
TEST_DIR = DATASET_ROOT / "test"

# Stage 1 writes generated Spanish descriptions and adapters here (gitignored).
OUTPUT_DIR = ROOT / "outputs"
ADAPTER_DIR = OUTPUT_DIR / "adapters"

LANGUAGES = ["guarani", "bribri", "maya", "wixarika", "nahuatl"]

# --- Models ------------------------------------------------------------------
# Generic baseline (Stage 1 ablation control). Two interchangeable backends:
OLLAMA_VLM = "qwen2.5vl:7b"                    # local, quantized, fast (Metal)
HF_QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"     # reproducible; 3B for local MPS, 7B on GCP

# Primary Stage 1 model, fine-tuned via LoRA (decoder only; SigLIP encoder frozen).
SMOLVLM_ID = "HuggingFaceTB/SmolVLM-Instruct"
# Tiny variant for the local MPS wiring smoke test (2B OOMs on Mac unified memory).
SMOLVLM_SMOKE_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# --- LoRA hyperparameters (from CLAUDE.md) -----------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
MAX_EPOCHS = 10


def device() -> str:
    """Best available torch device: CUDA (GCP) > MPS (Mac) > CPU."""
    import os

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        # Some VLM ops aren't implemented for Metal; fall back to CPU per-op
        # instead of raising. Read lazily by the dispatcher, so setting it here
        # (before the first forward pass) is sufficient.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def dtype(dev: str | None = None):
    """Best compute dtype for a device, safe across GPU generations.

    * CPU  -> float32 (half precision is slow/unsupported on CPU).
    * MPS  -> float16 (broadly supported; bf16 on Metal is torch-version-dependent).
    * CUDA -> bfloat16 on Ampere+ (compute >= 8.0), else float16. This matters for
      the T4 target (Turing, 7.5): it has no bf16 Tensor Cores, so bf16 runs
      unaccelerated. ``is_bf16_supported()`` is the canonical gate.
    """
    import torch

    dev = dev or device()
    if dev == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dev == "mps":
        return torch.float16
    return torch.float32
