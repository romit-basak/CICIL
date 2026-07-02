"""SmolVLM LoRA fine-tuning scaffold (Stage 1 primary model).

Freezes the SigLIP vision encoder and trains LoRA adapters on the language
decoder using the pilot set's Spanish reference captions (image -> Spanish
caption). Hyperparameters come from config.py (r=16, alpha=32, dropout=0.05,
lr=2e-4). This is designed to run on a GCP T4 (CUDA); locally it supports
``--smoke-test`` to verify the training loop wires up on MPS without doing a
real run.

Real training is out of scope for the local Mac session (T4 job); note also
that the pilot sets are tiny (Wixarika = 20), so LoRA will overfit — consider
pooling languages, which is why ``--langs`` accepts several.
"""

from __future__ import annotations

import argparse

from . import config
from .data_io import load_split

INSTRUCTION = "Describe esta imagen en español."


def build_examples(langs: list[str]) -> list[tuple]:
    """Collect (image_path, spanish_caption) pairs from the pilot split.

    NOTE: as shipped, the pilot split (the only source of Spanish reference
    captions) exists for Wixarika only. Languages without a pilot file are
    skipped rather than raising, so this naturally yields the available pairs.
    """
    pairs = []
    for lang in langs:
        try:
            examples = load_split(lang, "pilot")
        except FileNotFoundError:
            print(f"[skip] no pilot split for {lang}")
            continue
        for ex in examples:
            if ex.spanish_caption and ex.image_path.exists():
                pairs.append((ex.image_path, ex.spanish_caption))
    return pairs


def load_model_and_processor(model_id: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Disable image splitting: it multiplies the image into many sub-crops
    # (many image tokens), which is the main memory driver on MPS.
    processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
    device = config.device()
    # NOTE: on a pre-Ampere CUDA GPU (e.g. the T4 target) config.dtype() returns
    # float16. Training half-precision weights directly is stable in bf16 but
    # underflows in fp16 without autocast + GradScaler — add those before a real
    # T4 run (the --smoke-test's 2 steps are fine as-is).
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=config.dtype(device))
    return model.to(device), processor, device


def apply_lora(model):
    """Attach LoRA adapters to the decoder attention projections.

    peft freezes all base-model parameters and trains only the adapters, so
    targeting the text-decoder projections keeps the SigLIP vision encoder
    frozen as required.
    """
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # Keep the trainable LoRA params in fp32 for numerically stable optimisation
    # while the frozen base stays in half precision to save memory; autocast handles
    # the mixed-precision matmuls at forward time. This is what makes fp16 training
    # (e.g. on a T4) safe alongside a GradScaler.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


def collate(batch, processor, device):
    """Format (image, caption) pairs into model inputs + masked labels."""
    import torch
    from PIL import Image

    texts, images = [], []
    for image_path, caption in batch:
        messages = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": INSTRUCTION}]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]
        texts.append(processor.apply_chat_template(messages, add_generation_prompt=False))
        images.append([Image.open(image_path).convert("RGB")])

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is not None:
        labels[labels == image_token_id] = -100
    # TODO(real training): also mask the user-prompt tokens so loss is computed
    # only over the assistant caption. Masking pad+image is sufficient for the
    # wiring smoke test.
    inputs["labels"] = labels
    return {k: v.to(device) for k, v in inputs.items()}


def amp_context(device: str):
    """Return (autocast_dtype, use_autocast, GradScaler) for the device.

    Mixed precision keeps memory low while staying numerically safe across GPUs:
      * CUDA Ampere+  -> bfloat16 autocast, no loss scaling (bf16 has the range).
      * CUDA pre-Ampere (T4) / MPS -> float16 autocast; on CUDA a GradScaler is
        enabled so fp16 gradients don't underflow to zero.
      * CPU -> no autocast (fp32), scaler disabled.
    A disabled GradScaler is a transparent pass-through, so the training step is
    identical on every device.
    """
    import torch

    amp_dtype = config.dtype(device)
    use_autocast = device in ("cuda", "mps")
    scaler = torch.amp.GradScaler(
        enabled=(device == "cuda" and amp_dtype == torch.float16)
    )
    return amp_dtype, use_autocast, scaler


def train_step(model, batch, optim, scaler, device, amp_dtype, use_autocast):
    """One device-agnostic optimisation step (autocast + optional loss scaling)."""
    import torch

    optim.zero_grad()
    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    return loss


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune SmolVLM on pilot captions.")
    ap.add_argument("--langs", nargs="+", default=config.LANGUAGES)
    ap.add_argument("--model", default=None,
                    help="Model id. Defaults to SmolVLM-2B (GCP) or the 256M "
                         "variant under --smoke-test.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Load model, attach LoRA, run 2 steps on 2 examples, then exit.")
    args = ap.parse_args()

    import torch

    model_id = args.model or (config.SMOLVLM_SMOKE_ID if args.smoke_test else config.SMOLVLM_ID)

    pairs = build_examples(args.langs)
    print(f"Loaded {len(pairs)} (image, spanish_caption) pilot pairs from {args.langs}")
    if not pairs:
        raise SystemExit("No pilot pairs found — check the dataset clone.")

    model, processor, device = load_model_and_processor(model_id)
    model = apply_lora(model)
    model.train()
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.LEARNING_RATE
    )
    amp_dtype, use_autocast, scaler = amp_context(device)

    if args.smoke_test:
        batch = collate(pairs[:2], processor, device)
        for step in range(2):
            loss = train_step(model, batch, optim, scaler, device, amp_dtype, use_autocast)
            print(f"[smoke] step {step}: loss={loss.item():.4f}")
        assert torch.isfinite(loss), "Non-finite loss — training loop is broken."
        print("[smoke] OK — model loads, LoRA attaches, forward/backward/step run.")
        return

    # Full training loop (intended for GCP T4). Tiny data -> no held-out split;
    # early stopping is deferred until a validation strategy is agreed.
    from torch.utils.data import DataLoader

    loader = DataLoader(
        pairs, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, processor, device),
    )
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            loss = train_step(model, batch, optim, scaler, device, amp_dtype, use_autocast)
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}: mean_loss={total / len(loader):.4f}")

    config.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.ADAPTER_DIR)
    print(f"Saved LoRA adapter -> {config.ADAPTER_DIR}")


if __name__ == "__main__":
    main()
