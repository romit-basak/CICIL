"""SmolVLM LoRA fine-tuning (Stage 1 primary model).

Freezes the SigLIP vision encoder and trains LoRA adapters on the language
decoder using the pilot set's Spanish reference captions (image -> Spanish
caption). Hyperparameters come from config.py (r=16, alpha=32, dropout=0.05,
lr=2e-4). Designed for a GCP T4 (CUDA, fp16 + GradScaler); ``--smoke-test``
verifies the loop on MPS/CPU without a real run.

Modes (see ``main``):
  * ``--smoke-test``    load model, attach LoRA, 2 steps, exit (MPS wiring check).
  * ``--baseline-eval`` off-the-shelf model (no LoRA) scored on the pilot Spanish.
  * ``--loo``           leave-one-out CV: train on n-1, score the held-out one,
                        repeat; the honest held-out metric at n=20.
  * (default)           train on ALL pilot pairs and save the deployed adapter.

Data reality: the only split with ``spanish_caption`` is the Wixarika pilot (20
pairs), so this trains on ~20 examples. That scarcity is the point of RQ2, not a
bug — report the LOO number honestly.
"""

from __future__ import annotations

import argparse

from . import config, vqa_prompts
from .data_io import load_split

# Train and generate with the SAME instruction the generic pipeline uses, so the
# fine-tuned adapter is consistent between training, LOO eval, and dev generation
# (generate_descriptions --mode generic sends exactly this prompt).
INSTRUCTION = vqa_prompts.GENERIC_PROMPT

# Small, fixed epoch count: at n<=20, per-fold early stopping (1 val example) is
# pure noise, so we fix a conservative budget instead. Override with --epochs.
DEFAULT_EPOCHS = 4
GEN_MAX_NEW_TOKENS = 64


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
    # (many image tokens), which is the main memory driver.
    processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
    # Right-padding puts the prompt prefix at the start of each row, which the
    # caption-only label masking in collate() relies on.
    processor.tokenizer.padding_side = "right"
    device = config.device()
    # On a pre-Ampere CUDA GPU (the T4 target) config.dtype() is float16; the
    # autocast + GradScaler path in amp_context/train_step keeps that stable.
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
    # Keep trainable LoRA params in fp32 for numerically stable optimisation while
    # the frozen base stays half precision to save memory; autocast handles the
    # mixed-precision matmuls at forward time (fp16-on-T4 safe with a GradScaler).
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


def _prompt_length(processor, image, instruction: str) -> int:
    """Token length of the user/image/instruction prefix (incl. expanded image
    tokens and the generation prompt), i.e. everything before the caption."""
    prompt = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(prompt, add_generation_prompt=True)
    enc = processor(text=[text], images=[[image]], return_tensors="pt")
    return enc["input_ids"].shape[1]


def collate(batch, processor, device):
    """Format (image, caption) pairs into model inputs + caption-only labels.

    Loss is computed over the assistant caption tokens only: the user turn,
    image tokens, instruction, and padding are all masked to -100.
    """
    from PIL import Image

    texts, images, prompt_lens = [], [], []
    for image_path, caption in batch:
        img = Image.open(image_path).convert("RGB")
        full = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": INSTRUCTION}]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]
        texts.append(processor.apply_chat_template(full, add_generation_prompt=False))
        prompt_lens.append(_prompt_length(processor, img, INSTRUCTION))
        images.append([img])

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is not None:
        labels[labels == image_token_id] = -100
    # Mask the whole prompt prefix so loss falls only on the caption (right-padded
    # => prefix occupies the first prompt_len positions of each row).
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
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


def train_loop(model, processor, device, pairs, epochs, batch_size, quiet=False):
    """Train the (LoRA-wrapped) model in place over ``pairs``."""
    import torch
    from torch.utils.data import DataLoader

    model.train()
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.LEARNING_RATE
    )
    amp_dtype, use_autocast, scaler = amp_context(device)
    loader = DataLoader(
        pairs, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, processor, device),
    )
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            loss = train_step(model, batch, optim, scaler, device, amp_dtype, use_autocast)
            total += loss.item()
        if not quiet:
            print(f"  epoch {epoch + 1}/{epochs}: mean_loss={total / len(loader):.4f}")
    return model


def generate_caption(model, processor, device, image_path, max_new_tokens=GEN_MAX_NEW_TOKENS):
    """Greedy-decode a Spanish caption for one image (autocast for dtype safety)."""
    import torch
    from PIL import Image

    amp_dtype = config.dtype(device)
    use_autocast = device in ("cuda", "mps")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[[Image.open(image_path).convert("RGB")]],
                       return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def _free(model, device):
    """Release a model between LOO folds so the 2B base doesn't accumulate."""
    import gc

    import torch

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def run_baseline_eval(model_id, langs) -> list[float]:
    """Off-the-shelf model (no LoRA) scored vs pilot Spanish gold — the control."""
    from .evaluate import chrfpp

    pairs = build_examples(langs)
    model, processor, device = load_model_and_processor(model_id)
    scores = []
    for i, (img, cap) in enumerate(pairs):
        hyp = generate_caption(model, processor, device, img)
        s = chrfpp(hyp, cap)
        scores.append(s)
        print(f"[base {i + 1}/{len(pairs)}] chrF++={s:.2f}  hyp={hyp[:60]!r}")
    _report("baseline (off-the-shelf, no LoRA)", scores)
    return scores


def run_loo(model_id, langs, epochs, batch_size) -> list[float]:
    """Leave-one-out CV: train on n-1 pilot pairs, score the held-out one."""
    from .evaluate import chrfpp

    pairs = build_examples(langs)
    n = len(pairs)
    scores = []
    for j in range(n):
        held_img, held_cap = pairs[j]
        train_pairs = pairs[:j] + pairs[j + 1:]
        model, processor, device = load_model_and_processor(model_id)
        model = apply_lora(model)
        train_loop(model, processor, device, train_pairs, epochs, batch_size, quiet=True)
        hyp = generate_caption(model, processor, device, held_img)
        s = chrfpp(hyp, held_cap)
        scores.append(s)
        print(f"[loo {j + 1}/{n}] chrF++={s:.2f}  hyp={hyp[:60]!r}")
        _free(model, device)
    _report(f"SmolVLM+LoRA leave-one-out (n={n}, {epochs} epochs)", scores)
    return scores


def _report(label: str, scores: list[float]) -> None:
    import statistics

    mean = statistics.mean(scores)
    sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"\n== {label} ==")
    print(f"mean chrF++ = {mean:.2f} ± {sd:.2f}  (n={len(scores)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune SmolVLM on pilot captions.")
    ap.add_argument("--langs", nargs="+", default=config.LANGUAGES)
    ap.add_argument("--model", default=None,
                    help="Model id. Defaults to SmolVLM-2B (GCP) or the 256M "
                         "variant under --smoke-test.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true",
                      help="Load model, attach LoRA, run 2 steps on 2 examples, then exit.")
    mode.add_argument("--baseline-eval", action="store_true",
                      help="Score the off-the-shelf model (no LoRA) on the pilot Spanish.")
    mode.add_argument("--loo", action="store_true",
                      help="Leave-one-out CV; the honest held-out metric at n=20.")
    args = ap.parse_args()

    import torch

    model_id = args.model or (config.SMOLVLM_SMOKE_ID if args.smoke_test else config.SMOLVLM_ID)

    if args.baseline_eval:
        run_baseline_eval(model_id, args.langs)
        return
    if args.loo:
        run_loo(model_id, args.langs, args.epochs, args.batch_size)
        return

    pairs = build_examples(args.langs)
    print(f"Loaded {len(pairs)} (image, spanish_caption) pilot pairs from {args.langs}")
    if not pairs:
        raise SystemExit("No pilot pairs found — check the dataset clone.")

    model, processor, device = load_model_and_processor(model_id)
    model = apply_lora(model)

    if args.smoke_test:
        model.train()
        optim = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=config.LEARNING_RATE
        )
        amp_dtype, use_autocast, scaler = amp_context(device)
        batch = collate(pairs[:2], processor, device)
        for step in range(2):
            loss = train_step(model, batch, optim, scaler, device, amp_dtype, use_autocast)
            print(f"[smoke] step {step}: loss={loss.item():.4f}")
        assert torch.isfinite(loss), "Non-finite loss — training loop is broken."
        print("[smoke] OK — model loads, LoRA attaches, forward/backward/step run.")
        return

    # Default: train the deployed adapter on ALL pilot pairs and save it.
    train_loop(model, processor, device, pairs, args.epochs, args.batch_size)
    config.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.ADAPTER_DIR)
    print(f"Saved LoRA adapter -> {config.ADAPTER_DIR}")


if __name__ == "__main__":
    main()
