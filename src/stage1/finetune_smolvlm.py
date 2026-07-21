"""SmolVLM LoRA fine-tuning (Stage 1 primary model).

Freezes the SigLIP vision encoder and trains LoRA adapters on the language
decoder. Hyperparameters come from config.py (r=16, alpha=32, dropout=0.05,
lr=2e-4; ``--lora-r``/``--lora-dropout`` override for sweeps). Designed for a
GCP T4 (CUDA, fp16 + GradScaler); ``--smoke-test`` verifies the loop on MPS/CPU
without a real run.

Training items are (image_path, prompt, target) triples, so the same loop
serves both regimes:
  * pilot pairs — image -> Spanish gold caption under the generic prompt
    (the original n=20 setup; LOO gave 17.55 ± 6.52 vs 16.72 ± 3.60 base);
  * distillation triples — deployment-aligned prompts with Qwen2.5-VL silver
    targets (``src.stage1.distill_data``), the data-scarcity fix. The 20 gold
    pilot captions are never trained on in this regime, so they serve as a
    clean held-out test set (``--gold-eval``).

Modes (see ``main``):
  * ``--smoke-test``    load model, attach LoRA, 2 steps, exit (MPS wiring check).
  * ``--baseline-eval`` off-the-shelf model (no LoRA) scored on the pilot Spanish.
  * ``--loo``           leave-one-out CV over the 20 pilot pairs (v1 protocol).
  * ``--distill``       train on the silver triples, save the adapter.
  * ``--gold-eval``     score an adapter (or the base) on the 20 gold pilot
                        captions: generic prompt + full cultural-VQA pipeline.
  * (default)           train on ALL pilot pairs and save the deployed adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config, vqa_prompts
from .data_io import load_split

# Train and generate with the SAME instruction the generic pipeline uses, so the
# fine-tuned adapter is consistent between training, LOO eval, and dev generation
# (generate_descriptions --mode generic sends exactly this prompt).
INSTRUCTION = vqa_prompts.GENERIC_PROMPT

# Small, fixed epoch count for the n<=20 pilot regime: per-fold early stopping
# (1 val example) is pure noise, so we fix a conservative budget instead. The
# distillation regime (~10k pairs) defaults lower. Override with --epochs.
DEFAULT_EPOCHS = 4
DEFAULT_DISTILL_EPOCHS = 2
GEN_MAX_NEW_TOKENS = 64
# Base seed for the per-epoch shuffle (seed = base + global epoch index), so a
# mid-epoch resume can reproduce the exact order and skip what's been seen.
SHUFFLE_BASE_SEED = 20260718


def build_examples(langs: list[str]) -> list[tuple]:
    """Collect (image_path, prompt, spanish_caption) triples from the pilot split.

    NOTE: as shipped, the pilot split (the only source of Spanish reference
    captions) exists for Wixarika only. Languages without a pilot file are
    skipped rather than raising, so this naturally yields the available pairs.
    """
    triples = []
    for lang in langs:
        try:
            examples = load_split(lang, "pilot")
        except FileNotFoundError:
            print(f"[skip] no pilot split for {lang}")
            continue
        for ex in examples:
            if ex.spanish_caption and ex.image_path.exists():
                triples.append((ex.image_path, INSTRUCTION, ex.spanish_caption))
    return triples


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


def apply_lora(model, r: int | None = None, dropout: float | None = None):
    """Attach LoRA adapters to the decoder attention projections.

    peft freezes all base-model parameters and trains only the adapters, so
    targeting the text-decoder projections keeps the SigLIP vision encoder
    frozen as required. ``r``/``dropout`` override config for sweeps (alpha
    scales with r at the conventional 2x so the effective LR stays comparable).
    """
    from peft import LoraConfig, get_peft_model

    r = r if r is not None else config.LORA_R
    lora = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=dropout if dropout is not None else config.LORA_DROPOUT,
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
    """Format (image, prompt, target) triples into inputs + target-only labels.

    Loss is computed over the assistant target tokens only: the user turn,
    image tokens, per-example prompt, and padding are all masked to -100.
    """
    from PIL import Image

    texts, images, prompt_lens = [], [], []
    for image_path, prompt, target in batch:
        img = Image.open(image_path).convert("RGB")
        full = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]
        texts.append(processor.apply_chat_template(full, add_generation_prompt=False))
        prompt_lens.append(_prompt_length(processor, img, prompt))
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


def train_loop(model, processor, device, pairs, epochs, batch_size, quiet=False,
               checkpoint_dir=None, epoch_offset=0, start_step=0,
               step_checkpoint_every=500):
    """Train the (LoRA-wrapped) model in place over ``pairs``.

    ``checkpoint_dir``: save the adapter after every epoch AND every
    ``step_checkpoint_every`` steps ("step_latest" + train_state.json). Epoch
    snapshots double as a free epochs ablation; the step snapshot bounds a
    spot-preemption loss to minutes, not an epoch — with ~7h epochs on a T4 and
    preemptions arriving faster than that, epoch-only checkpointing can starve
    forever (observed 2026-07-18: three restarts, zero completed epochs).

    Mid-epoch resume (``start_step``): the per-epoch order is drawn with a
    seeded RNG, so the resumed run reproduces the same shuffle and skips the
    first ``start_step`` items. The optimizer restarts fresh (AdamW moments are
    not persisted) — an accepted approximation at LoRA scale.
    """
    import json
    import random as _random

    import torch

    model.train()
    # Gradient checkpointing: the distillation synthesis prompts run ~2k tokens,
    # which OOMs a 16GB T4 even at batch 1 without it. ~30% slower, big memory
    # win; enable_input_require_grads is required for gradients to reach LoRA
    # adapters through checkpointed blocks.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.LEARNING_RATE
    )
    amp_dtype, use_autocast, scaler = amp_context(device)

    def _save(tag: str, epoch_done: int, step: int) -> None:
        ckpt = checkpoint_dir / tag
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        (checkpoint_dir / "train_state.json").write_text(
            json.dumps({"epochs_done": epoch_done, "step": step}))

    for epoch in range(epochs):
        global_epoch = epoch_offset + epoch
        # Seeded shuffle: reproducible order per epoch => resumable mid-epoch.
        order = list(range(len(pairs)))
        _random.Random(SHUFFLE_BASE_SEED + global_epoch).shuffle(order)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

        skip = start_step if epoch == 0 and start_step else 0
        total, n_seen = 0.0, 0
        for step, idxs in enumerate(batches):
            if step < skip:
                continue
            batch = collate([pairs[i] for i in idxs], processor, device)
            loss = train_step(model, batch, optim, scaler, device, amp_dtype, use_autocast)
            total += loss.item()
            n_seen += 1
            if (checkpoint_dir is not None and step_checkpoint_every
                    and n_seen % step_checkpoint_every == 0):
                _save("step_latest", global_epoch, step + 1)
                if not quiet:
                    print(f"  [step {step + 1}/{len(batches)}] mean_loss="
                          f"{total / n_seen:.4f} (step checkpoint)", flush=True)
        if not quiet:
            print(f"  epoch {global_epoch + 1}/{epoch_offset + epochs}: "
                  f"mean_loss={total / max(n_seen, 1):.4f}", flush=True)
        if checkpoint_dir is not None:
            _save(f"epoch{global_epoch + 1}", global_epoch + 1, 0)
            if not quiet:
                print(f"  checkpoint -> {checkpoint_dir / f'epoch{global_epoch + 1}'}",
                      flush=True)
    return model


def generate_caption(model, processor, device, image_path,
                     prompt=INSTRUCTION, max_new_tokens=GEN_MAX_NEW_TOKENS):
    """Greedy-decode a Spanish caption for one image (autocast for dtype safety)."""
    import torch
    from PIL import Image

    amp_dtype = config.dtype(device)
    use_autocast = device in ("cuda", "mps")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
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

    triples = build_examples(langs)
    model, processor, device = load_model_and_processor(model_id)
    scores = []
    for i, (img, prompt, cap) in enumerate(triples):
        hyp = generate_caption(model, processor, device, img, prompt)
        s = chrfpp(hyp, cap)
        scores.append(s)
        print(f"[base {i + 1}/{len(triples)}] chrF++={s:.2f}  hyp={hyp[:60]!r}")
    _report("baseline (off-the-shelf, no LoRA)", scores)
    return scores


def run_loo(model_id, langs, epochs, batch_size, lora_r=None, lora_dropout=None) -> list[float]:
    """Leave-one-out CV: train on n-1 pilot pairs, score the held-out one."""
    from .evaluate import chrfpp

    triples = build_examples(langs)
    n = len(triples)
    scores = []
    for j in range(n):
        held_img, held_prompt, held_cap = triples[j]
        train_triples = triples[:j] + triples[j + 1:]
        model, processor, device = load_model_and_processor(model_id)
        model = apply_lora(model, r=lora_r, dropout=lora_dropout)
        train_loop(model, processor, device, train_triples, epochs, batch_size, quiet=True)
        hyp = generate_caption(model, processor, device, held_img, held_prompt)
        s = chrfpp(hyp, held_cap)
        scores.append(s)
        print(f"[loo {j + 1}/{n}] chrF++={s:.2f}  hyp={hyp[:60]!r}")
        _free(model, device)
    _report(f"SmolVLM+LoRA leave-one-out (n={n}, {epochs} epochs)", scores)
    return scores


def run_distill(model_id, triples_path, epochs, batch_size, out_dir,
                lora_r=None, lora_dropout=None, resume=False):
    """Train on the silver distillation triples and save the adapter.

    ``resume=True`` restarts from the newest ``epochN`` checkpoint under
    ``out_dir`` (adapter weights only — the optimizer restarts fresh for the
    remaining epochs, an accepted simplification for LoRA-scale runs). This is
    the preemption-recovery path for spot VMs.
    """
    from .distill_data import load_triples

    triples = [t for t in load_triples(triples_path) if t[0].exists()]
    if not triples:
        raise SystemExit(f"No triples with resolvable images in {triples_path} — "
                         "run `python -m src.stage1.distill_data` first.")
    print(f"Distilling on {len(triples)} triples from {triples_path}", flush=True)
    model, processor, device = load_model_and_processor(model_id)

    def _load_adapter(ckpt):
        from peft import PeftModel
        m = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
        for p in m.parameters():  # fp32 trainables, as in apply_lora
            if p.requires_grad:
                p.data = p.data.float()
        print(f"[resume] adapter restored from {ckpt}", flush=True)
        return m

    start_epoch, start_step = 0, 0
    if resume:
        # Prefer the newest completed epoch; else the mid-epoch step snapshot.
        for e in range(epochs, 0, -1):
            ckpt = out_dir / f"epoch{e}"
            if (ckpt / "adapter_model.safetensors").exists():
                model = _load_adapter(ckpt)
                start_epoch = e
                break
        else:
            step_ckpt = out_dir / "step_latest"
            state_file = out_dir / "train_state.json"
            if (step_ckpt / "adapter_model.safetensors").exists() and state_file.exists():
                state = json.loads(state_file.read_text())
                model = _load_adapter(step_ckpt)
                start_epoch = state.get("epochs_done", 0)
                start_step = state.get("step", 0)
                print(f"[resume] mid-epoch: epoch {start_epoch + 1}, "
                      f"step {start_step}", flush=True)
    if start_epoch == 0 and start_step == 0 and not isinstance(
            getattr(model, "peft_config", None), dict):
        model = apply_lora(model, r=lora_r, dropout=lora_dropout)

    if epochs - start_epoch > 0:
        train_loop(model, processor, device, triples, epochs - start_epoch,
                   batch_size, checkpoint_dir=out_dir, epoch_offset=start_epoch,
                   start_step=start_step)
    else:
        print(f"[resume] all {epochs} epochs already checkpointed", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"Saved distilled LoRA adapter -> {out_dir}", flush=True)


def run_gold_eval(model_id, langs, adapter=None):
    """Score a model on the 20 gold pilot captions — the clean held-out test.

    Two rows, matching how the paper reports Stage 1:
      * generic — single generic prompt (comparable to the 16.72 base /
        17.55 LOO / 21.0 Qwen numbers);
      * cultural-vqa pipeline — the deployment path (joint category questions +
        synthesis via the backend), with the same meta-prefix strip Stage 2
        applies when it consumes descriptions.
    """
    from .backends import SmolVLMBackend
    from .evaluate import chrfpp
    from .generate_descriptions import _run_cultural_vqa
    from .postprocess import strip_meta_prefix

    triples = build_examples(langs)
    backend = SmolVLMBackend(model_id, adapter=adapter)
    tag = f"adapter={adapter}" if adapter else "off-the-shelf"
    gen_scores, vqa_scores = [], []
    for i, (img, prompt, cap) in enumerate(triples):
        gen_hyp = backend.caption(img, prompt)
        gen_scores.append(chrfpp(gen_hyp, cap))
        desc, _ = _run_cultural_vqa(backend, img, joint=True)
        vqa_hyp = strip_meta_prefix(desc)
        vqa_scores.append(chrfpp(vqa_hyp, cap))
        print(f"[gold {i + 1}/{len(triples)}] generic={gen_scores[-1]:.2f} "
              f"cultural={vqa_scores[-1]:.2f}  hyp={gen_hyp[:50]!r}")
    _report(f"gold-20 generic prompt ({tag})", gen_scores)
    _report(f"gold-20 cultural-VQA pipeline ({tag})", vqa_scores)
    return gen_scores, vqa_scores


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
    ap.add_argument("--epochs", type=int, default=None,
                    help=f"Default {DEFAULT_EPOCHS} (pilot) / "
                         f"{DEFAULT_DISTILL_EPOCHS} (--distill).")
    ap.add_argument("--lora-r", type=int, default=None,
                    help=f"LoRA rank override (default {config.LORA_R}; alpha=2r).")
    ap.add_argument("--lora-dropout", type=float, default=None,
                    help=f"LoRA dropout override (default {config.LORA_DROPOUT}).")
    ap.add_argument("--triples", type=Path,
                    default=config.OUTPUT_DIR / "distill_triples.jsonl",
                    help="Distillation triples JSONL (--distill / --smoke-test).")
    ap.add_argument("--adapter", default=None,
                    help="Adapter dir to evaluate (--gold-eval).")
    ap.add_argument("--adapter-out", type=Path, default=None,
                    help="Where to save the trained adapter "
                         "(default: outputs/adapters, or outputs/adapters/distill).")
    ap.add_argument("--resume", action="store_true",
                    help="--distill: restart from the newest epoch checkpoint "
                         "(spot-VM preemption recovery).")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true",
                      help="Load model, attach LoRA, run 2 steps on 2 examples, then exit.")
    mode.add_argument("--baseline-eval", action="store_true",
                      help="Score the off-the-shelf model (no LoRA) on the pilot Spanish.")
    mode.add_argument("--loo", action="store_true",
                      help="Leave-one-out CV; the honest held-out metric at n=20.")
    mode.add_argument("--distill", action="store_true",
                      help="Train on the silver distillation triples (see distill_data).")
    mode.add_argument("--gold-eval", action="store_true",
                      help="Score --adapter (or the base) on the 20 gold pilot captions.")
    args = ap.parse_args()

    import torch

    model_id = args.model or (config.SMOLVLM_SMOKE_ID if args.smoke_test else config.SMOLVLM_ID)
    epochs = args.epochs if args.epochs is not None else (
        DEFAULT_DISTILL_EPOCHS if args.distill else DEFAULT_EPOCHS)

    if args.baseline_eval:
        run_baseline_eval(model_id, args.langs)
        return
    if args.loo:
        run_loo(model_id, args.langs, epochs, args.batch_size,
                args.lora_r, args.lora_dropout)
        return
    if args.distill:
        out_dir = args.adapter_out or (config.ADAPTER_DIR / "distill")
        run_distill(model_id, args.triples, epochs, args.batch_size, out_dir,
                    args.lora_r, args.lora_dropout, resume=args.resume)
        return
    if args.gold_eval:
        run_gold_eval(model_id, args.langs, adapter=args.adapter)
        return

    if args.smoke_test:
        # Prefer real distillation triples when built (exercises the per-example
        # prompt path); fall back to pilot pairs so the check runs regardless.
        if args.triples.exists():
            from .distill_data import load_triples
            triples = [t for t in load_triples(args.triples) if t[0].exists()][:2]
            print(f"[smoke] using 2 distillation triples from {args.triples}")
        else:
            triples = build_examples(args.langs)[:2]
            print("[smoke] no triples file — using 2 pilot pairs")
        model, processor, device = load_model_and_processor(model_id)
        model = apply_lora(model, r=args.lora_r, dropout=args.lora_dropout)
        model.train()
        optim = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=config.LEARNING_RATE
        )
        amp_dtype, use_autocast, scaler = amp_context(device)
        batch = collate(triples, processor, device)
        for step in range(2):
            loss = train_step(model, batch, optim, scaler, device, amp_dtype, use_autocast)
            print(f"[smoke] step {step}: loss={loss.item():.4f}")
        assert torch.isfinite(loss), "Non-finite loss — training loop is broken."
        print("[smoke] OK — model loads, LoRA attaches, forward/backward/step run.")
        return

    # Default: train the deployed adapter on ALL pilot pairs and save it.
    triples = build_examples(args.langs)
    print(f"Loaded {len(triples)} pilot triples from {args.langs}")
    if not triples:
        raise SystemExit("No pilot pairs found — check the dataset clone.")
    model, processor, device = load_model_and_processor(model_id)
    model = apply_lora(model, r=args.lora_r, dropout=args.lora_dropout)
    train_loop(model, processor, device, triples, epochs, args.batch_size)
    out_dir = args.adapter_out or config.ADAPTER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"Saved LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()
