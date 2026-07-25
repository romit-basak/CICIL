"""Stage 2 — STRETCH / SCAFFOLD: NLLB-200 fine-tune as a Stage 2 alternative
for Guaraní, to compare against the retrieval+Gemini pipeline (translate.py).

STATUS: scaffold only. Not runnable as-is:
  - Needs `transformers`, `datasets`, `sentencepiece`, `accelerate` (transformers/
    datasets/accelerate are already project dependencies; sentencepiece is not
    listed, but this exact tokenizer has been verified to load without it in
    this repo's venv -- add it only if a future transformers version needs it).
  - Needs the actual Guaraní parallel data (AmericasNLP 2023 ~53k pairs +
    MultiScript30k) landed and reformatted -- see build_index.py's docstring
    on that data's status. This script assumes it already exists as
    ``data/guarani_train.jsonl`` / ``data/guarani_dev.jsonl`` in the
    {"spanish": ..., "target": ...} format `_load_pairs` in build_index.py
    already uses, so the same raw-data reformatting work serves both the
    retrieval bank AND this fine-tune.
  - Untested end-to-end (needs the real data above). Treat every
    hyperparameter here as a reasonable starting point, not a tuned value.

WHY THIS EXISTS: Stage 2's primary approach (retrieval-augmented Gemini
prompting) has an obvious alternative worth benchmarking against -- actually
fine-tuning a dedicated MT model on the same parallel data instead of using
it as few-shot context for a much larger general-purpose model. NLLB-200 is
the natural choice: Guaraní (``grn_Latn``) is one of its 200 supported
languages, so this is fine-tuning an existing direction rather than training
a new language pair from scratch.

CAVEAT if extended to Bribri/Maya: NLLB-200's tokenizer does NOT error on an
unrecognized FLORES-200 code -- confirmed empirically (this exact model/
tokenizer, this exact venv) that ``bzd_Latn`` (Bribri) and ``yua_Latn``
(Yucatec Maya) are not in this model's vocabulary and silently resolve to
``<unk>`` (id 3), NOT ``<s>``/BOS (id 0). A silent ``<unk>`` as
``forced_bos_token_id`` makes ``generate()`` emit ordinary (often English)
text with no error -- much harder to catch than a crash. See
``_assert_lang_in_vocab()`` below, called before any train/translate run.

USAGE (once deps + data exist):
    uv run python -m src.stage2.nllb_finetune_guarani --train
    uv run python -m src.stage2.nllb_finetune_guarani --translate --checkpoint checkpoints/nllb-grn/best

Then score the output .txt the same way as translate.py's predictions:
    uv run python -m src.stage1.evaluate --lang guarani --translations predictions/guarani_nllb_predictions.txt
"""

from __future__ import annotations

import argparse
import json

from .paths import PRED_DIR

# =============================================================
# CONFIG
# =============================================================

# facebook/nllb-200-distilled-600M is the practical starting point: small
# enough to fine-tune on a single GPU, still covers all 200 FLORES-200
# languages including Guaraní. Swap for -1.3B / -3.3B if compute allows --
# larger NLLB checkpoints score meaningfully higher on low-resource pairs in
# the original NLLB paper's own evals.
BASE_MODEL = "facebook/nllb-200-distilled-600M"

SRC_LANG = "spa_Latn"   # NLLB FLORES-200 code for Spanish
TGT_LANG = "grn_Latn"   # NLLB FLORES-200 code for Guaraní -- verified present in vocab

TRAIN_FILE = "data/guarani_train.jsonl"   # {"spanish": ..., "target": ...} per line
DEV_FILE = "data/guarani_dev.jsonl"

CHECKPOINT_DIR = "checkpoints/nllb-grn"

# Conservative low-resource fine-tune defaults: small LR + few epochs to
# avoid catastrophic forgetting of NLLB's existing (weak-but-nonzero)
# Guaraní capability. Tune against dev ChrF++ before trusting these.
TRAIN_CONFIG = {
    "learning_rate": 3e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_source_length": 128,
    "max_target_length": 128,
    "fp16": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
}


# =============================================================
# Vocab safety guard
# =============================================================

def _assert_lang_in_vocab(tokenizer, lang_code: str) -> None:
    """Fail loudly if lang_code isn't a real token in this tokenizer's vocab.

    NLLB does not raise on an unknown FLORES-200 code; it silently maps it to
    <unk> (confirmed: id 3, not <s>/BOS id 0). A silent <unk> as
    forced_bos_token_id makes generate() produce ordinary (often English) text
    with no error -- much harder to catch than a crash.
    """
    unk_id = tokenizer.unk_token_id
    resolved_id = tokenizer.convert_tokens_to_ids(lang_code)
    if resolved_id == unk_id:
        raise ValueError(
            f"{lang_code!r} is not in {BASE_MODEL}'s vocabulary (resolves to "
            f"<unk>, id {unk_id}). Using it as forced_bos_token_id would silently "
            f"steer generation into the wrong language with no error. Use a real "
            f"FLORES-200 code, or add it as a new special token and retrain first."
        )


# =============================================================
# Data
# =============================================================

def load_pairs(path: str) -> list[dict]:
    """Load {"spanish": ..., "target": ...} pairs -- same shape build_index.py
    already produces, so no separate reformatting step is needed for this."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_hf_dataset(pairs: list[dict]):
    """Wrap pairs in a HF Dataset with the raw text columns; tokenization
    happens in a separate map() step so the tokenizer (and its src/tgt lang
    tokens) is loaded once, in main(), not per-call here."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "spanish": [p["spanish"] for p in pairs],
            "target": [p["target"] for p in pairs],
        }
    )


def make_tokenize_fn(tokenizer, max_source_length: int, max_target_length: int):
    def _tokenize(batch):
        tokenizer.src_lang = SRC_LANG
        model_inputs = tokenizer(
            batch["spanish"], max_length=max_source_length, truncation=True
        )
        tokenizer.src_lang = TGT_LANG  # NLLB targets are tokenized in their own lang mode
        labels = tokenizer(
            batch["target"], max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _tokenize


# =============================================================
# Train
# =============================================================

def train():
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG)
    _assert_lang_in_vocab(tokenizer, SRC_LANG)
    _assert_lang_in_vocab(tokenizer, TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    print(f"Loading data: {TRAIN_FILE} / {DEV_FILE}")
    train_pairs = load_pairs(TRAIN_FILE)
    dev_pairs = load_pairs(DEV_FILE)
    print(f"  train={len(train_pairs):,}  dev={len(dev_pairs):,}")

    train_ds = build_hf_dataset(train_pairs)
    dev_ds = build_hf_dataset(dev_pairs)

    tokenize_fn = make_tokenize_fn(
        tokenizer, TRAIN_CONFIG["max_source_length"], TRAIN_CONFIG["max_target_length"]
    )
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["spanish", "target"])
    dev_ds = dev_ds.map(tokenize_fn, batched=True, remove_columns=["spanish", "target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        fp16=TRAIN_CONFIG["fp16"],
        eval_strategy=TRAIN_CONFIG["eval_strategy"],
        save_strategy=TRAIN_CONFIG["save_strategy"],
        load_best_model_at_end=TRAIN_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAIN_CONFIG["metric_for_best_model"],
        predict_with_generate=True,
        report_to=[],  # TODO: wire to wandb/tensorboard if the team wants run tracking
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    best_path = f"{CHECKPOINT_DIR}/best"
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"Saved best checkpoint -> {best_path}")


# =============================================================
# Translate (inference), matching translate.py's output contract so
# run_ablations.py can score it identically to the Gemini predictions
# =============================================================

def translate(checkpoint: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading fine-tuned checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang=SRC_LANG)
    _assert_lang_in_vocab(tokenizer, SRC_LANG)
    _assert_lang_in_vocab(tokenizer, TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    dev_pairs = load_pairs(DEV_FILE)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    predictions = []
    for pair in dev_pairs:
        inputs = tokenizer(pair["spanish"], return_tensors="pt", truncation=True)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=TRAIN_CONFIG["max_target_length"],
        )
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        predictions.append(text)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PRED_DIR / "guarani_nllb_predictions.txt"
    with out_file.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")
    print(f"Written {len(predictions)} lines -> {out_file}")
    print(
        "Score with:\n"
        f"    uv run python -m src.stage1.evaluate --lang guarani --translations {out_file}"
    )


# =============================================================
# Entry point
# =============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="STRETCH SCAFFOLD: NLLB-200 fine-tune for Guarani (Stage 2 alternative)"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", action="store_true", help="Fine-tune NLLB-200 on Guarani data")
    g.add_argument("--translate", action="store_true", help="Translate the dev split with a fine-tuned checkpoint")
    p.add_argument("--checkpoint", default=f"{CHECKPOINT_DIR}/best", help="Checkpoint path for --translate")
    return p.parse_args()


def main():
    args = parse_args()
    if args.train:
        train()
    else:
        translate(args.checkpoint)


if __name__ == "__main__":
    main()
