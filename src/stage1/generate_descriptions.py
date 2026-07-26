"""Generate Stage 1 Spanish descriptions over a CICIL split.

Two modes (both needed for RQ1):
  * generic      — one captioning prompt -> Spanish description (baseline control).
  * cultural-vqa — ask the structured cultural questions, then synthesize a
                   culturally grounded description; emits cultural_annotations
                   that become Stage 2 retrieval keys.

Output JSONL (the hand-off contract to Mehek's Stage 2) has one record per image:
  {id, filename, language, iso_lang, mode, backend,
   generated_spanish, cultural_annotations}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from . import config, vqa_prompts
from .backends import get_backend
from .data_io import load_split


def _run_generic(backend, image_path, context: str | None = None) -> tuple[str, dict, dict]:
    prompt = vqa_prompts.with_context(vqa_prompts.GENERIC_PROMPT, context)
    return backend.caption(image_path, prompt), {}, {}


def _run_cultural_vqa(backend, image_path, context: str | None = None,
                      joint: bool = False, text_bank=None) -> tuple[str, dict, dict]:
    """Ask the cultural questions, then synthesize a grounded description.

    ``context`` prepends retrieved/source metadata to every prompt (CBIR image
    neighbors at deployment, encyclopedic source text during silver-captioning).
    ``joint`` asks each category's questions in one call — matching the
    category-level distillation pairs the student trains on (and cutting 7
    calls/image to 5). ``text_bank`` (a rag_context.TextBank) switches the
    synthesis step to the RAG prompt: the VQA answers themselves query the
    culture's Wikipedia bank, and retrieved snippets are supplied with
    calibrated-hedging instructions ("posiblemente X" for uncertain matches).
    """
    annotations: dict[str, str] = {}
    for category, questions in vqa_prompts.CULTURAL_QUESTIONS.items():
        if joint:
            prompts = [vqa_prompts.joint_question(category)]
        else:
            prompts = questions
        answers = [backend.caption(image_path, vqa_prompts.with_context(p, context))
                   for p in prompts]
        annotations[category] = " ".join(a for a in answers if a)

    extras: dict = {}
    if text_bank is not None:
        query = " ".join(v for v in annotations.values() if v)
        hits = text_bank.retrieve(query) if query else []
        snippets = [f"{h['title']}: {h['extract'][:200]}" for h in hits]
        extras["text_rag_snippets"] = snippets
        synthesis = vqa_prompts.format_synthesis_rag(annotations, snippets)
    else:
        synthesis = vqa_prompts.format_synthesis(annotations)
    description = backend.caption(
        image_path, vqa_prompts.with_context(synthesis, context),
    )
    return description, annotations, extras


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage 1 Spanish descriptions.")
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--split", default="dev", choices=["pilot", "dev", "test"])
    ap.add_argument("--mode", default="generic", choices=["generic", "cultural-vqa"])
    ap.add_argument("--backend", default="ollama",
                    choices=["ollama", "hf", "smolvlm", "smolvlm-noctx",
                             "smolvlm-rag", "ollama-rag"],
                    help="Model backend. Suffixed variants ('smolvlm-noctx', "
                         "'smolvlm-rag', 'ollama-rag') run the same base code path "
                         "but tag the output filename honestly for the run variant — "
                         "no more post-hoc renames.")
    ap.add_argument("--model", default=None, help="Override model id/tag for the backend.")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir (smolvlm backend only) for the fine-tuned Stage 1 model.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    ap.add_argument("--joint-questions", action="store_true",
                    help="cultural-vqa: one call per category (questions combined) — "
                         "matches the distilled student's training pairs.")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="Sampling temperature (ollama only). Use 0 for deterministic decoding.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed (ollama only); pair with --temperature 0 for reproducibility.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting an existing output JSONL. Off by default: "
                         "the output filename is derived from (lang, split, mode, "
                         "backend), and reusing a backend tag for a different run has "
                         "already silently destroyed a finished file once.")
    ap.add_argument("--context-json", type=Path, default=None,
                    help="JSON file mapping example id -> context string (e.g. CBIR "
                         "image-neighbor context from src.stage1.rag_context --lookup). "
                         "Injected into every prompt via vqa_prompts.with_context.")
    ap.add_argument("--text-rag", action="store_true",
                    help="cultural-vqa only: query the culture's Wikipedia text bank "
                         "with the VQA answers and synthesize with the RAG prompt "
                         "(calibrated hedging). Needs indices/wikitext_<lang>.index.")
    args = ap.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / f"{args.lang}_{args.split}_{args.mode}_{args.backend}.jsonl"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"ABORT: {out_path} already exists ({sum(1 for _ in out_path.open())} lines). "
            f"Pass --overwrite to replace it, or use a different --backend tag."
        )

    examples = load_split(args.lang, args.split)
    if args.limit is not None:
        examples = examples[: args.limit]

    # Suffixed tags (smolvlm-noctx/-rag, ollama-rag) share their base code path;
    # the tag only differentiates the output filename (which run variant made it).
    if args.backend.startswith("smolvlm"):
        backend_key = "smolvlm"
    elif args.backend.startswith("ollama"):
        backend_key = "ollama"
    else:
        backend_key = args.backend
    backend = get_backend(backend_key, args.model,
                          temperature=args.temperature, seed=args.seed,
                          adapter=args.adapter)

    ctx: dict[str, str] = {}
    if args.context_json:
        ctx = json.loads(args.context_json.read_text(encoding="utf-8"))
        print(f"Loaded CBIR context for {len(ctx)} images from {args.context_json}")

    text_bank = None
    if args.text_rag:
        if args.mode != "cultural-vqa":
            raise SystemExit("--text-rag only applies to --mode cultural-vqa.")
        from .rag_context import TextBank
        text_bank = TextBank(args.lang)
        print(f"Loaded Wikipedia text bank for {args.lang} "
              f"({text_bank.index.ntotal} extracts)")

    if args.mode == "generic":
        def runner(b, p, c=None):
            return _run_generic(b, p, context=c)
    else:
        def runner(b, p, c=None):  # noqa: E731 - closes over joint/text_bank
            return _run_cultural_vqa(b, p, context=c, joint=args.joint_questions,
                                     text_bank=text_bank)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc=f"{args.lang}/{args.mode}"):
            if not ex.image_path.exists():
                tqdm.write(f"[skip] missing image for {ex.id}: {ex.image_path}")
                continue
            try:
                description, annotations, extras = runner(
                    backend, ex.image_path, ctx.get(ex.id))
            except Exception as e:  # noqa: BLE001 - one bad image must not abort the run
                tqdm.write(f"[error] {ex.id} ({ex.image_path.name}): {e}")
                description, annotations, extras = "", {}, {}
            record = {
                "id": ex.id,
                "filename": ex.image_path.name,
                "language": ex.language,
                "iso_lang": ex.iso_lang,
                "mode": args.mode,
                "backend": args.backend,
                "generated_spanish": description,
                "cultural_annotations": annotations,
            }
            if ctx:
                record["cbir_context"] = ctx.get(ex.id, "")
            if extras.get("text_rag_snippets") is not None:
                record["text_rag_snippets"] = extras["text_rag_snippets"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # keep the file inspectable mid-run and crash-resilient
            n_written += 1

    print(f"Wrote {n_written} descriptions -> {out_path}")


if __name__ == "__main__":
    main()
