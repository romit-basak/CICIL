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

from tqdm import tqdm

from . import config, vqa_prompts
from .backends import get_backend
from .data_io import load_split


def _run_generic(backend, image_path) -> tuple[str, dict]:
    return backend.caption(image_path, vqa_prompts.GENERIC_PROMPT), {}


def _run_cultural_vqa(backend, image_path) -> tuple[str, dict]:
    annotations: dict[str, str] = {}
    for category, questions in vqa_prompts.CULTURAL_QUESTIONS.items():
        answers = [backend.caption(image_path, q) for q in questions]
        annotations[category] = " ".join(a for a in answers if a)
    description = backend.caption(
        image_path, vqa_prompts.format_synthesis(annotations)
    )
    return description, annotations


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage 1 Spanish descriptions.")
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--split", default="dev", choices=["pilot", "dev", "test"])
    ap.add_argument("--mode", default="generic", choices=["generic", "cultural-vqa"])
    ap.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    ap.add_argument("--model", default=None, help="Override model id/tag for the backend.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="Sampling temperature (ollama only). Use 0 for deterministic decoding.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed (ollama only); pair with --temperature 0 for reproducibility.")
    args = ap.parse_args()

    examples = load_split(args.lang, args.split)
    if args.limit is not None:
        examples = examples[: args.limit]

    backend = get_backend(args.backend, args.model,
                          temperature=args.temperature, seed=args.seed)
    runner = _run_generic if args.mode == "generic" else _run_cultural_vqa

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / f"{args.lang}_{args.split}_{args.mode}_{args.backend}.jsonl"

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc=f"{args.lang}/{args.mode}"):
            if not ex.image_path.exists():
                tqdm.write(f"[skip] missing image for {ex.id}: {ex.image_path}")
                continue
            try:
                description, annotations = runner(backend, ex.image_path)
            except Exception as e:  # noqa: BLE001 - one bad image must not abort the run
                tqdm.write(f"[error] {ex.id} ({ex.image_path.name}): {e}")
                description, annotations = "", {}
            f.write(json.dumps({
                "id": ex.id,
                "filename": ex.image_path.name,
                "language": ex.language,
                "iso_lang": ex.iso_lang,
                "mode": args.mode,
                "backend": args.backend,
                "generated_spanish": description,
                "cultural_annotations": annotations,
            }, ensure_ascii=False) + "\n")
            f.flush()  # keep the file inspectable mid-run and crash-resilient
            n_written += 1

    print(f"Wrote {n_written} descriptions -> {out_path}")


if __name__ == "__main__":
    main()
