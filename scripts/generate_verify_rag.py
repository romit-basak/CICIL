"""Run the verify-RAG arm (prototype v3, hardened) over full splits.

Promotes analysis/human_eval/prototype_verify_rag.py from a 3-image
qualitative check to a full eval arm, tag ``smolvlm-verify``:

  1. Legible text: closed gate -> transcribe -> reject descriptive answers ->
     closed confirm; locked text is spliced verbatim into the final caption.
  2. Patterns/symbols question (null-permitting) -> text-RAG retrieval on that
     answer alone.
  3. If the top hit clears MIN_VERIFY_SCORE, one closed verification question
     ("Wikipedia says X is identified by [feature] -- do you see it? SI/NO/
     INCIERTO"); the concept is named only on SI, hedged on INCIERTO.
  4. Final caption assembled in Python (generic base description + verified
     splices) -- no open-ended synthesis call.

Differences from the prototype: the 4 diagnostic category questions are
dropped (the assembly never used them; ~2x fewer calls), and each record
carries a ``verify_rag`` block with every intermediate so failures can be
audited per image. Deterministic like the other arms (greedy decoding in the
smolvlm backend). Resumable (appends; skips ids already present); refuses to
touch an existing output without --overwrite (see the filename-collision
guard rationale in generate_descriptions.py).

Run:
  uv run python scripts/generate_verify_rag.py --limit 1                # smoke
  uv run python scripts/generate_verify_rag.py                          # all 5 langs, dev
  uv run python scripts/generate_verify_rag.py --langs wixarika --split pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # noqa: F401 -- see rag_context.py: must load before faiss on macOS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.human_eval.prototype_verify_rag import (  # noqa: E402
    LEGIBLE_GATE_QUESTION,
    MIN_VERIFY_SCORE,
    PATTERN_QUESTION,
    TRANSCRIBE_QUESTION,
    clean_transcription,
    extract_diagnostic_claim,
    parse_closed_answer,
)

OUTPUTS = ROOT / "outputs"
LANGS = ["guarani", "maya", "bribri", "wixarika", "nahuatl"]
BACKEND_TAG = "smolvlm-verify"


def caption_image(backend, text_bank, image_path: Path) -> dict:
    """One image through the verify-RAG flow; returns final + all intermediates."""
    from src.stage1 import vqa_prompts

    out: dict = {}

    # Step 1: legible text (gate -> transcribe -> reject -> confirm).
    locked_text = None
    gate = backend.caption(image_path, LEGIBLE_GATE_QUESTION)
    out["gate"] = gate
    if parse_closed_answer(gate) == "SI":
        raw = backend.caption(image_path, TRANSCRIBE_QUESTION)
        out["transcription_raw"] = raw
        text = clean_transcription(raw)
        if text is not None:
            confirm = backend.caption(
                image_path,
                f'¿La imagen contiene exactamente el texto "{text}"? '
                f"Responde únicamente con una palabra: SI o NO.")
            out["transcription_confirm"] = confirm
            if parse_closed_answer(confirm) == "SI":
                locked_text = text
    out["locked_text"] = locked_text

    # Step 2: patterns question -> text-RAG on that answer alone.
    pattern_answer = backend.caption(image_path, PATTERN_QUESTION)
    out["pattern_answer"] = pattern_answer
    hits = text_bank.retrieve(pattern_answer, k=3) if pattern_answer else []
    out["hits"] = [{"title": h["title"], "score": round(h["score"], 4)} for h in hits]

    # Step 3: closed verification on the top hit's diagnostic claim.
    verify_answer, concept_name, verdict = None, None, None
    top = hits[0] if hits else None
    if top and top["score"] >= MIN_VERIFY_SCORE:
        claim = extract_diagnostic_claim(top["extract"]) or top["extract"][:150]
        concept_name = top["title"]
        verify_answer = backend.caption(
            image_path,
            f"Wikipedia describe {concept_name} así: \"{claim}\" "
            f"¿Se observa específicamente esto en la imagen? Responde "
            f"únicamente con una palabra: SI, NO, o INCIERTO, seguida de "
            f"una breve razón.")
        verdict = parse_closed_answer(verify_answer)
    out.update(concept=concept_name, verify_answer=verify_answer, verdict=verdict)

    # Step 4: Python-assembled final caption.
    base = backend.caption(image_path, vqa_prompts.GENERIC_PROMPT)
    out["base"] = base
    parts = [base.rstrip(".")]
    if verdict == "SI":
        parts.append(f"posiblemente {concept_name}")
    elif verdict == "INCIERTO":
        parts.append(f"posiblemente relacionado con {concept_name}")
    if locked_text:
        parts.append(f'texto visible: "{locked_text}"')
    out["final"] = ". ".join(parts) + "."
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    parser.add_argument("--split", default="dev", choices=["dev", "pilot"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="start the output file over instead of resuming")
    args = parser.parse_args()

    from src.stage1.backends import get_backend
    from src.stage1.data_io import load_split
    from src.stage1.rag_context import TextBank

    backend = get_backend("smolvlm", adapter=str(OUTPUTS / "adapters" / "distill_full"))

    for lang in args.langs:
        out_path = OUTPUTS / f"{lang}_{args.split}_cultural-vqa_{BACKEND_TAG}.jsonl"
        if out_path.exists() and args.overwrite:
            out_path.unlink()
        done: set[str] = set()
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                done = {json.loads(line)["id"] for line in f if line.strip()}
            print(f"[{lang}] resuming: {len(done)} already done in {out_path.name}")

        examples = load_split(lang, args.split)
        if args.limit:
            examples = examples[: args.limit]
        todo = [ex for ex in examples if ex.id not in done]
        print(f"[{lang}] {args.split}: {len(todo)} to run ({len(examples)} total)")

        text_bank = TextBank(lang)
        with out_path.open("a", encoding="utf-8") as f:
            for i, ex in enumerate(todo, 1):
                result = caption_image(backend, text_bank, ex.image_path)
                record = {
                    "id": ex.id,
                    "filename": ex.image_path.name,
                    "language": ex.language,
                    "iso_lang": ex.iso_lang,
                    "mode": "cultural-vqa",
                    "backend": BACKEND_TAG,
                    "generated_spanish": result["final"],
                    # patterns answer doubles as the cultural query for Stage 2
                    "cultural_annotations": {"patterns": result["pattern_answer"]},
                    "verify_rag": {k: v for k, v in result.items() if k != "final"},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                flags = []
                if result["locked_text"]:
                    flags.append(f'text="{result["locked_text"]}"')
                if result["verdict"]:
                    flags.append(f'{result["verdict"]}:{result["concept"]}')
                print(f"  [{lang} {i}/{len(todo)}] {ex.id} "
                      f"{' '.join(flags) or '(base only)'}", flush=True)

    print("done.")


if __name__ == "__main__":
    main()
