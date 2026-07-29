"""Prototype: a targeted pattern/symbol VQA question feeding text-RAG,
instead of one open-ended verbose description (prototype_text_rag.py).

Motivated by a gap found in the existing question bank while reviewing that
first prototype: `CULTURAL_QUESTIONS["material_culture"]` asks WHAT object is
visible and WHAT MATERIAL it's made of, but never asks about surface-level
pattern/motif/symbol detail -- exactly the level of specificity that found
ñandutí in the first prototype ("radial geometric pattern imitating a spider
web"). Idea: keep the existing 4 category questions unchanged (reuses the real
pipeline's annotations dict and format_synthesis_rag as-is), add ONE new
targeted question asked on its own (not bundled, so it gets a dedicated,
focused answer rather than sharing a response with two other sub-questions),
and use THAT answer specifically as the text-RAG query instead of joining all
4 categories together (which dilutes the signal: a landscape/kinship/ceremony
answer mixed into the query moves the embedding away from a pattern-specific
match).

Qualitative check only, same 3 images as prototype_text_rag.py for direct
comparison, plus hch_021 doubles as a null-case check: a bare canyon has no
patterns, so a good answer here is "no patterns visible," not a hallucinated
one. Does not modify vqa_prompts.py or generate_descriptions.py -- self-
contained so nothing else can regress.

Run:
    uv run python -m analysis.human_eval.prototype_vqa_rag
"""

from __future__ import annotations

import json
from pathlib import Path

import torch  # noqa: F401 -- see rag_context.py: must load before faiss on macOS

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

CASES = [
    ("guarani", "grn_019", "smolvlm-rag"),
    ("wixarika", "hch_021", "smolvlm-rag"),
    ("bribri", "bzd_042", "smolvlm-rag"),
]

PATTERN_QUESTION = (
    "¿Qué patrones, símbolos o diseños geométricos se observan en los objetos, "
    "vestimenta o superficies de la imagen, si los hay? Si no se observa ningún "
    "patrón o diseño distintivo, responde explícitamente que no hay ninguno."
)


def main() -> None:
    from src.stage1 import vqa_prompts
    from src.stage1.backends import get_backend
    from src.stage1.data_io import load_split
    from src.stage1.rag_context import TextBank

    backend = get_backend("smolvlm", adapter=str(OUTPUTS / "adapters" / "distill_full"))
    text_banks: dict[str, TextBank] = {}
    old_captions: dict[str, dict] = {}
    for culture, _, tag in CASES:
        if culture not in old_captions:
            path = OUTPUTS / f"{culture}_dev_cultural-vqa_{tag}.jsonl"
            old_captions[culture] = {r["id"]: r for r in
                                     (json.loads(l) for l in path.open(encoding="utf-8"))}

    for culture, image_id, tag in CASES:
        ex = next(e for e in load_split(culture, "dev") if e.id == image_id)
        if culture not in text_banks:
            text_banks[culture] = TextBank(culture)

        print(f"\n{'=' * 70}\n{image_id} ({culture})\n{'=' * 70}")
        old = old_captions[culture].get(image_id, {})
        print(f"OLD (smolvlm-rag, image-CBIR channel):\n  {old.get('generated_spanish', '(missing)')}")

        # Same 4 existing categories, unchanged -- real production questions.
        annotations: dict[str, str] = {}
        for category, questions in vqa_prompts.CULTURAL_QUESTIONS.items():
            answers = [backend.caption(ex.image_path, vqa_prompts.joint_question(category))]
            annotations[category] = " ".join(a for a in answers if a)
        print("\nSTEP 1 -- existing 4 category answers:")
        for cat, ans in annotations.items():
            print(f"  {cat}: {ans}")

        # NEW: one targeted, standalone question -- not bundled with the others.
        pattern_answer = backend.caption(ex.image_path, PATTERN_QUESTION)
        annotations["patterns"] = pattern_answer
        print(f"\nSTEP 2 -- NEW targeted question (\"patterns/symbols\"):\n  {pattern_answer}")

        # Retrieval query = ONLY the pattern answer, not all 5 joined -- tests
        # whether a focused query beats the diluted combined-annotations query
        # the real pipeline currently uses.
        hits = text_banks[culture].retrieve(pattern_answer, k=3) if pattern_answer else []
        snippets = [f"{h['title']}: {h['extract'][:200]}" for h in hits]
        print("\nSTEP 3 -- text-RAG retrieval (query = ONLY the pattern answer):")
        for h, s in zip(hits, snippets):
            print(f"  {h['score']:.2f} -- {s}")

        # Reuse the REAL production synthesis function unmodified -- annotations
        # dict just has one extra "patterns" key now, which it already handles
        # generically.
        synthesis_prompt = vqa_prompts.format_synthesis_rag(annotations, snippets, culture)
        new_caption = backend.caption(ex.image_path, synthesis_prompt)
        print(f"\nSTEP 4 -- final caption (NEW, via real format_synthesis_rag):\n  {new_caption}")

    print(f"\n{'=' * 70}\nDone. Qualitative comparison only -- no new eval numbers.")


if __name__ == "__main__":
    main()
