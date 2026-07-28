"""Prototype: verbose-observation + text-only RAG reasoning, on the 3 known
CBIR-failure images (grn_019/nandutí, hch_021/Wirikuta, bzd_042/Cahuita).

Motivated by the 2026-07-28 CBIR reliability finding (3 of 4 spot-checked
image-retrieval neighbors were visual mismatches copied into captions anyway).
Idea: skip image retrieval, make Stage 1 describe visual DETAIL verbosely
first ("spiral thread patterns", not "textile"), text-search Wikipedia on that
description (MiniLM matches concepts, not pixels), then have the model reason
explicitly over the retrieved snippets -- and preserve any fact it's already
certain of (e.g. legible poster text) rather than let retrieval override it.

This is a qualitative demonstration, not a replacement pipeline: 3 images, one
backend, no training, no new eval numbers. Does NOT modify vqa_prompts.py or
any production prompt -- self-contained here so nothing else can regress.

Run:
    uv run python -m analysis.human_eval.prototype_text_rag
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

VERBOSE_OBSERVATION_PROMPT = (
    "Describe en español, con el máximo detalle visual posible, ÚNICAMENTE lo "
    "que se ve en la imagen: formas, patrones, colores, texturas, disposición "
    "espacial, y cualquier texto legible (carteles, letreros). NO nombres "
    "ninguna cultura, lugar o artesanía específica todavía -- solo describe "
    "los elementos visuales en bruto, como lo haría alguien sin conocimiento "
    "cultural previo. 3-4 oraciones."
)

REASONING_SYNTHESIS_PROMPT = (
    "Tienes una descripción visual detallada de una imagen de la cultura "
    "{culture} y fragmentos de Wikipedia recuperados a partir de esa "
    "descripción (pueden no ser relevantes). Razona brevemente en 2-3 pasos: "
    "(1) identifica qué detalle visual observado podría corresponder a un "
    "concepto cultural descrito en los fragmentos; (2) si la correspondencia "
    "es clara y específica, nómbralo directamente; si es incierta o solo "
    "temática, usa «posiblemente»; (3) si la descripción visual ya menciona un "
    "hecho legible con certeza (texto de un cartel, un letrero, una fecha), "
    "NUNCA lo reemplaces por algo de los fragmentos -- los fragmentos solo "
    "pueden añadir información, no contradecir lo ya observado con certeza. "
    "Termina con UNA oración final de descripción (máx 35 palabras).\n\n"
    "Descripción visual:\n{observation}\n\n"
    "Fragmentos de Wikipedia recuperados:\n{snippets}\n\n"
    "Razonamiento y descripción final:"
)


def main() -> None:
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

        observation = backend.caption(ex.image_path, VERBOSE_OBSERVATION_PROMPT)
        print(f"\nSTEP 1 -- verbose observation:\n  {observation}")

        hits = text_banks[culture].retrieve(observation, k=3)
        snippets = [f"{h['title']}: {h['extract'][:200]}" for h in hits]
        print("\nSTEP 2 -- text-RAG retrieval (query = the observation above):")
        for h, s in zip(hits, snippets):
            print(f"  {h['score']:.2f} -- {s}")

        prompt = REASONING_SYNTHESIS_PROMPT.format(
            culture=culture, observation=observation,
            snippets="\n".join(f"- {s}" for s in snippets) or "- (ninguno)")
        new_caption = backend.caption(ex.image_path, prompt)
        print(f"\nSTEP 3 -- reasoning + final caption (NEW):\n  {new_caption}")

    print(f"\n{'=' * 70}\nDone. Qualitative comparison only -- no new eval numbers.")


if __name__ == "__main__":
    main()
