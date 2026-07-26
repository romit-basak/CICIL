"""Cultural VQA prompting module — the core novel contribution of Stage 1.

Instead of asking a generic VLM for a single caption (which discards cultural
detail), we interrogate the image with a structured set of culturally targeted
questions, then synthesize a culturally grounded Spanish description from the
answers. This mirrors the CIC framework's ask-then-generate approach (Yun &
Kim, 2024), specialized to indigenous cultures of the Americas.

All prompts are written in Spanish to bias the model toward Spanish output,
since the Stage 1 intermediate feeds a Spanish->indigenous translation stage.

NOTE: the category/question bank below is PROVISIONAL. It is grounded in the
four ethnographic categories named in the proposal (ceremony, material culture,
landscape, kinship) plus the example questions in CLAUDE.md. It is structured
as a single dict so Nandita's finalized taxonomy can be swapped in cleanly.
"""

from __future__ import annotations

# Generic baseline: single-shot Spanish caption, no cultural prompting.
# This is the RQ1 control (mirrors the Gators/Dhawan Stage 1 with Qwen2.5-VL).
GENERIC_PROMPT = (
    "Describe esta imagen en español en una o dos oraciones. "
    "Sé concreto y objetivo."
)

# Cultural VQA question bank: category -> ordered list of targeted questions.
CULTURAL_QUESTIONS: dict[str, list[str]] = {
    "ceremony": [
        "¿Se representa alguna ceremonia, ritual o celebración? Si es así, descríbela.",
        "¿Qué prácticas rituales o agrícolas se muestran?",
    ],
    "material_culture": [
        "¿Qué objetos tradicionales o artesanías son visibles (vestimenta, "
        "instrumentos, utensilios, textiles)?",
        "¿De qué materiales parecen estar hechos esos objetos?",
    ],
    "landscape": [
        "¿Cómo es el entorno o paisaje (geografía, vegetación, arquitectura)?",
    ],
    "kinship": [
        "¿Qué personas aparecen y qué relaciones o roles sociales sugieren "
        "(familia, comunidad, edad, género)?",
    ],
}

# Synthesis step: fold the Q&A into one culturally grounded Spanish description.
# v2 (2026-07-02): the v1 prompt ("culturalmente detallada", "una o dos oraciones")
# produced ~105-token, hedged descriptions that diluted ChrF++ against the short gold
# captions and lost to the generic baseline. This version demands a single short,
# direct sentence and explicitly bans filler ("la imagen muestra") and hedging
# ("posiblemente", "parece") — isolating conciseness as the only changed variable.
SYNTHESIS_PROMPT = (
    "A partir de las observaciones siguientes, escribe UNA sola oración en español "
    "(máximo 30 palabras) que describa la imagen nombrando los elementos culturales "
    "concretos (objetos, prácticas, lugar). Sé directo y objetivo: NO uses frases de "
    "relleno como «la imagen muestra» ni expresiones de incertidumbre como "
    "«posiblemente» o «parece». No inventes información que no esté en las "
    "observaciones.\n\nObservaciones:\n{observations}\n\nDescripción (una oración):"
)


def format_synthesis(annotations: dict[str, str]) -> str:
    """Build the synthesis prompt from {category: answer} annotations."""
    lines = [f"- {cat}: {ans}" for cat, ans in annotations.items() if ans]
    return SYNTHESIS_PROMPT.format(observations="\n".join(lines))


# RAG variant (2026-07-26 pilot): same one-sentence contract, but retrieved
# encyclopedia snippets are supplied and — unlike the base prompt — calibrated
# hedging is PERMITTED, narrowly, for retrieved concept names ("posiblemente
# ñandutí"). The base prompt's blanket hedging ban stays for everything else.
# This is a deliberate prompt+context confound in the RAG arm; state it in any
# writeup (see paper_notes.md).
SYNTHESIS_PROMPT_RAG = (
    "A partir de las observaciones y los fragmentos de enciclopedia siguientes, "
    "escribe UNA sola oración en español (máximo 35 palabras) que describa la "
    "imagen nombrando los elementos culturales concretos (objetos, prácticas, "
    "lugar). Si una observación coincide con un concepto cultural descrito en "
    "los fragmentos, nómbralo explícitamente; si la coincidencia no es segura, "
    "márcala con «posiblemente». NUNCA nombres un concepto sin apoyo visual en "
    "las observaciones. No uses frases de relleno como «la imagen muestra».\n\n"
    "Fragmentos de enciclopedia sobre esta cultura (pueden no ser relevantes):\n"
    "{snippets}\n\nObservaciones:\n{observations}\n\nDescripción (una oración):"
)


def format_synthesis_rag(annotations: dict[str, str], snippets: list[str]) -> str:
    """RAG synthesis prompt: observations + retrieved encyclopedia snippets."""
    obs = [f"- {cat}: {ans}" for cat, ans in annotations.items() if ans]
    snips = [f"- {s}" for s in snippets]
    return SYNTHESIS_PROMPT_RAG.format(observations="\n".join(obs),
                                       snippets="\n".join(snips) or "- (ninguno)")


# Distillation: when silver-captioning scraped images, the *teacher* may be given
# the source's encyclopedic description as context — it names ceremonies, objects
# and places the image alone might not reveal. The student never sees this prefix
# (no such context exists at deployment), so training prompts stay unchanged.
CONTEXT_PREFIX = (
    "Contexto sobre la imagen (puede contener información útil; ignóralo si "
    "contradice lo que ves): {context}\n\n"
)


def with_context(prompt: str, context: str | None) -> str:
    """Prepend source context to a teacher prompt; no-op without context."""
    if not context or not context.strip():
        return prompt
    return CONTEXT_PREFIX.format(context=context.strip()) + prompt


def joint_question(category: str) -> str:
    """One combined prompt per category (matches category-level training pairs)."""
    return " ".join(CULTURAL_QUESTIONS[category])
