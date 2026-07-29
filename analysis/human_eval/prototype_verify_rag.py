"""Prototype v3: preserve verified facts, verify (don't re-synthesize) the rest.

Both prior prototypes regenerate the ENTIRE caption from scratch in one open
"write a sentence incorporating relevant context" call -- which is exactly how
v1's hch_021 case overwrote a correct poster-reading with a fabricated wrong
one (documented in STAGE1_HANDOFF.md's retraction), and how v1/v2 both failed
to act on a dead-on-correct ñandutí retrieval. Both failures share a cause:
open-ended synthesis is a hard task for a 2B model to execute reliably.

v3 changes the shape of the problem instead of the prompt wording:
1. Extract any LEGIBLE TEXT (poster, sign) via a hardened four-step flow
   (closed gate -> transcribe -> reject descriptive answers -> closed
   confirm); if it survives, it is spliced into the final caption verbatim,
   not re-derived through a call that could corrupt it.
2. Ask the (v2) patterns/symbols question as before; retrieve on it.
3. NEW: turn the top retrieved snippet's most specific diagnostic claim into
   an explicit yes/no/unsure VERIFICATION question asked directly against the
   image ("Wikipedia says X is identified by [specific feature] -- do you see
   [specific feature] in this image?") -- a closed comparison task, not open
   synthesis.
4. Assemble the final caption in Python, not by asking the model to do it:
   legible text goes in verbatim; the retrieved concept is named only if the
   verification step said yes (hedged if "unsure," omitted if "no"). This
   removes the open-ended synthesis step for the parts we can control
   entirely, and reserves model judgment for exactly one narrow, closed
   question.

3 cases chosen for what each isolates:
  - grn_025: does the poster fact ("Corrientes"/"Mundial de Chamamé") survive
    untouched now that it's never re-synthesized?
  - grn_019: does an explicit closed verification question finally get
    "ñandutí" named, where two open-synthesis attempts didn't?
  - hch_021: does the null case still correctly avoid fabricating a specific
    site now that there's no snippet with a strong enough diagnostic claim to
    ask about?

Qualitative only, 3 images, no training, no changes to production code.

Run:
    uv run python -m analysis.human_eval.prototype_verify_rag
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import torch  # noqa: F401 -- see rag_context.py: must load before faiss on macOS

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

CASES = [
    ("guarani", "grn_025", "smolvlm-rag"),
    ("guarani", "grn_019", "smolvlm-rag"),
    ("wixarika", "hch_021", "smolvlm-rag"),
]

# Hardened legible-text flow (first run's single open question failed two ways:
# the model ignored "responde solamente: NINGUNO" and answered verbosely, which
# the old `\bninguno\b` check missed and spliced raw into the caption; and on
# grn_025 it hallucinated a transcription outright). Now: closed gate ->
# transcribe -> reject non-transcription answers -> closed confirm. Text is
# spliced ONLY if it survives all four steps.
LEGIBLE_GATE_QUESTION = (
    "¿Hay algún texto legible en la imagen (carteles, letreros, pancartas, "
    "fechas, nombres de lugares)? Responde únicamente con una palabra: SI o NO."
)

TRANSCRIBE_QUESTION = (
    "Transcribe el texto legible de la imagen palabra por palabra, EXACTAMENTE "
    "como aparece. Escribe únicamente el texto transcrito, sin describir la "
    "imagen ni añadir nada más."
)

# A real transcription is short and doesn't narrate. Answers that describe the
# image instead of quoting it are rejected rather than spliced.
DESCRIPTIVE_MARKERS = re.compile(
    r"la imagen|se (?:puede|pueden|observa|observan|muestra)|no hay|"
    r"ning[uú]n|texto legible", re.I)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(c))


def parse_closed_answer(answer: str) -> str:
    """Accent-insensitive prefix parse -> SI / NO / INCIERTO / UNPARSEABLE.
    (First run's `verdict.startswith("SI")` would miss 'Sí, ...' -- Í != I.)"""
    head = strip_accents(answer.strip().upper())
    for verdict in ("SI", "NO", "INCIERTO"):
        if head.startswith(verdict):
            return verdict
    return "UNPARSEABLE"


def clean_transcription(raw: str) -> str | None:
    """Accept only answers that look like an actual transcription."""
    quoted = re.findall(r'["“]([^"”]+)["”]', raw)
    text = " / ".join(q.strip() for q in quoted) if quoted else raw.strip().strip('"').strip()
    if not text or len(text) > 120 or DESCRIPTIVE_MARKERS.search(text):
        return None
    return text

PATTERN_QUESTION = (
    "¿Qué patrones, símbolos o diseños geométricos se observan en los objetos, "
    "vestimenta o superficies de la imagen, si los hay? Si no se observa ningún "
    "patrón o diseño distintivo, responde explícitamente que no hay ninguno."
)

# Retrieval score floor: below this, don't even attempt verification -- there's
# no candidate concept worth asking the model to check.
MIN_VERIFY_SCORE = 0.40


def extract_diagnostic_claim(snippet_text: str) -> str | None:
    """Pull the clause most likely to name a checkable visual feature.
    Heuristic for the prototype only: first clause containing a shape/pattern
    word. A real implementation would do this more carefully."""
    m = re.search(r"[^.]*\b(radial|geom[ée]tric|espiral|zoomorfo|telara[ñn]a|"
                  r"patr[oó]n|dise[ñn]o)[^.]*\.", snippet_text, re.I)
    return m.group(0).strip() if m else None


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

        # Existing 4 categories, unchanged.
        annotations: dict[str, str] = {}
        for category in vqa_prompts.CULTURAL_QUESTIONS:
            annotations[category] = backend.caption(ex.image_path, vqa_prompts.joint_question(category))

        # NEW (hardened): legible text via closed gate -> transcribe -> reject
        # descriptive answers -> closed confirm. Locked only if all pass.
        locked_text: str | None = None
        gate = backend.caption(ex.image_path, LEGIBLE_GATE_QUESTION)
        note = f"gate={parse_closed_answer(gate)}"
        if parse_closed_answer(gate) == "SI":
            raw = backend.caption(ex.image_path, TRANSCRIBE_QUESTION)
            text = clean_transcription(raw)
            if text is None:
                note += f"; rejected non-transcription answer: {raw[:80]!r}"
            else:
                confirm = backend.caption(
                    ex.image_path,
                    f'¿La imagen contiene exactamente el texto "{text}"? '
                    f"Responde únicamente con una palabra: SI o NO.")
                if parse_closed_answer(confirm) == "SI":
                    locked_text = text
                else:
                    note += f"; failed closed confirm ({confirm.strip()[:60]!r})"
        print(f"\nSTEP 1 -- legible text ({note}):\n  "
              f"{'LOCKED: ' + locked_text if locked_text else '(none locked)'}")

        # Patterns question (v2), retrieval on it (v2).
        pattern_answer = backend.caption(ex.image_path, PATTERN_QUESTION)
        hits = text_banks[culture].retrieve(pattern_answer, k=3) if pattern_answer else []
        print(f"\nSTEP 2 -- pattern answer: {pattern_answer}")
        print("STEP 3 -- retrieval:")
        for h in hits:
            print(f"  {h['score']:.2f} -- {h['title']}: {h['extract'][:150]}")

        # NEW: closed verification question on the top hit's diagnostic claim,
        # only if score clears the floor and a checkable claim can be extracted.
        top = hits[0] if hits else None
        verify_answer, concept_name = None, None
        if top and top["score"] >= MIN_VERIFY_SCORE:
            claim = extract_diagnostic_claim(top["extract"]) or top["extract"][:150]
            concept_name = top["title"]
            verify_prompt = (
                f"Wikipedia describe {concept_name} así: \"{claim}\" "
                f"¿Se observa específicamente esto en la imagen? Responde "
                f"únicamente con una palabra: SI, NO, o INCIERTO, seguida de "
                f"una breve razón."
            )
            verify_answer = backend.caption(ex.image_path, verify_prompt)
            print(f"\nSTEP 4 -- closed verification (score {top['score']:.2f} >= "
                  f"floor {MIN_VERIFY_SCORE}):\n  Q: {verify_prompt}\n  A: {verify_answer}")
        else:
            print(f"\nSTEP 4 -- skipped: top score "
                  f"{top['score'] if top else 0:.2f} below floor "
                  f"{MIN_VERIFY_SCORE} -- nothing worth verifying.")

        # Assemble in Python -- no open-ended synthesis call for the parts we
        # can control. Only the base description comes from a model call.
        base = backend.caption(ex.image_path, vqa_prompts.GENERIC_PROMPT)
        parts = [base.rstrip(".")]
        if verify_answer:
            verdict = parse_closed_answer(verify_answer)
            if verdict == "SI":
                parts.append(f"posiblemente {concept_name}")
            elif verdict == "INCIERTO":
                parts.append(f"posiblemente relacionado con {concept_name}")
            # NO / UNPARSEABLE -> omit entirely, don't mention the concept
        if locked_text:
            parts.append(f'texto visible: "{locked_text}"')
        final = ". ".join(parts) + "."
        print(f"\nSTEP 5 -- assembled final caption (Python-spliced, not model-synthesized):\n  {final}")

    print(f"\n{'=' * 70}\nDone. Qualitative comparison only -- no new eval numbers.")


if __name__ == "__main__":
    main()
