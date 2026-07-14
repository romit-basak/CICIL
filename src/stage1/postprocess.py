"""Light normalization of Stage 1 Spanish descriptions.

The cultural-VQA v2 synthesis prompt bans meta/filler openings ("la imagen
muestra …", hedging, etc.), but Qwen2.5-VL still occasionally emits them. Those
openers add no cultural content and only dilute the Spanish that Stage 2 hands
to Gemini, so we strip a leading meta-phrase before translation.

This is applied *non-destructively* at read time in Stage 2: the Stage 1 JSONLs
remain the raw record of what the VLM produced (reproducibility); only the text
fed downstream is cleaned. ``strip_meta_prefix`` is idempotent and leaves text
that doesn't start with a banned phrase untouched.
"""

from __future__ import annotations

import re

# Leading meta/filler openers to remove (case-insensitive, anchored at start).
# Each pattern consumes the phrase plus any trailing "que"/connective and the
# separator ("," / ":" / whitespace) up to the start of the real description.
_META_PREFIX = re.compile(
    r"""^\s*
    (?:
        la\s+imagen\s+(?:muestra|presenta|representa|retrata)
      | la\s+foto(?:graf[íi]a)?\s+muestra
      | en\s+(?:la\s+imagen|la\s+foto(?:graf[íi]a)?)\s+se\s+(?:ve|ven|observa|observan|muestra|aprecia|aprecian)
      | en\s+(?:la\s+imagen|la\s+foto(?:graf[íi]a)?)
      | se\s+(?:ve|ven|observa|observan|muestra|muestran|aprecia|aprecian|puede\s+ver|pueden\s+ver)
      | aqu[íi]\s+se\s+(?:ve|ven|observa|observan|muestra)
    )
    \s*(?:que\s+)?
    [\s,:;-]*
    """,
    re.IGNORECASE | re.VERBOSE,
)


def strip_meta_prefix(text: str) -> str:
    """Remove a single leading meta/filler opener and re-capitalize.

    Idempotent: only one leading phrase is stripped (real descriptions don't
    stack them), and text without such an opener is returned unchanged aside
    from surrounding whitespace.
    """
    if not text:
        return text
    cleaned = _META_PREFIX.sub("", text, count=1).strip()
    if not cleaned:
        # The whole string was filler; keep the original rather than emptying it.
        return text.strip()
    # Re-capitalize the new first letter (the stripped remainder usually starts
    # lowercase, e.g. "la imagen muestra un patio" -> "un patio" -> "Un patio").
    return cleaned[0].upper() + cleaned[1:]
