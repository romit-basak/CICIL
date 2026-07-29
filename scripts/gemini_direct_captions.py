"""Gemini-direct ceiling arm: one frontier-VLM call per image, image included.

The capability ladder's top rung: what does captioning cost/gain when the
image itself goes to the API? (The multi-agent arms send only text derived
from the image; this arm deliberately breaks that property to measure the
ceiling.) Comparable rules to the agent arms' assembler for fairness: length
cap, no invented names/places, culture given.

Run:
  uv run python scripts/gemini_direct_captions.py --lang wixarika --split pilot
  # -> outputs/{lang}_{split}_geminidirect.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CULTURE_NAMES_ES = {
    "guarani": "guaraní (Paraguay/Argentina)",
    "wixarika": "wixárika / huichol (México)",
    "maya": "maya yucateco (México)",
    "bribri": "bribri (Costa Rica)",
    "nahuatl": "náhuatl (México)",
}

PROMPT = (
    "Describe esta imagen de la cultura {culture} en español, en una o dos "
    "oraciones (máximo 40 palabras). Menciona los elementos culturales que "
    "reconozcas con seguridad, y la acción principal si hay personas. Reglas: "
    "no inventes; no identifiques a personas con nombre propio; no nombres "
    "lugares que no se vean o lean en la imagen; si el contenido cultural no "
    "es identificable, describe la escena sin especular. Responde SOLO con la "
    "descripción."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--lang", default="wixarika", choices=list(CULTURE_NAMES_ES))
    parser.add_argument("--split", default="pilot", choices=["pilot", "dev"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from google.genai import types

    from src.stage1.data_io import load_split
    from src.stage2.translate import (
        GEMINI_MODEL,
        GEMINI_SEED,
        _get_client,
        ensure_vertex_credentials,
    )

    ensure_vertex_credentials()
    out_path = ROOT / "outputs" / f"{args.lang}_{args.split}_geminidirect.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.open(encoding="utf-8") if l.strip()}

    examples = load_split(args.lang, args.split)
    if args.limit:
        examples = examples[: args.limit]
    prompt = PROMPT.format(culture=CULTURE_NAMES_ES[args.lang])

    with out_path.open("a", encoding="utf-8") as f:
        for i, ex in enumerate(ex for ex in examples if ex.id not in done):
            image_bytes = ex.image_path.read_bytes()
            for attempt, delay in enumerate((0, 15, 45, 120)):
                if delay:
                    time.sleep(delay)
                try:
                    resp = _get_client().models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                                  prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.2, seed=GEMINI_SEED,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                            max_output_tokens=200,
                        ),
                    )
                    caption = (resp.text or "").strip()
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    print(f"  WARNING: {str(e)[:70]}, retrying")
            f.write(json.dumps({"id": ex.id, "filename": ex.image_path.name,
                                "backend": "gemini-direct", "generated_spanish": caption},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{ex.id}] {caption[:110]}")
            time.sleep(1)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
