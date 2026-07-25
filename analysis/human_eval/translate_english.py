"""One-shot ES→EN translation of the human-eval sample, via Vertex Gemini.

The team judges Stage 1's Spanish captions through English translations (see
RUBRIC.md, "English-assisted annotation") because no team member reads Spanish
well enough to annotate directly. ES→EN is a high-resource direction where LLM
translation is state-of-the-art, so the pivot adds far less noise than the 0-2
rubric dimensions it supports.

Reads  analysis/human_eval/sample_spanish.csv  (caption_A / caption_B per row)
Writes analysis/human_eval/sample_english.csv  (sample_id, english_A, english_B)

Run once (60 Gemini Flash calls ≈ pennies, Vertex ADC auth required):
    uv run python -m analysis.human_eval.translate_english
    uv run python -m analysis.human_eval.translate_english --limit 2   # smoke test

Idempotent: re-running overwrites sample_english.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_FILE = HERE / "sample_spanish.csv"
OUT_FILE = HERE / "sample_english.csv"

# Same Vertex configuration as src/stage2/translate.py (kept standalone on
# purpose -- that module's call path bakes in the ES->indigenous translation
# system prompt; only the client pattern is shared).
GEMINI_MODEL = "gemini-2.5-flash"
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "cicil-501318")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
API_PAUSE = 0.5  # seconds between calls

SYSTEM_PROMPT = (
    "You are a careful Spanish-to-English translator. Translate the given text "
    "faithfully and completely. Output ONLY the English translation, nothing "
    "else. Keep culturally specific terms (e.g. mate, milpa, chamamé) in their "
    "original form followed by a short gloss in brackets the first time they "
    "appear, rather than replacing them with a generic English word."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
        )
    return _client


def translate(text: str) -> str:
    """ES→EN with one retry, mirroring translate.py's call_llm_with_retry."""
    from google.genai import types

    for attempt in (1, 2):
        try:
            response = _get_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=256,
                ),
            )
            out = (response.text or "").strip()
            if out:
                return out
            raise RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001
            if attempt == 1:
                print(f"    WARNING: API error ({e}) -- retrying in 5 s...")
                time.sleep(5)
            else:
                print(f"    ERROR: giving up on this caption ({e})")
                return ""
    return ""


def ensure_vertex_credentials() -> None:
    adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if not os.path.exists(adc) and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise SystemExit(
            "ERROR: no Application Default Credentials found for Vertex AI.\n"
            "Run once:  gcloud auth application-default login"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate the human-eval sample ES→EN.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only translate the first N rows (smoke test).")
    args = ap.parse_args()

    ensure_vertex_credentials()
    print(f"Vertex AI: project={VERTEX_PROJECT} location={VERTEX_LOCATION} "
          f"model={GEMINI_MODEL}")

    with IN_FILE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: args.limit]

    out_rows = []
    for i, row in enumerate(rows, 1):
        sid = row["sample_id"]
        print(f"[{i}/{len(rows)}] {sid} ({row['language']})")
        english_a = translate(row["caption_A"])
        time.sleep(API_PAUSE)
        english_b = translate(row["caption_B"])
        time.sleep(API_PAUSE)
        out_rows.append({"sample_id": sid, "english_A": english_a,
                         "english_B": english_b})

    with OUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "english_A", "english_B"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows -> {OUT_FILE}")


if __name__ == "__main__":
    main()
