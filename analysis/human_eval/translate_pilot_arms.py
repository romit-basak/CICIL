"""One-off Stage 2 translation for the 20-image pilot split (round 3 human eval).

src/stage2/translate.py's translate_language() is hardcoded to the dev split's
filenames, and Mehek's sweep is actively running that module -- editing it here
risks interfering with a live process. This script imports only the safe,
side-effect-free pieces (retrieval, prompt building, the Gemini call) and
targets the pilot split explicitly, writing to its own filenames so nothing
collides with any dev-based output.

Run:
    uv run python -m analysis.human_eval.translate_pilot_arms
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.stage1.postprocess import strip_meta_prefix
from src.stage2.paths import PRED_DIR
from src.stage2.retrieval import Retriever, build_query_from_record
from src.stage2.translate import (
    LANGUAGE_NAMES, build_prompt, call_llm_with_retry, ensure_vertex_credentials,
)

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs"
LANG = "wixarika"
K = 5
ARMS = ["smolvlm-rag", "smolvlm-ragdistill"]
API_PAUSE = 1.0


def translate_arm(backend: str) -> None:
    in_file = OUTPUTS / f"{LANG}_pilot_cultural-vqa_{backend}.jsonl"
    records = [json.loads(l) for l in in_file.open(encoding="utf-8") if l.strip()]
    print(f"[{backend}] {len(records)} records from {in_file.name}")

    retriever = Retriever(LANG, K)  # wixarika is INDEXED_LANGS -- real bank
    lang_name = LANGUAGE_NAMES.get(LANG, LANG)

    predictions = []
    for i, record in enumerate(records):
        spanish = strip_meta_prefix(record.get("generated_spanish", "").strip())
        if not spanish:
            predictions.append("")
            continue
        query = build_query_from_record(record, query_arm="auto")
        examples = list(reversed(retriever.retrieve(query))) if retriever else []
        prompt = build_prompt(spanish, lang_name, examples)
        pred = call_llm_with_retry(prompt)
        predictions.append(pred)
        print(f"  [{i + 1}/{len(records)}] {record['id']} -> {pred[:70]}...")
        time.sleep(API_PAUSE)

    out_file = PRED_DIR / f"{LANG}_pilot_cultural-vqa_{backend}_k{K}_predictions.txt"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(p + "\n")
    print(f"  -> {out_file}")


def main() -> None:
    ensure_vertex_credentials()
    for backend in ARMS:
        translate_arm(backend)


if __name__ == "__main__":
    main()
