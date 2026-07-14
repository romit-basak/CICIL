"""Stage 2 — culturally-indexed retrieval + Gemini translation.

Consumes the Stage 1 Spanish descriptions (``outputs/<lang>_dev_<mode>_ollama.jsonl``),
retrieves few-shot Spanish↔indigenous pairs from a per-language FAISS index, and
translates to the target language with Gemini 2.5 Flash. Predictions are scored
with ``src.stage1.evaluate`` (the official ChrF++ scorer).

Pipeline order: ``build_index`` (once) → ``translate`` (per lang/mode/k) →
``run_ablations`` (scoring tables). ``retrieval`` is a library used by ``translate``.
"""
