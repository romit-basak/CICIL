"""Stage 2 — Step 2: translate Spanish → indigenous language via Gemini 2.5 Flash.

Reads the Stage 1 JSONL (``outputs/<lang>_dev_<mode>_<backend>.jsonl``), retrieves k
similar parallel pairs from the FAISS index (when one exists for the language),
and calls Gemini 2.5 Flash to translate. Writes a plain ``.txt`` (one caption per
line, input order) for scoring by ``src.stage1.evaluate``.

!! BACKEND / LICENSE NOTE !!
    Uses Gemini 2.5 Flash on **Vertex AI** (not the AI Studio / Gemini Developer
    API). Rationale: the free AI Studio tier trains on requests, violating the
    CC BY-NC 4.0 dataset license, and the Education grant CANNOT unlock the AI
    Studio paid tier — but it DOES cover Vertex AI, which also doesn't train on
    your data. Auth is via Application Default Credentials (no API key):

        gcloud auth application-default login
        gcloud auth application-default set-quota-project cicil-501318
        gcloud services enable aiplatform.googleapis.com --project cicil-501318

    Project/location default to cicil-501318 / us-central1 (override via
    GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION).

Usage:
    uv run python -m src.stage2.translate --lang all --mode generic --k 5
    uv run python -m src.stage2.translate --lang all --mode cultural-vqa --k 5

    # Score a non-ollama Stage 1 backend (e.g. the distilled SmolVLM adapter):
    uv run python -m src.stage2.translate --lang wixarika --mode cultural-vqa --k 5 --backend smolvlm

    # RQ1 headline ablation -- does culturally-indexed retrieval beat vanilla
    # text retrieval? Same mode, same k, only the retrieval *query* changes:
    uv run python -m src.stage2.translate --lang all --mode cultural-vqa --k 5 --query-arm cultural
    uv run python -m src.stage2.translate --lang all --mode cultural-vqa --k 5 --query-arm text

    # Both axes together:
    uv run python -m src.stage2.translate --lang wixarika --mode cultural-vqa --k 5 \\
        --backend smolvlm --query-arm cultural

    # Validate the pipeline WITHOUT auth or any API call:
    uv run python -m src.stage2.translate --lang all --mode generic --k 5 --dry-run

For the full k x query-arm sweep in one command, see run_sweep.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from src.stage1.postprocess import strip_meta_prefix
from .paths import INPUT_DIR, PRED_DIR
from .retrieval import Retriever, build_query_from_record

# =============================================================
# CONFIG
# =============================================================

LANGUAGES = ["guarani", "bribri", "maya", "wixarika", "nahuatl"]

# Human-readable names used inside the translation prompt.
LANGUAGE_NAMES = {
    "guarani": "Guarani",
    "bribri": "Bribri",
    "maya": "Yucatec Maya",
    "wixarika": "Wixarika",
    "nahuatl": "Nahuatl",
}

# Languages that have a FAISS index (build_index.py). Today only the Wixárika
# pilot has genuine Spanish↔target pairs; the rest fall back to zero-shot.
INDEXED_LANGS = {"wixarika"}

# Which Stage 1 backend produced the input JSONL. Shared with run_sweep.py so
# the choices list has one source of truth.
BACKEND_CHOICES = ["ollama", "smolvlm", "hf", "vllm", "smolvlm-devonly",
                   "smolvlm-noctx", "smolvlm-rag", "ollama-rag", "vllm-rag",
                   "smolvlm-ragdistill"]

GEMINI_MODEL = "gemini-2.5-flash"
# Decoding: see call_gemini's docstring for the ablation behind these values.
GEMINI_TEMPERATURE = 0.7
GEMINI_SEED = 20260725

# Vertex AI backend (the Education grant covers Vertex, but NOT the AI Studio /
# Gemini Developer API paid tier). Auth is via Application Default Credentials
# (gcloud auth application-default login) — no API key. Project/location are
# overridable via env for portability.
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "cicil-501318")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Pause between API calls (seconds). Vertex paid quotas are generous; keep a
# small pause to stay well under per-minute limits.
API_PAUSE = 0.5

# Stage 1 cultural-VQA dev files are the concise **v2** regen — safe to score.
CULTURAL_VQA_V2_READY = True


# =============================================================
# Argument parsing
# =============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2: translate Stage 1 outputs to indigenous languages"
    )
    p.add_argument(
        "--lang", required=True,
        help="Language code: guarani / bribri / maya / wixarika / nahuatl / all",
    )
    p.add_argument(
        "--mode", required=True, choices=["generic", "cultural-vqa"],
        help="generic = baseline arm; cultural-vqa = treatment arm",
    )
    p.add_argument(
        "--k", type=int, default=5,
        help="Number of retrieved few-shot examples (ablation: 3, 5, 8). Default: 5",
    )
    p.add_argument(
        "--backend", default="ollama", choices=BACKEND_CHOICES,
        help="Which Stage 1 backend produced the input JSONL. Non-ollama runs "
             "read <lang>_dev_<mode>_<backend>.jsonl and write predictions "
             "tagged with the backend (ollama keeps the original filenames).",
    )
    p.add_argument(
        "--query-arm", choices=["cultural", "text", "auto"], default="auto",
        help=(
            "Retrieval query strategy (RQ1 headline ablation, independent of --mode "
            "and --backend): 'cultural' = query on cultural-annotation values; "
            "'text' = query on plain Spanish text regardless of annotations; "
            "'auto' = legacy behavior (cultural if present, else text -- couples "
            "the query strategy to --mode, so it doesn't isolate the retrieval "
            "effect). Default: auto."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate load/retrieve/prompt without calling Gemini (no key, no quota). "
             "Writes sample prompts to predictions/_dryrun/ and skips *_predictions.txt.",
    )
    return p.parse_args()


# =============================================================
# Prompt
# =============================================================

SYSTEM_PROMPT = (
    "You are an expert translator specializing in indigenous languages of the Americas.\n"
    "Translate the given Spanish image caption into the specified indigenous language.\n"
    "The caption describes a culturally situated image. Pay close attention to cultural\n"
    "vocabulary, traditional practices, and community-specific terms.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Output ONLY the translation. Nothing else.\n"
    "- No preamble like 'Here is the translation:'\n"
    "- No explanation, alternatives, or commentary.\n"
    "- Do not transliterate or leave Spanish words untranslated.\n"
    "- One sentence only.\n"
)


def build_prompt(spanish: str, target_lang: str, examples: list) -> str:
    """Few-shot translation prompt; most-similar example placed last (nearest the source)."""
    lines = [f"Translate the following Spanish caption into {target_lang}.\n"]
    if examples:
        lines.append("Reference translations:\n")
        for ex in examples:
            lines.append(f"Spanish:      {ex['spanish']}")
            lines.append(f"{target_lang}: {ex['target']}\n")
    lines.append("Now translate this:")
    lines.append(f"Spanish:      {spanish}")
    lines.append(f"{target_lang}:")
    return "\n".join(lines)


# =============================================================
# Gemini via Vertex AI (grant-covered; ADC auth, no API key)
# =============================================================

_client = None


def _get_client():
    """Lazily build a Vertex-backed google-genai client (uses ADC)."""
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
        )
    return _client


def call_gemini(prompt: str) -> str:
    """Call Gemini 2.5 Flash on Vertex AI. Sampled at a fixed seed.

    Was temperature 0 through the prelim runs, but greedy decoding degenerates
    into repetition loops on the lowest-resource languages (86% of Wixárika /
    64% of Bribri dev captions). A/B ablation (2026-07-25): frequency_penalty
    left both rate and ChrF++ unchanged, while temperature 0.7 cut degeneration
    to ~30% and raised ChrF++ (+4.96 Wixárika, +1.72 Bribri) with no regression
    on a healthy-language check (Guaraní +0.64). Seed pinned for
    reproducibility; prelim temp-0 numbers remain the documented prelim record.
    """
    from google.genai import types

    response = _get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=GEMINI_TEMPERATURE,
            seed=GEMINI_SEED,
            # Disable 2.5-Flash "thinking": translation needs no reasoning trace,
            # and the default dynamic budget adds ~30 s/call (and thinking-token cost).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            # A caption is one short sentence. Cap output so any residual
            # repetition loop on very low-resource languages can't run to a
            # huge length — bounds latency/cost; truncated degenerate text
            # still scores as poor (the honest RQ2 result is unchanged).
            max_output_tokens=128,
        ),
    )
    # First non-empty line only — guards against stray commentary that would
    # break the line-count alignment the scorer depends on.
    for line in (response.text or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def call_llm_with_retry(prompt: str) -> str:
    """One automatic retry on transient API errors."""
    for attempt in range(2):
        try:
            return call_gemini(prompt)
        except Exception as e:  # noqa: BLE001 - one bad call must not abort the run
            if attempt == 0:
                print(f"    WARNING: API error ({e}) -- retrying in 5 s...")
                time.sleep(5)
            else:
                print(f"    ERROR: failed after retry: {e}")
                return ""
    return ""


# =============================================================
# Filename convention (shared with run_ablations.py / the notebook)
# =============================================================

def pred_filename(lang: str, mode: str, k: int, *, backend: str = "ollama",
                   query_arm: str = "auto") -> str:
    """Canonical predictions filename for a (lang, mode, k, backend, query_arm) run.

    backend="ollama" and query_arm="auto" (the defaults -- i.e. neither new axis
    explicitly invoked) MUST keep producing today's bare filename with zero extra
    segments: analysis/rq3_category.py and analysis/human_eval/build_sample.py both
    construct this exact string via f-string with no glob/fallback and no
    try/except around the read.
    """
    tag = ""
    if backend != "ollama":
        tag += f"_{backend}"
    if query_arm != "auto":
        tag += f"_{query_arm}query"
    return f"{lang}_{mode}{tag}_k{k}_predictions.txt"


# =============================================================
# Translation loop
# =============================================================

def _load_retriever(lang: str, k: int):
    if lang not in INDEXED_LANGS:
        return None
    try:
        return Retriever(lang=lang, k=k)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}\n  WARNING: falling back to zero-shot (no retrieval).")
        return None


def translate_language(lang: str, mode: str, k: int, *, dry_run: bool,
                       backend: str = "ollama", query_arm: str = "auto"):
    lang_name = LANGUAGE_NAMES.get(lang, lang)
    input_file = INPUT_DIR / f"{lang}_dev_{mode}_{backend}.jsonl"
    if not input_file.exists():
        print(f"  WARNING: {input_file} not found -- skipping {lang}.")
        return

    records = []
    with input_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from {input_file.name}")

    retriever = _load_retriever(lang, k)

    predictions = []
    n_empty = 0
    n_retrieved = 0
    n_cultural_fallback = 0
    sample_prompts = []
    for i, record in enumerate(records):
        record_id = record.get("id", f"#{i+1}")
        # Non-destructive filler cleanup: strip a leading "la imagen muestra …"
        # so it never propagates into the translation (Stage 1 JSONL untouched).
        spanish = strip_meta_prefix(record.get("generated_spanish", "").strip())

        if not spanish:
            n_empty += 1
            predictions.append("")
            if not dry_run:
                print(f"  [{i+1}/{len(records)}] {record_id} -- empty Spanish, empty line")
            continue

        examples = []
        if retriever is not None:
            if query_arm == "cultural" and not record.get("cultural_annotations"):
                n_cultural_fallback += 1
            query = build_query_from_record(record, query_arm=query_arm)
            examples = list(reversed(retriever.retrieve(query)))
            if examples:
                n_retrieved += 1

        prompt = build_prompt(spanish, lang_name, examples)

        if dry_run:
            if len(sample_prompts) < 2:  # keep the first couple per language
                sample_prompts.append(f"### {record_id}\n{prompt}")
            predictions.append(None)  # placeholder; not written in dry-run
            continue

        predictions.append(call_llm_with_retry(prompt))
        print(f"  [{i+1}/{len(records)}] {record_id} -> {predictions[-1][:75]}...")
        time.sleep(API_PAUSE)

    if dry_run:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        dry_dir = PRED_DIR / "_dryrun"
        dry_dir.mkdir(parents=True, exist_ok=True)
        sample_name = pred_filename(lang, mode, k, backend=backend, query_arm=query_arm)
        sample_file = dry_dir / sample_name.replace("_predictions.txt", "_prompt_sample.txt")
        sample_file.write_text("\n\n".join(sample_prompts), encoding="utf-8")
        retr = "retrieval" if retriever is not None else "zero-shot"
        print(
            f"  DRY-RUN {lang}/{mode}/backend={backend}/query-arm={query_arm}: "
            f"{len(records)} records, {retr}, "
            f"{n_retrieved} with examples, {n_empty} empty-source, "
            f"{len(records) - n_empty} prompts built. Sample -> {sample_file}"
        )
        if n_cultural_fallback:
            print(
                f"  WARNING: query-arm=cultural but {n_cultural_fallback}/{len(records)} records "
                f"had no cultural_annotations -- those silently used the Spanish-text query instead. "
                f"This arm is NOT a clean cultural-query test unless this is 0."
            )
        return

    out_file = PRED_DIR / pred_filename(lang, mode, k, backend=backend, query_arm=query_arm)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write((pred or "") + "\n")
    print(f"\n  Written {len(predictions)} lines -> {out_file}")
    if n_cultural_fallback:
        print(
            f"  WARNING: query-arm=cultural but {n_cultural_fallback}/{len(records)} records "
            f"had no cultural_annotations -- those silently used the Spanish-text query instead. "
            f"This arm is NOT a clean cultural-query test unless this is 0."
        )
    print(f"  Score with:\n    uv run python -m src.stage1.evaluate --lang {lang} --translations {out_file}")


# =============================================================
# Entry point
# =============================================================

def ensure_vertex_credentials() -> None:
    """Exit with a clear message if Vertex ADC creds are missing. No-op check
    only -- callers still skip this entirely in --dry-run mode. Shared by
    translate.py's own CLI and by run_sweep.py so the whole grid fails fast
    once, up front, instead of on run 1 of 45.
    """
    adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if not os.path.exists(adc) and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print()
        print("ERROR: no Application Default Credentials found for Vertex AI.")
        print("Run once:  gcloud auth application-default login")
        print(f"           gcloud auth application-default set-quota-project {VERTEX_PROJECT}")
        print("Or validate the pipeline without auth: add --dry-run")
        print()
        raise SystemExit(1)
    print(f"Vertex AI: project={VERTEX_PROJECT} location={VERTEX_LOCATION} model={GEMINI_MODEL}")


def main():
    args = parse_args()

    if args.mode == "cultural-vqa" and not CULTURAL_VQA_V2_READY:
        print("ABORT: cultural-vqa v2 files not marked ready (set CULTURAL_VQA_V2_READY=True).")
        return

    if args.query_arm == "auto":
        print(
            "WARNING: --query-arm defaulted to 'auto'. Output will be the plain "
            "'{lang}_{mode}[_<backend>]_k{k}_predictions.txt' -- the SAME filename "
            "Stage 2 has always used, with no new tag. run_ablations.py's query-arm "
            "tables only look for '..._culturalquery_...' / '..._textquery_...' files, "
            "so this run will NOT show up there. Pass --query-arm cultural or "
            "--query-arm text explicitly for anything you want scored by the "
            "retrieval-arm ablation."
        )

    if not args.dry_run:
        ensure_vertex_credentials()

    langs = LANGUAGES if args.lang == "all" else [args.lang]
    for lang in langs:
        tag = " [DRY-RUN]" if args.dry_run else ""
        print(f"\n== {lang.upper()} | mode={args.mode} | backend={args.backend} "
              f"| query-arm={args.query_arm} | k={args.k}{tag} ==")
        translate_language(
            lang, args.mode, args.k,
            dry_run=args.dry_run,
            backend=args.backend,
            query_arm=args.query_arm,
        )

    print("\n== All done. ==")
    if not args.dry_run:
        print("Run: uv run python -m src.stage2.run_ablations")


if __name__ == "__main__":
    main()
