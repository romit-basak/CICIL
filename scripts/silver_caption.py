"""Silver-caption scraped Commons images with the Qwen2.5-VL teacher.

For every image scraped by ``scrape_commons.py`` the teacher produces the same
two outputs the CICIL dev images already have (from the prelim runs): a generic
Spanish caption and the full cultural-VQA record (per-category annotations +
synthesized description). These are the *silver* targets the SmolVLM student
distills from.

Two twists over the dev runs: (1) the Commons **description** (encyclopedic,
often naming the ceremony/objects/place) is fed to the *teacher* as prompt
context via ``vqa_prompts.with_context`` — enriching the silver targets — while
the recorded task for the student stays the plain deployment prompt; (2) the
teacher answers each category's questions **jointly** (one call per category),
so the silver target is the answer to *exactly* the prompt the student trains
on — and the run drops from 7 to 5 calls per image.

Two backends, same teacher model, same prompts/targets — pick whichever is
available:
  * ``--backend ollama`` (default) — local, quantized (q4), serializes on the
    Ollama daemon (``--workers`` beyond 1 does not help; it's one GPU/process).
  * ``--backend vllm`` — bf16, served via vLLM on a cloud GPU (e.g. an L4);
    the server batches concurrent requests itself, so ``--workers 8`` gives
    real throughput. Point ``--base-url`` at the server (default localhost, for
    running the client alongside ``vllm serve`` on the same VM).
Both write to backend-tagged files (``commons_<culture>_<mode>_<backend>.jsonl``)
so a vLLM run and a prior/concurrent Ollama run never collide; ``done_ids``
unions *all* backend files for the same culture/mode so work already finished
by one backend is never repeated by the other — this is how a Commons scrape
started locally can be finished on the cloud without wasted teacher calls.

Deterministic decoding (temperature 0 + fixed seed), resumable (appends; skips
ids already present in any backend's output for that culture/mode).

Run:
  uv run python scripts/silver_caption.py --limit 2                     # smoke
  uv run python scripts/silver_caption.py                               # full, local Ollama
  uv run python scripts/silver_caption.py --backend vllm --workers 8 \\
      --base-url http://localhost:8000/v1                               # cloud vLLM
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage1 import config  # noqa: E402
from src.stage1.backends import get_backend  # noqa: E402
from src.stage1.generate_descriptions import _run_cultural_vqa, _run_generic  # noqa: E402

CULTURES = ["guarani", "bribri", "maya", "wixarika", "nahuatl"]
CONTEXT_MAX_CHARS = 600  # keep teacher prompts bounded; descriptions can be long
# Decode cap: distill_data clips targets to 80 words, so tokens past ~160 are
# paid for and discarded. 160 (~110 Spanish words) leaves sentence boundaries
# for the clipper.
NUM_PREDICT = 160


def load_provenance(commons_root: Path) -> list[dict]:
    csv_path = commons_root / "provenance.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no provenance.csv — run scrape_commons.py first: {csv_path}")
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def done_ids(culture: str, mode: str, prefix: str = "commons") -> set[str]:
    """Union of ids already captioned by ANY backend, for this culture/mode.

    The prefix isolates ablation arms: the no-context regeneration writes and
    resumes under ``commons-noctx_*`` and never collides with the main
    ``commons_*`` files (nor vice versa — the glob is prefix-anchored).
    """
    ids: set[str] = set()
    for path in config.OUTPUT_DIR.glob(f"{prefix}_{culture}_{mode}_*.jsonl"):
        with path.open(encoding="utf-8") as f:
            ids |= {json.loads(line)["id"] for line in f if line.strip()}
    return ids


def caption_one(backend, backend_name: str, row: dict, commons_root: Path,
                culture: str, mode: str, no_context: bool = False) -> dict | None:
    image_path = commons_root / row["local_file"]
    if not image_path.exists():
        return None
    # no_context = the ablation arm: identical teacher/prompts/seed, but the
    # Commons description is withheld, isolating its contribution to the silver.
    context = None if no_context else (
        (row.get("description") or "")[:CONTEXT_MAX_CHARS] or None)
    try:
        if mode == "generic":
            description, annotations = _run_generic(backend, image_path, context=context)
        else:
            description, annotations = _run_cultural_vqa(
                backend, image_path, context=context, joint=True)
    except Exception as e:  # noqa: BLE001 - one bad image must not abort the run
        tqdm.write(f"[error] {image_path.name}: {e}")
        description, annotations = "", {}
    return {
        "id": image_path.stem,
        "filename": image_path.name,
        "language": culture,
        "iso_lang": "",
        "mode": mode,
        "backend": backend_name,
        "generated_spanish": description,
        "cultural_annotations": annotations,
        "context": context or "",
        "source": "wikimedia-commons",
        "license": row.get("license", ""),
    }


def run(commons_root: Path, cultures: list[str], modes: list[str],
        backend_name: str, base_url: str | None, model: str | None,
        seed: int, workers: int, limit: int | None,
        no_context: bool = False) -> None:
    provenance = load_provenance(commons_root)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    prefix = "commons-noctx" if no_context else "commons"

    def make_backend():
        kwargs = {"temperature": 0.0, "seed": seed}
        if backend_name == "vllm":
            if base_url:
                kwargs["base_url"] = base_url
        else:  # ollama
            kwargs["num_predict"] = NUM_PREDICT
        if model:
            kwargs["model"] = model
        return get_backend(backend_name, **kwargs)

    # One backend instance per worker thread: Ollama's client isn't documented
    # thread-safe, and vLLM's is stateless HTTP so a fresh instance is free.
    thread_local = threading.local()

    def backend_for_thread():
        if not hasattr(thread_local, "backend"):
            thread_local.backend = make_backend()
        return thread_local.backend

    for culture in cultures:
        rows = [r for r in provenance if r["culture"] == culture]
        if limit is not None:
            rows = rows[:limit]
        for mode in modes:
            out_path = config.OUTPUT_DIR / f"{prefix}_{culture}_{mode}_{backend_name}.jsonl"
            skip = done_ids(culture, mode, prefix)
            todo = [r for r in rows if Path(r["local_file"]).stem not in skip]
            if not todo:
                print(f"[{culture}/{mode}] nothing to do "
                      f"({len(skip)} already captioned by some backend)")
                continue
            print(f"[{culture}/{mode}] {len(todo)} to caption "
                  f"({len(skip)} already done) -> {out_path.name}")

            with out_path.open("a", encoding="utf-8") as f, \
                 ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(caption_one, backend_for_thread(), backend_name,
                                       row, commons_root, culture, mode, no_context)
                          for row in todo]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"{culture}/{mode}"):
                    record = fut.result()
                    if record is None:
                        continue
                    with write_lock:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
            print(f"[{culture}/{mode}] -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Teacher silver captions for Commons images.")
    ap.add_argument("--cultures", nargs="+", default=CULTURES, choices=CULTURES)
    ap.add_argument("--modes", nargs="+", default=["cultural-vqa", "generic"],
                    choices=["cultural-vqa", "generic"])
    ap.add_argument("--backend", default="ollama", choices=["ollama", "vllm"])
    ap.add_argument("--base-url", default=None, help="vLLM server URL (vllm backend only)")
    ap.add_argument("--model", default=None, help="Override model id/tag for the backend.")
    ap.add_argument("--commons-root", type=Path,
                    default=config.ROOT / "data" / "external" / "commons")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent captioning threads (vllm: use ~8; ollama: keep at 1).")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N images per culture (smoke test)")
    ap.add_argument("--no-context", action="store_true",
                    help="Ablation arm: withhold Commons descriptions from the "
                         "teacher. Writes/resumes under commons-noctx_* files.")
    args = ap.parse_args()
    run(args.commons_root, args.cultures, args.modes, args.backend,
        args.base_url, args.model, args.seed, args.workers, args.limit,
        no_context=args.no_context)


if __name__ == "__main__":
    main()
