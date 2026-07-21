"""Build the multi-task distillation set: (image, prompt, target) triples.

Sources (both produced by the Qwen2.5-VL teacher):
  * ``outputs/{lang}_dev_{cultural-vqa,generic}_ollama.jsonl`` — the 250 CICIL
    dev images (already generated for the prelim; dev training is allowed by
    the task rules, test never is).
  * ``outputs/commons_{culture}_{mode}_ollama.jsonl`` — scraped Commons images
    (see ``scripts/scrape_commons.py`` / ``silver_caption.py``).

Per image, up to six deployment-aligned pairs:
  * one per cultural category — prompt = that category's questions combined
    (``vqa_prompts.joint_question``), target = the teacher's annotation;
  * synthesis — prompt = ``format_synthesis(annotations)``, target = the
    teacher's synthesized description;
  * generic — prompt = ``GENERIC_PROMPT``, target = the generic caption.

The prompts are exactly what the student sees at deployment (the Commons
*context* the teacher saw is deliberately absent). Targets longer than
``--max-target-words`` are cut back to the last sentence boundary that fits,
or dropped — the student decodes 64-96 tokens, so unbounded targets would
teach truncation.

Contamination guard: any image colliding with the 20 gold pilot images
(sha1/dHash via ``dedup.pilot_index``) is excluded, whatever its source.

Run:  uv run python -m src.stage1.distill_data
      uv run python -m src.stage1.distill_data --sources dev
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from . import config, vqa_prompts
from .data_io import load_split
from .dedup import pilot_index

DEFAULT_OUT = config.OUTPUT_DIR / "distill_triples.jsonl"
COMMONS_ROOT = config.ROOT / "data" / "external" / "commons"
MAX_TARGET_WORDS = 80
SHUFFLE_SEED = 20260717


def clip_target(text: str, max_words: int) -> str | None:
    """Fit a target under the word cap, cutting at a sentence boundary.

    Returns the (possibly shortened) text, or None when nothing sentence-shaped
    fits — better to drop a pair than teach mid-sentence truncation.
    """
    text = " ".join((text or "").split())
    if not text:
        return None
    words = text.split(" ")
    if len(words) <= max_words:
        return text
    head = " ".join(words[:max_words])
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut > 0:
        return head[: cut + 1]
    return head + "." if head.endswith((".", "!", "?")) else None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _triples_for(record: dict, image_path: Path, source: str, culture: str,
                 max_words: int) -> list[dict]:
    """Deployment-aligned pairs from one teacher record (either mode)."""
    triples = []

    def add(task: str, prompt: str, target: str | None) -> None:
        target = clip_target(target or "", max_words)
        if target:
            triples.append({
                "id": record["id"],
                "image": str(image_path.relative_to(config.ROOT)),
                "culture": culture,
                "source": source,
                "task": task,
                "prompt": prompt,
                "target": target,
            })

    if record.get("mode") == "cultural-vqa":
        annotations = record.get("cultural_annotations") or {}
        for cat in vqa_prompts.CULTURAL_QUESTIONS:
            add(f"category:{cat}", vqa_prompts.joint_question(cat),
                annotations.get(cat, ""))
        if annotations:
            add("synthesis", vqa_prompts.format_synthesis(annotations),
                record.get("generated_spanish", ""))
    else:  # generic
        add("generic", vqa_prompts.GENERIC_PROMPT,
            record.get("generated_spanish", ""))
    return triples


def build(sources: list[str], out_path: Path, max_words: int,
          commons_prefix: str = "commons") -> None:
    guard = pilot_index()
    triples: list[dict] = []
    guarded = Counter()

    def guarded_extend(records: list[dict], resolve, source: str, culture: str):
        for rec in records:
            image_path = resolve(rec)
            if image_path is None or not image_path.exists():
                continue
            if guard.match(image_path):
                guarded[f"{source}/{culture}"] += 1
                continue
            triples.extend(_triples_for(rec, image_path, source, culture, max_words))

    if "dev" in sources:
        for lang in config.LANGUAGES:
            by_id = {ex.id: ex.image_path for ex in load_split(lang, "dev")}
            for mode in ("cultural-vqa", "generic"):
                records = _read_jsonl(
                    config.OUTPUT_DIR / f"{lang}_dev_{mode}_ollama.jsonl")
                guarded_extend(records, lambda r: by_id.get(r["id"]), "dev", lang)

    if "commons" in sources:
        for culture in config.LANGUAGES:
            img_dir = COMMONS_ROOT / culture / "images"
            for mode in ("cultural-vqa", "generic"):
                # Backend-tagged files (ollama/vllm/...); a Commons scrape can be
                # split across a local Ollama run and a cloud vLLM run with no
                # id overlap (silver_caption.py's done_ids guarantees this).
                # commons_prefix selects the ablation arm ("commons-noctx" =
                # silver generated without the Commons descriptions).
                records = [rec for path in config.OUTPUT_DIR.glob(
                              f"{commons_prefix}_{culture}_{mode}_*.jsonl")
                          for rec in _read_jsonl(path)]
                guarded_extend(records, lambda r: img_dir / r["filename"],
                               "commons", culture)

    random.Random(SHUFFLE_SEED).shuffle(triples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Report: pair counts by task and source, images excluded by the guard.
    by_task = Counter(t["task"] for t in triples)
    by_source = Counter(f"{t['source']}/{t['culture']}" for t in triples)
    n_images = len({t["image"] for t in triples})
    print(f"{len(triples)} triples over {n_images} images -> {out_path}")
    print("by task:  ", dict(sorted(by_task.items())))
    print("by source:", dict(sorted(by_source.items())))
    if guarded:
        print("excluded by pilot contamination guard:", dict(guarded))
    else:
        print("pilot contamination guard: nothing to exclude")


def load_triples(path: Path = DEFAULT_OUT) -> list[tuple[Path, str, str]]:
    """(image_path, prompt, target) tuples for the trainer."""
    return [(config.ROOT / t["image"], t["prompt"], t["target"])
            for t in _read_jsonl(path)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build distillation triples.")
    ap.add_argument("--sources", nargs="+", default=["dev", "commons"],
                    choices=["dev", "commons"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-target-words", type=int, default=MAX_TARGET_WORDS)
    ap.add_argument("--commons-prefix", default="commons",
                    choices=["commons", "commons-noctx"],
                    help="Which Commons silver arm to build from "
                         "(commons-noctx = the description-ablation arm).")
    args = ap.parse_args()
    build(args.sources, args.out, args.max_target_words,
          commons_prefix=args.commons_prefix)


if __name__ == "__main__":
    main()
