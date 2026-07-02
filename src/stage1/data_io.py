"""Loading CICIL JSONL records and resolving their image paths.

The shared-task `filename` field is a *logical* path that does not match the
on-disk layout (dev images live in ``data/dev/<lang>/images/`` while pilot
images live in ``data/pilot/images/<lang>/``). We therefore resolve each image
by its basename within the JSONL file's sibling ``images/`` directory, which is
robust to both layouts.
"""

from __future__ import annotations

import argparse
import functools
import json
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Example:
    id: str
    image_path: Path
    language: str
    iso_lang: str
    culture: str
    split: str
    target_caption: str | None = None    # gold indigenous-language caption (dev/pilot)
    spanish_caption: str | None = None   # reference Spanish caption (PILOT ONLY)


def _jsonl_path(lang: str, split: str) -> Path:
    if split == "pilot":
        return config.PILOT_DIR / f"{lang}.jsonl"
    base = config.DEV_DIR if split == "dev" else config.TEST_DIR
    return base / lang / f"{lang}.jsonl"


@functools.lru_cache(maxsize=None)
def _image_index(images_dir: Path) -> dict[str, Path]:
    """Map image basename -> absolute path, recursively under an images dir."""
    index: dict[str, Path] = {}
    if images_dir.is_dir():
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                index[p.name] = p
    return index


def _resolve_image(record_filename: str, jsonl_path: Path) -> Path:
    images_dir = jsonl_path.parent / "images"
    index = _image_index(images_dir)
    basename = Path(record_filename).name
    if basename in index:
        return index[basename]
    # Fall back to a naive join so the caller gets a path to report if missing.
    return images_dir / basename


def load_split(lang: str, split: str = "dev") -> list[Example]:
    """Load all examples for a language/split with resolved image paths."""
    path = _jsonl_path(lang, split)
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL not found: {path}\n"
            "Did the dataset clone succeed? Expected under data/americasnlp2026/."
        )
    examples: list[Example] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            examples.append(
                Example(
                    id=r["id"],
                    image_path=_resolve_image(r["filename"], path),
                    language=r.get("language", lang),
                    iso_lang=r.get("iso_lang", ""),
                    culture=r.get("culture", lang),
                    split=r.get("split", split),
                    target_caption=r.get("target_caption"),
                    spanish_caption=r.get("spanish_caption"),
                )
            )
    return examples


def _main() -> None:
    ap = argparse.ArgumentParser(description="Inspect CICIL split records.")
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--split", default="dev", choices=["pilot", "dev", "test"])
    ap.add_argument("--head", type=int, default=3)
    args = ap.parse_args()

    examples = load_split(args.lang, args.split)
    print(f"{args.lang}/{args.split}: {len(examples)} records")
    for ex in examples[: args.head]:
        exists = "OK" if ex.image_path.exists() else "MISSING"
        print(f"\n[{ex.id}] image={ex.image_path}  ({exists})")
        if ex.spanish_caption:
            print(f"  spanish: {ex.spanish_caption[:80]}")
        if ex.target_caption:
            print(f"  target:  {ex.target_caption[:80]}")


if __name__ == "__main__":
    _main()
