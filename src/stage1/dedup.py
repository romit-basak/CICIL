"""Image dedup: exact (sha1) + perceptual (dHash) matching against CICIL images.

Two jobs:
  * drop Wikimedia Commons scrapes that duplicate shared-task images (the CICIL
    dev/test photos were themselves sourced from the public web, so overlap is
    plausible), and
  * the contamination guard — nothing that collides with the 20 gold pilot
    images (our held-out test set) may enter the distillation training set.

dHash is a 64-bit difference hash computed with PIL only (no new dependency):
downscale to 9x8 grayscale and record whether each pixel is brighter than its
right neighbour. Near-duplicates (recompressed, resized, lightly cropped
copies) land within a small Hamming distance; 0-6 bits is the conventional
"same image" band, which we adopt.

CLI:
  python -m src.stage1.dedup --check-overlap          # pilot vs dev/test, per lang
  python -m src.stage1.dedup --scan-dir data/external/commons/wixarika/images
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import config
from .data_io import load_split

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MAX_HAMMING = 6


@dataclass(frozen=True)
class ImageRef:
    """A CICIL image with enough metadata to report a collision usefully."""

    path: Path
    lang: str
    split: str
    id: str


@dataclass(frozen=True)
class Collision:
    candidate: Path
    match: ImageRef
    kind: str  # "sha1" or "dhash(d=N)"


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int | None:
    """64-bit difference hash; None when the file isn't a readable image."""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
            px = im.tobytes()  # mode "L": one byte per pixel, row-major
    except Exception:
        return None
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# --- CICIL index ---------------------------------------------------------------

def cicil_images(splits: tuple[str, ...] = ("pilot", "dev", "test")) -> list[ImageRef]:
    """Every resolvable shared-task image across languages and splits."""
    refs: list[ImageRef] = []
    for split in splits:
        for lang in config.LANGUAGES:
            try:
                examples = load_split(lang, split)
            except FileNotFoundError:
                continue  # e.g. pilot exists for wixarika only
            for ex in examples:
                if ex.image_path.exists():
                    refs.append(ImageRef(ex.image_path, lang, split, ex.id))
    return refs


class DedupIndex:
    """sha1 + dHash index over a set of images, queryable by candidate path."""

    def __init__(self, refs: list[ImageRef], max_hamming: int = DEFAULT_MAX_HAMMING):
        self.max_hamming = max_hamming
        self._by_sha: dict[str, ImageRef] = {}
        self._hashes: list[tuple[int, ImageRef]] = []
        for ref in refs:
            self._by_sha.setdefault(sha1_file(ref.path), ref)
            h = dhash(ref.path)
            if h is not None:
                self._hashes.append((h, ref))

    def match(self, candidate: Path) -> Collision | None:
        """First collision for a candidate image, or None if it's novel."""
        sha_hit = self._by_sha.get(sha1_file(candidate))
        if sha_hit is not None:
            return Collision(candidate, sha_hit, "sha1")
        h = dhash(candidate)
        if h is None:
            return None
        best: tuple[int, ImageRef] | None = None
        for other, ref in self._hashes:
            d = hamming(h, other)
            if d <= self.max_hamming and (best is None or d < best[0]):
                best = (d, ref)
        if best is not None:
            return Collision(candidate, best[1], f"dhash(d={best[0]})")
        return None


def cicil_index(splits: tuple[str, ...] = ("pilot", "dev", "test"),
                max_hamming: int = DEFAULT_MAX_HAMMING) -> DedupIndex:
    return DedupIndex(cicil_images(splits), max_hamming)


def pilot_index(max_hamming: int = DEFAULT_MAX_HAMMING) -> DedupIndex:
    """Index of just the gold-eval pilot images — the contamination guard."""
    return DedupIndex(cicil_images(("pilot",)), max_hamming)


# --- CLI -----------------------------------------------------------------------

def _iter_images(directory: Path):
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def _scan_dir(directory: Path, max_hamming: int) -> int:
    index = cicil_index(max_hamming=max_hamming)
    n = hits = 0
    for p in _iter_images(directory):
        n += 1
        col = index.match(p)
        if col:
            hits += 1
            m = col.match
            print(f"COLLISION {col.kind:<12} {p.name}  ->  {m.split}/{m.lang}/{m.id}")
    print(f"\n{directory}: {n} images scanned, {hits} collide with CICIL data.")
    return hits


def _check_overlap(max_hamming: int) -> None:
    """Does any pilot (gold-eval) image also appear in dev or test?"""
    guard = pilot_index(max_hamming)
    total = 0
    for split in ("dev", "test"):
        for ref in cicil_images((split,)):
            col = guard.match(ref.path)
            if col:
                total += 1
                m = col.match
                print(f"OVERLAP {col.kind:<12} {split}/{ref.lang}/{ref.id} "
                      f"({ref.path.name})  ==  pilot/{m.lang}/{m.id}")
    if total == 0:
        print("No pilot image reappears in dev or test — gold-eval set is clean.")
    else:
        print(f"\n{total} dev/test images collide with the pilot set; "
              "exclude them from any training data.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dedup images against CICIL data.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan-dir", type=Path, help="check a directory of images")
    mode.add_argument("--check-overlap", action="store_true",
                      help="check pilot images against dev/test")
    ap.add_argument("--max-hamming", type=int, default=DEFAULT_MAX_HAMMING)
    args = ap.parse_args()

    if args.check_overlap:
        _check_overlap(args.max_hamming)
    else:
        _scan_dir(args.scan_dir, args.max_hamming)


if __name__ == "__main__":
    main()
