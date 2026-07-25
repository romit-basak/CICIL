"""Stage 2 — Step 1: build a FAISS retrieval index per language.

Encodes the Spanish side of each available parallel corpus with
``paraphrase-multilingual-MiniLM-L12-v2`` and writes one ``IndexFlatIP`` (cosine
similarity over L2-normalized embeddings) per language to ``indices/``.

Run once; ``translate.py`` reuses the saved indices.

    uv run python -m src.stage2.build_index

Current reality (2026-07-24): real retrieval banks landed in ``Dataset/`` (Nandita,
via ``byuild_corpora.py`` -- AmericasNLP 2021+2023 train+dev, deduplicated). See
``DATA_LICENSES.md`` for source/license per file. Bribri, Guaraní, Nahuatl, and
Wixárika all get a real index now; **Yucatec Maya has no retrieval-bank source** (not
in AmericasNLP 2021's language list -- see ``STAGE2_HANDOFF.md``) and is still
translated zero-shot by ``translate.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.stage1 import config
from .paths import ENCODER_MODEL, INDEX_DIR

DATASET_DIR = config.ROOT / "Dataset"

# lang -> Dataset/*.jsonl path. Extend once a Maya retrieval-bank source is found.
CORPORA: dict[str, Path] = {
    "bribri": DATASET_DIR / "bribri.jsonl",
    "guarani": DATASET_DIR / "guarani.jsonl",
    "nahuatl": DATASET_DIR / "nahuatl.jsonl",
    "wixarika": DATASET_DIR / "wixarika.jsonl",
}


def _load_pairs(path: Path) -> list[dict]:
    """Spanish↔target pairs from a Dataset/*.jsonl file, keeping only complete rows."""
    if not path.exists():
        raise FileNotFoundError(path)
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            spanish = (row.get("spanish_caption") or "").strip()
            target = (row.get("target_caption") or "").strip()
            if spanish and target:
                pairs.append({"spanish": spanish, "target": target})
    return pairs


def main() -> None:
    from sentence_transformers import SentenceTransformer
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading sentence encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    for lang, path in CORPORA.items():
        print(f"\n-- {lang.upper()} ({path.relative_to(config.ROOT)}) " + "-" * 30)
        try:
            pairs = _load_pairs(path)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} -- skipping.")
            continue
        if not pairs:
            print("  WARNING: 0 Spanish↔target pairs -- skipping (no Spanish side).")
            continue
        print(f"  Loaded {len(pairs):,} pairs")

        embeddings = encoder.encode(
            [p["spanish"] for p in pairs],
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        print(f"  Index built: {index.ntotal:,} vectors, dim={embeddings.shape[1]}")

        index_path = INDEX_DIR / f"{lang}.index"
        pairs_path = INDEX_DIR / f"{lang}_pairs.jsonl"
        faiss.write_index(index, str(index_path))
        with pairs_path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  Saved: {index_path}\n  Saved: {pairs_path}")

    print("\nDone. Languages without an index are translated zero-shot by translate.py.")


if __name__ == "__main__":
    main()
