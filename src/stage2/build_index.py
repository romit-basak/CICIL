"""Stage 2 — Step 1: build a FAISS retrieval index per language.

Encodes the Spanish side of each available parallel corpus with
``paraphrase-multilingual-MiniLM-L12-v2`` and writes one ``IndexFlatIP`` (cosine
similarity over L2-normalized embeddings) per language to ``indices/``.

Run once; ``translate.py`` reuses the saved indices.

    uv run python -m src.stage2.build_index

Current reality (2026-07): the large parallel banks (AmericasNLP 2023 ~53k
Guaraní, 2021 ~7.5k Bribri) are **not downloaded** and the dev JSONLs carry only
``target_caption`` (no Spanish side). The one corpus with genuine Spanish↔target
pairs is the **Wixárika pilot** (20 rows). So only ``wixarika`` gets an index;
the other languages have no corpus here and are translated zero-shot by
``translate.py``. Add entries to ``CORPORA`` once Nandita's banks land.
"""

from __future__ import annotations

import json

from src.stage1.data_io import load_split
from .paths import ENCODER_MODEL, INDEX_DIR

# (lang, split) sources that actually contain Spanish↔target pairs today.
# Extend this dict as real retrieval banks become available.
CORPORA: dict[str, tuple[str, str]] = {
    "wixarika": ("wixarika", "pilot"),  # 20 pairs (only split with spanish_caption)
}


def _load_pairs(lang: str, split: str) -> list[dict]:
    """Spanish↔target pairs for a (lang, split), keeping only complete rows."""
    pairs = []
    for ex in load_split(lang, split):
        spanish = (ex.spanish_caption or "").strip()
        target = (ex.target_caption or "").strip()
        if spanish and target:
            pairs.append({"spanish": spanish, "target": target})
    return pairs


def main() -> None:
    from sentence_transformers import SentenceTransformer
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading sentence encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL)

    for lang, (src_lang, split) in CORPORA.items():
        print(f"\n-- {lang.upper()} ({src_lang}/{split}) " + "-" * 30)
        try:
            pairs = _load_pairs(src_lang, split)
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
