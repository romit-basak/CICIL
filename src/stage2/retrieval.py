"""Stage 2 — retrieval module (imported by translate.py; not run directly).

Loads a pre-built FAISS index for one language and returns the top-k most
semantically/culturally similar Spanish↔target pairs for a query string. Build
the index first with ``uv run python -m src.stage2.build_index``.
"""

from __future__ import annotations

import json

from .paths import ENCODER_MODEL, INDEX_DIR

# Singleton encoder: loaded once, reused for every retrieve() call.
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(ENCODER_MODEL)
    return _encoder


class Retriever:
    """FAISS index + parallel pairs for one language.

    Parameters
    ----------
    lang : str   e.g. "wixarika"
    k    : int   number of examples to retrieve per query (ablation: 3 / 5 / 8)
    """

    def __init__(self, lang: str, k: int = 5):
        import faiss

        self.lang = lang
        self.k = k

        index_path = INDEX_DIR / f"{lang}.index"
        pairs_path = INDEX_DIR / f"{lang}_pairs.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run: uv run python -m src.stage2.build_index"
            )

        self.index = faiss.read_index(str(index_path))
        self.pairs = []
        with pairs_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pairs.append(json.loads(line))

        print(f"[Retriever] {lang}: {self.index.ntotal:,} vectors loaded, k={k}")

    def retrieve(self, query: str) -> list:
        """Return up to k similar pairs (dicts with 'spanish' and 'target')."""
        encoder = _get_encoder()
        query_vec = encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        # Never request more neighbours than the index holds (tiny pilot banks).
        top = min(self.k, self.index.ntotal)
        _scores, indices = self.index.search(query_vec, top)
        return [self.pairs[i] for i in indices[0] if 0 <= i < len(self.pairs)]


def build_query_from_record(record: dict) -> str:
    """Retrieval query from a Stage 1 record — the key Stage 1↔2 coupling.

    Cultural-VQA arm (``cultural_annotations`` populated): query on the cultural
    annotation values, so retrieval targets cultural relevance over surface
    lexical similarity. Generic arm (empty annotations): fall back to the
    generated Spanish description.
    """
    annotations = record.get("cultural_annotations", {})
    if annotations:
        parts = [f"{k}: {v}" for k, v in annotations.items() if v and str(v).strip()]
        if parts:
            return "  ".join(parts)
    return record.get("generated_spanish", "")
