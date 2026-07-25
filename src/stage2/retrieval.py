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


def _cultural_annotation_query(record: dict) -> str:
    """Join cultural-annotation values into a single query string, or ''."""
    annotations = record.get("cultural_annotations", {}) or {}
    parts = [f"{k}: {v}" for k, v in annotations.items() if v and str(v).strip()]
    return "  ".join(parts)


def build_query_from_record(record: dict, query_arm: str = "auto") -> str:
    """Retrieval query from a Stage 1 record — the key Stage 1↔2 coupling.

    This is the RQ1 headline switch: "does culturally-indexed retrieval beat
    vanilla text retrieval?" is only a real test if the query strategy can be
    set independently of the Stage 1 prompt mode. Three arms:

    query_arm="cultural"
        Force the cultural-annotation query (join annotation values) so
        retrieval is indexed on cultural relevance rather than surface
        lexical similarity. If a record genuinely carries no annotations
        (e.g. a generic-mode record run through this arm by mistake), falls
        back to the Spanish text so the pipeline doesn't crash or emit an
        empty query -- but that fallback should be rare/absent for records
        actually drawn from the cultural-vqa split.
    query_arm="text"
        Force the plain Spanish-text query, regardless of whether cultural
        annotations are present. This is the vanilla-retrieval control arm:
        same k, same index, same prompt template -- only the query changes.
    query_arm="auto" (default; pre-ablation / back-compat behavior)
        Cultural annotations if present, else Spanish text. This is what the
        pipeline did before the arms were split out, and it silently couples
        query strategy to Stage 1 prompt mode (cultural-vqa records always
        got cultural queries, generic records always got text queries) --
        which is exactly the confound the "cultural" / "text" arms above are
        for. Kept only so existing callers that don't pass query_arm keep
        their old behavior.
    """
    if query_arm not in ("auto", "cultural", "text"):
        raise ValueError(
            f"Unknown query_arm: {query_arm!r} (expected 'auto', 'cultural', or 'text')"
        )

    spanish_text = record.get("generated_spanish", "")

    if query_arm == "text":
        return spanish_text

    cultural_query = _cultural_annotation_query(record)
    if query_arm == "cultural":
        return cultural_query or spanish_text

    # auto
    return cultural_query or spanish_text
