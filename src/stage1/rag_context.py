"""Stage 1 RAG banks: CBIR over Commons images + text retrieval over Wikipedia.

Two retrieval channels that give Stage 1 an "encyclopedia at inference time"
(the thing distillation demonstrably could not bake into 2B weights):

  * IMAGE (CBIR): CLIP-embed each culture's Commons scrape; at lookup time a
    dev image retrieves its nearest same-culture neighbors, whose titles and
    descriptions carry artifact/place names ("Nanduti detalle.jpg",
    "Wirikuta"). Written to a per-image context JSON consumed by
    generate_descriptions --context-json.
  * TEXT (encyclopedia): MiniLM-embed Wikipedia lead extracts harvested by
    scripts/harvest_wikipedia.py. Queried at generation time with the image's
    own VQA answers (see generate_descriptions --text-rag); the TextBank class
    here is that hook.

Build (once per bank refresh):
    uv run python -m src.stage1.rag_context --build-images --cultures guarani wixarika
    uv run python -m src.stage1.rag_context --build-text   --cultures guarani wixarika

Lookup / audit:
    uv run python -m src.stage1.rag_context --lookup --lang guarani
    uv run python -m src.stage1.rag_context --audit  --lang guarani

Indices land in indices/ (gitignored, regenerable); context JSONs in outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# torch MUST load before faiss on macOS: both bundle an OpenMP runtime, and if
# faiss's loads first the torch image-encode segfaults (exit 139, no traceback
# -- reproduced and bisected 2026-07-26). Importing torch here pins the winner.
import torch  # noqa: F401  (import-order side effect, see above)

from . import config
from .data_io import load_split

INDEX_DIR = config.ROOT / "indices"
COMMONS_DIR = config.ROOT / "data" / "external" / "commons"
WIKI_DIR = config.ROOT / "data" / "external" / "wikipedia"

CLIP_MODEL = "google/siglip-base-patch16-224"  # safetensors; same family as SmolVLM's vision tower
TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # = Stage 2's ENCODER_MODEL

CBIR_K = 3
CBIR_MIN_SCORE = 0.55   # CLIP cosine; tune from --lookup's printed distribution
CBIR_STRONG_SCORE = 0.80  # «coincidencia fuerte» band: rare by design (~5% of
                          # dev neighbors: 7/150 grn, 3/150 hch on 2026-07-26)
TEXT_K = 3
CONTEXT_CHAR_CAP = 600  # was 500; band tags add ~25 chars/neighbor


# ---------------------------------------------------------------- shared bits

class _SiglipEncoder:
    """Minimal image-embedding wrapper (SigLIP via transformers; fp32 for MPS)."""

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.device = device or config.device()
        self.model = AutoModel.from_pretrained(
            CLIP_MODEL, dtype=torch.float32).to(self.device).eval()
        # Image processor only: we never tokenize text with SigLIP, and the full
        # AutoProcessor would drag in a SentencePiece dependency for nothing.
        self.processor = AutoImageProcessor.from_pretrained(CLIP_MODEL)

    def encode(self, images: list, batch_size: int = 32):
        import numpy as np
        import torch
        chunks = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(images=images[i:i + batch_size],
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            if not torch.is_tensor(feats):  # newer transformers returns ModelOutput
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1)
            chunks.append(feats.cpu().numpy())
        return np.concatenate(chunks).astype("float32")


def _clip(device: str | None = None):
    return _SiglipEncoder(device)


def _minilm(device: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TEXT_MODEL, device=device or config.device())


def clean_title(commons_title: str) -> str:
    t = re.sub(r"^File:", "", commons_title)
    t = re.sub(r"\.(jpe?g|png|webp)$", "", t, flags=re.I)
    return t.replace("_", " ").strip()


def _load_provenance(culture: str) -> list[dict]:
    rows = []
    with (COMMONS_DIR / "provenance.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["culture"] == culture:
                rows.append(row)
    return rows


# ---------------------------------------------------------------- build

def build_images(cultures: list[str], device: str | None = None) -> None:
    import faiss
    from PIL import Image

    import numpy as np

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    model = _clip(device)
    for culture in cultures:
        rows = _load_provenance(culture)
        paths, meta = [], []
        for row in rows:
            path = COMMONS_DIR / row["local_file"]
            if not path.exists():
                continue
            paths.append(path)
            meta.append({"local_file": row["local_file"],
                         "title": clean_title(row["commons_title"]),
                         "description": row["description"][:300]})
        print(f"[{culture}] encoding {len(paths)} Commons images ...")
        # Stream in small batches: decoding hundreds of full-res Commons images
        # into RAM at once OOM-killed this build on a 16GB machine.
        chunks, kept_meta = [], []
        BATCH = 16
        for i in range(0, len(paths), BATCH):
            batch_imgs, batch_meta = [], []
            for p, m in zip(paths[i:i + BATCH], meta[i:i + BATCH]):
                try:
                    with Image.open(p) as im:
                        batch_imgs.append(im.convert("RGB"))
                    batch_meta.append(m)
                except OSError:
                    continue
            if batch_imgs:
                chunks.append(model.encode(batch_imgs, batch_size=BATCH))
                kept_meta += batch_meta
            if (i // BATCH) % 10 == 0:
                print(f"  {min(i + BATCH, len(paths))}/{len(paths)}")
        emb = np.concatenate(chunks)
        meta = kept_meta
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(INDEX_DIR / f"cbir_{culture}.index"))
        with (INDEX_DIR / f"cbir_{culture}_meta.jsonl").open("w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"[{culture}] cbir index: {index.ntotal} vectors")


def build_text(cultures: list[str]) -> None:
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    model = _minilm()
    for culture in cultures:
        path = WIKI_DIR / f"{culture}_text.jsonl"
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        texts = [f"{r['title']}. {r['extract']}" for r in rows]
        print(f"[{culture}] encoding {len(texts)} Wikipedia extracts ...")
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           convert_to_numpy=True).astype("float32")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(INDEX_DIR / f"wikitext_{culture}.index"))
        with (INDEX_DIR / f"wikitext_{culture}_meta.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{culture}] wikitext index: {index.ntotal} vectors")


# ---------------------------------------------------------------- retrieval

class ImageBank:
    def __init__(self, culture: str, device: str | None = None):
        import faiss
        self.index = faiss.read_index(str(INDEX_DIR / f"cbir_{culture}.index"))
        self.meta = [json.loads(l) for l in
                     (INDEX_DIR / f"cbir_{culture}_meta.jsonl").open(encoding="utf-8")]
        self.model = _clip(device)

    def neighbors(self, image_path: Path, k: int = CBIR_K) -> list[dict]:
        from PIL import Image
        with Image.open(image_path) as im:
            emb = self.model.encode([im.convert("RGB")])
        scores, idxs = self.index.search(emb, k)
        return [{**self.meta[i], "score": float(s)}
                for s, i in zip(scores[0], idxs[0]) if i >= 0]


class TextBank:
    """Generation-time hook: retrieve encyclopedia snippets for a VQA-answer query."""

    def __init__(self, culture: str, device: str | None = None):
        import faiss
        self.index = faiss.read_index(str(INDEX_DIR / f"wikitext_{culture}.index"))
        self.meta = [json.loads(l) for l in
                     (INDEX_DIR / f"wikitext_{culture}_meta.jsonl").open(encoding="utf-8")]
        self.model = _minilm(device)

    def retrieve(self, query: str, k: int = TEXT_K) -> list[dict]:
        emb = self.model.encode([query], normalize_embeddings=True,
                                convert_to_numpy=True).astype("float32")
        scores, idxs = self.index.search(emb, k)
        return [{**self.meta[i], "score": float(s)}
                for s, i in zip(scores[0], idxs[0]) if i >= 0]


def build_image_context(neighbors: list[dict], min_score: float = CBIR_MIN_SCORE) -> str:
    hits = [n for n in neighbors if n["score"] >= min_score]
    if not hits:
        return ""
    parts = []
    for i, n in enumerate(hits, 1):
        desc = n["description"].strip()
        band = ("coincidencia fuerte" if n["score"] >= CBIR_STRONG_SCORE
                else "coincidencia posible")
        parts.append(f"{i}) [{band}] {n['title']}" + (f": {desc}" if desc else ""))
    ctx = "Imágenes similares de esta cultura en Wikimedia Commons: " + " ".join(parts)
    return ctx[:CONTEXT_CHAR_CAP]


# ---------------------------------------------------------------- commands

def cmd_lookup(lang: str, split: str, k: int, min_score: float,
               device: str | None = None) -> None:
    bank = ImageBank(lang, device)
    examples = load_split(lang, split)
    out, scores_all = {}, []
    for ex in examples:
        if not ex.image_path.exists():
            continue
        neigh = bank.neighbors(ex.image_path, k)
        scores_all += [n["score"] for n in neigh]
        ctx = build_image_context(neigh, min_score)
        if ctx:
            out[ex.id] = ctx
    path = config.OUTPUT_DIR / f"cbir_context_{lang}_{split}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    import numpy as np
    s = np.array(scores_all)
    print(f"[{lang}] {len(out)}/{len(examples)} images got context "
          f"(min_score={min_score}); neighbor score distribution: "
          f"min={s.min():.2f} p25={np.percentile(s,25):.2f} "
          f"median={np.median(s):.2f} p75={np.percentile(s,75):.2f} max={s.max():.2f}")
    print(f"Wrote {path}")


def cmd_lookup_commons(culture: str, k: int, min_score: float) -> None:
    """CBIR context for each bank image itself — RAG-aware silver captioning.

    The bank vectors already live in the FAISS index, so this searches the index
    against its own reconstructed vectors (no image re-encoding, no torch model).
    Self-matches are excluded: same local_file, or near-duplicates at >= 0.995
    (copying a bank image's own Commons caption back to it would be circular).
    Output keyed by local_file, silver_caption's per-row handle.
    """
    import numpy as np
    import torch
    torch.ones(4).sum()  # init torch's OpenMP before faiss's loads (macOS dual-OMP)
    import faiss

    index = faiss.read_index(str(INDEX_DIR / f"cbir_{culture}.index"))
    meta = [json.loads(l) for l in
            (INDEX_DIR / f"cbir_{culture}_meta.jsonl").open(encoding="utf-8")]
    vecs = index.reconstruct_n(0, index.ntotal)
    # Self-similarity in numpy, not faiss.search: faiss's OMP parallel region
    # aborts under the torch/faiss dual-runtime on macOS, and an n~500 matmul
    # doesn't need faiss anyway.
    sims = vecs @ vecs.T
    idxs = np.argsort(-sims, axis=1)[:, : k + 3]  # +3: self + possible near-dupes
    scores = np.take_along_axis(sims, idxs, axis=1)
    out, n_strong = {}, 0
    for row_i, m in enumerate(meta):
        neigh = []
        for s, i in zip(scores[row_i], idxs[row_i]):
            if i < 0 or i == row_i:
                continue
            n = meta[i]
            if n["local_file"] == m["local_file"] or s >= 0.995:
                continue
            neigh.append({**n, "score": float(s)})
        neigh = neigh[:k]
        ctx = build_image_context(neigh, min_score)
        if ctx:
            out[m["local_file"]] = ctx
            n_strong += "coincidencia fuerte" in ctx
    path = config.OUTPUT_DIR / f"cbir_context_commons_{culture}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{culture}] {len(out)}/{len(meta)} bank images got context "
          f"({n_strong} with a strong-band neighbor)")
    print(f"Wrote {path}")


def cmd_audit(lang: str) -> None:
    """Image neighbors + text snippets for this lang's human-eval sample images."""
    key_csv = config.ROOT / "analysis" / "human_eval" / "sample_key.csv"
    ids = []
    with key_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["image_id"].split("_")[0] == {"guarani": "grn", "maya": "yua",
                                                  "wixarika": "hch", "nahuatl": "nlv",
                                                  "bribri": "bzd"}[lang]:
                ids.append(row["image_id"])
    examples = {ex.id: ex for ex in load_split(lang, "dev")}

    # existing VQA answers as the text query (ollama-arm dev file)
    vqa_path = config.OUTPUT_DIR / f"{lang}_dev_cultural-vqa_ollama.jsonl"
    answers = {}
    for line in vqa_path.open(encoding="utf-8"):
        r = json.loads(line)
        answers[r["id"]] = " ".join(v for v in r["cultural_annotations"].values() if v)

    ibank, tbank = ImageBank(lang), TextBank(lang)
    lines = [f"# RAG audit — {lang} (human-eval sample images)\n"]
    for iid in ids:
        ex = examples.get(iid)
        if ex is None:
            continue
        lines.append(f"\n## {iid}\n")
        lines.append("**Image neighbors (CLIP):**\n")
        for n in ibank.neighbors(ex.image_path):
            lines.append(f"- {n['score']:.2f} — {n['title']}"
                         + (f" — {n['description'][:120]}" if n['description'] else ""))
        q = answers.get(iid, "")
        lines.append(f"\n**Text query (VQA answers, truncated):** {q[:200]}\n")
        lines.append("**Text snippets (MiniLM):**\n")
        for t in tbank.retrieve(q or ex.id):
            lines.append(f"- {t['score']:.2f} — {t['title']} — {t['extract'][:160]}")
    path = config.OUTPUT_DIR / f"rag_audit_{lang}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 RAG banks: build/lookup/audit.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build-images", action="store_true")
    g.add_argument("--build-text", action="store_true")
    g.add_argument("--lookup", action="store_true")
    g.add_argument("--lookup-commons", action="store_true",
                   help="CBIR context for the bank images themselves (self-matches "
                        "excluded) — feeds RAG-aware silver captioning.")
    g.add_argument("--audit", action="store_true")
    ap.add_argument("--cultures", nargs="+", default=["guarani", "wixarika"],
                    choices=config.LANGUAGES)
    ap.add_argument("--lang", default=None, choices=config.LANGUAGES)
    ap.add_argument("--split", default="dev", choices=["pilot", "dev", "test"])
    ap.add_argument("--k", type=int, default=CBIR_K)
    ap.add_argument("--min-score", type=float, default=CBIR_MIN_SCORE)
    ap.add_argument("--device", default=None,
                    help="Override torch device (e.g. cpu). MPS segfaults intermittently "
                         "on this SigLIP encode workload; cpu is the safe one-time choice.")
    args = ap.parse_args()

    if args.build_images:
        build_images(args.cultures, device=args.device)
    elif args.build_text:
        build_text(args.cultures)
    elif args.lookup:
        if not args.lang:
            raise SystemExit("--lookup needs --lang")
        cmd_lookup(args.lang, args.split, args.k, args.min_score, device=args.device)
    elif args.lookup_commons:
        for culture in args.cultures:
            cmd_lookup_commons(culture, args.k, args.min_score)
    else:
        if not args.lang:
            raise SystemExit("--audit needs --lang")
        cmd_audit(args.lang)


if __name__ == "__main__":
    main()
