"""Attach each human-eval image's top CBIR retrieval neighbor for annotation.

For RAG arms, every caption was generated WITH a retrieved Commons image as
context (src/stage1/rag_context.py). This script re-runs that same lookup
(deterministic -- same index, same encoder) for the sample images in a given
round, and writes the neighbor's title/description/score plus a live link to
its actual Wikimedia Commons page (never the image bytes themselves -- Commons
images stay local per DATA_LICENSES.md; a hyperlink is not redistribution).

Purpose: let the annotator judge retrieval quality directly ("does this
reference actually look related?"), not just the caption it fed into --
a bad-but-unused retrieval and a bad-and-misleading one are different failures.

Only meaningful for rounds comparing RAG arms (round 2, round 3); round 1
(generic vs cultural, pre-RAG) has nothing to attach and is skipped by
build_interface.py automatically when this file doesn't exist.

Run:
    uv run python -m analysis.human_eval.build_cbir_refs --suffix _round2
    uv run python -m analysis.human_eval.build_cbir_refs --suffix _round3 --split pilot
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import quote

import torch  # noqa: F401  -- see rag_context.py: must load before faiss on macOS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMONS_DIR = ROOT / "data" / "external" / "commons"

LANG_DIR = {"grn": "guarani", "yua": "maya", "hch": "wixarika",
            "nlv": "nahuatl", "bzd": "bribri"}


def _load_provenance(culture: str) -> dict[str, dict]:
    path = COMMONS_DIR / "provenance.csv"
    by_local_file = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["culture"] == culture:
                by_local_file[row["local_file"]] = row
    return by_local_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach top CBIR neighbor per sample.")
    ap.add_argument("--suffix", required=True, help="round suffix, e.g. _round2")
    ap.add_argument("--split", default="dev", choices=["pilot", "dev", "test"])
    args = ap.parse_args()

    from src.stage1.data_io import load_split
    from src.stage1.rag_context import ImageBank, CBIR_STRONG_SCORE

    with (HERE / f"sample_key{args.suffix}.csv").open(encoding="utf-8") as f:
        key_rows = list(csv.DictReader(f))

    banks: dict[str, ImageBank] = {}
    prov: dict[str, dict] = {}
    ex_by_id: dict[str, dict] = {}

    out_rows = []
    for row in key_rows:
        iid = row["image_id"]
        culture = LANG_DIR[iid.split("_")[0]]
        if culture not in banks:
            banks[culture] = ImageBank(culture, device="cpu")
            prov[culture] = _load_provenance(culture)
            ex_by_id[culture] = {ex.id: ex for ex in load_split(culture, args.split)}

        ex = ex_by_id[culture].get(iid)
        if ex is None:
            print(f"  [skip] {iid} not found in {culture}/{args.split}")
            continue
        neigh = banks[culture].neighbors(ex.image_path, k=1)
        if not neigh:
            out_rows.append({"sample_id": row["sample_id"], "image_id": iid,
                             "cbir_title": "", "cbir_description": "",
                             "cbir_image_url": "", "cbir_score": "", "cbir_band": "",
                             "cbir_page_url": ""})
            continue
        n = neigh[0]
        band = "fuerte" if n["score"] >= CBIR_STRONG_SCORE else "posible"
        prow = prov[culture].get(n["local_file"], {})
        # Special:FilePath is Commons' own hotlink redirect to the actual file --
        # needs the raw filename (with extension), not the display title (which
        # clean_title() stripped). A live <img> at this URL is NOT redistribution
        # (no bytes copied into the repo); it's the same mechanism Wikipedia
        # itself uses to embed Commons media.
        raw_title = prow.get("commons_title", "").removeprefix("File:")
        image_url = (f"https://commons.wikimedia.org/wiki/Special:FilePath/"
                     f"{quote(raw_title)}?width=500") if raw_title else ""
        out_rows.append({
            "sample_id": row["sample_id"], "image_id": iid,
            "cbir_title": n["title"], "cbir_description": n["description"],
            "cbir_image_url": image_url,
            "cbir_score": f"{n['score']:.3f}", "cbir_band": band,
            "cbir_page_url": prow.get("page_url", ""),
        })
        print(f"  {row['sample_id']} ({iid}): {n['title']} [{band}, {n['score']:.2f}]")

    out_path = HERE / f"sample_cbir_ref{args.suffix}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "image_id", "cbir_title",
                                         "cbir_description", "cbir_image_url",
                                         "cbir_score", "cbir_band", "cbir_page_url"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
