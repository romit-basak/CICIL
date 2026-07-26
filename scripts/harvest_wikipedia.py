"""Harvest a per-culture Wikipedia knowledge bank: text extracts + Commons seeds.

For each culture, walk 1 hop out from hub articles on es/en Wikipedia, keep
culturally relevant links (two-rule filter below), and fetch for each kept
article (a) its lead extract -> data/external/wikipedia/<culture>_text.jsonl,
(b) its Commons category (if any) -> <culture>_commons_seeds.txt for
scrape_commons.py --extra-seeds.

Relevance filter (order matters):
  1. KEEP if the lead extract mentions the culture's name(s) -- type-agnostic,
     so sacred geography (Wirikuta), territories, and missions pass exactly
     like artifacts and ceremonies. Landscape is one of the dataset's four VQA
     categories; places are first-class cultural content here.
  2. Otherwise KEEP if embedding similarity to a multi-facet keyword centroid
     (artifacts + ceremonies + sacred-geography vocabulary) clears --min-sim.
  Everything else (politicians, sports clubs, incidental links) is dropped.

The final seed list is printed for human review BEFORE any scraping -- the
Bribri "Talamanca" homonym poisoning is the cautionary tale.

Wikipedia text is CC BY-SA (attribution recorded per row: title + url).

    uv run python scripts/harvest_wikipedia.py --cultures guarani wixarika
    uv run python scripts/harvest_wikipedia.py --cultures guarani --min-sim 0.35
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stage1 import config  # noqa: E402

OUT_DIR = config.ROOT / "data" / "external" / "wikipedia"

UA = {"User-Agent": "CICIL-coursework/0.1 (basak.r@northeastern.edu) requests"}
API = {"es": "https://es.wikipedia.org/w/api.php",
       "en": "https://en.wikipedia.org/w/api.php"}
SUMMARY = {"es": "https://es.wikipedia.org/api/rest_v1/page/summary/",
           "en": "https://en.wikipedia.org/api/rest_v1/page/summary/"}

# Hub articles per culture, per wiki. 1-hop links from these are the candidates.
HUBS: dict[str, dict[str, list[str]]] = {
    "guarani": {
        "es": ["Pueblo guaraní", "Cultura de Paraguay", "Arte de Paraguay",
               "Misiones jesuíticas guaraníes"],
        "en": ["Guaraní people", "Culture of Paraguay"],
    },
    "wixarika": {
        "es": ["Huicholes", "Arte huichol", "Wirikuta"],
        "en": ["Huichol", "Huichol art"],
    },
    "maya": {
        "es": ["Cultura maya", "Pueblo maya", "Península de Yucatán"],
        "en": ["Maya civilization", "Maya peoples"],
    },
    "bribri": {
        "es": ["Bribri", "Cordillera de Talamanca"],
        "en": ["Bribri people"],
    },
    "nahuatl": {
        "es": ["Pueblos nahuas", "Cultura mexica"],
        "en": ["Nahuas"],
    },
}

# Rule-1 names: keep any article whose lead mentions one of these (TEXT BANK —
# deliberately broad; per-query retrieval ranking tolerates breadth).
CULTURE_NAMES: dict[str, list[str]] = {
    "guarani": ["guaraní", "guarani", "avañe'ẽ", "paraguay"],
    "wixarika": ["wixárika", "wixarika", "huichol"],
    "maya": ["maya", "yucatec", "yucateco"],
    "bribri": ["bribri", "talamanca"],
    "nahuatl": ["náhuatl", "nahuatl", "nahua", "mexica"],
}

# Narrow names for COMMONS SEED selection only: seed categories multiply into
# hundreds of scraped images, so "the extract mentions Paraguay" (true of every
# dictator and football club) is not enough — the culture itself must be named.
SEED_NAMES: dict[str, list[str]] = {
    "guarani": ["guaraní", "guarani", "avañe'ẽ"],
    "wixarika": ["wixárika", "wixarika", "huichol"],
    "maya": ["maya", "yucatec", "yucateco"],
    "bribri": ["bribri"],
    "nahuatl": ["náhuatl", "nahuatl", "nahua", "mexica"],
}

# Rule-2 centroid seeds: all four VQA categories, including sacred geography.
CENTROID_TERMS = [
    # material culture / artifacts
    "artesanía tradicional indígena", "textil bordado tejido tradicional",
    "cerámica cestería arte popular", "vestimenta traje típico indígena",
    # ceremony / practices
    "ceremonia ritual danza tradicional", "fiesta religiosa peregrinación",
    "música instrumento tradicional", "chamán curandero rito",
    # landscape / sacred geography (first-class, not noise)
    "sitio sagrado montaña sagrada", "territorio ancestral indígena",
    "peregrinación lugar sagrado", "misión jesuítica pueblo histórico",
    "cenote río selva sagrada",
    # kinship / community life
    "comunidad indígena vida cotidiana", "familia parentesco aldea",
    # food/agriculture (recurrent in the dataset's daily-life images)
    "milpa maíz agricultura tradicional", "comida gastronomía tradicional",
]


def api_get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def get_links(wiki: str, title: str) -> list[str]:
    """All main-namespace links from one article (paginated)."""
    links, cont = [], {}
    while True:
        data = api_get(API[wiki], {
            "action": "query", "prop": "links", "titles": title,
            "plnamespace": 0, "pllimit": "max", "format": "json", **cont,
        })
        for page in data.get("query", {}).get("pages", {}).values():
            links += [l["title"] for l in page.get("links", [])]
        if "continue" not in data:
            return links
        cont = data["continue"]
        time.sleep(0.1)


def get_summary(wiki: str, title: str) -> dict | None:
    try:
        r = requests.get(SUMMARY[wiki] + requests.utils.quote(title.replace(" ", "_")),
                         headers=UA, timeout=30)
        if r.status_code != 200:
            return None
        d = r.json()
        if d.get("type") == "disambiguation" or not d.get("extract"):
            return None
        return {"title": d["title"], "extract": d["extract"],
                "url": d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "lang": wiki}
    except requests.RequestException:
        return None


def get_commons_category(wiki: str, titles: list[str]) -> dict[str, str]:
    """title -> Commons category name (from wikibase items' P373 via pageprops)."""
    out: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = api_get(API[wiki], {
            "action": "query", "prop": "pageprops", "titles": "|".join(batch),
            "ppprop": "wikibase_item", "format": "json", "redirects": 1,
        })
        qids = {p["title"]: p["pageprops"]["wikibase_item"]
                for p in data.get("query", {}).get("pages", {}).values()
                if "pageprops" in p and "wikibase_item" in p["pageprops"]}
        if not qids:
            continue
        wd = api_get("https://www.wikidata.org/w/api.php", {
            "action": "wbgetclaims" if len(qids) == 1 else "wbgetentities",
            **({"entity": list(qids.values())[0], "property": "P373"}
               if len(qids) == 1 else
               {"ids": "|".join(qids.values()), "props": "claims", "format": "json"}),
            "format": "json",
        })
        entities = wd.get("entities", {}) or (
            {list(qids.values())[0]: {"claims": wd.get("claims", {})}})
        for title, qid in qids.items():
            claims = entities.get(qid, {}).get("claims", {})
            p373 = claims.get("P373", [])
            if p373:
                try:
                    out[title] = p373[0]["mainsnak"]["datavalue"]["value"]
                except (KeyError, TypeError):
                    pass
        time.sleep(0.2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest Wikipedia text bank + Commons seeds per culture.")
    ap.add_argument("--cultures", nargs="+", default=["guarani", "wixarika"],
                    choices=list(HUBS))
    ap.add_argument("--min-sim", type=float, default=0.30,
                    help="Rule-2 centroid cosine threshold (rule-1 name matches always pass).")
    ap.add_argument("--cap", type=int, default=150, help="Max kept articles per culture.")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import numpy as np
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",
                                  device=config.device())
    centroid = encoder.encode(CENTROID_TERMS, normalize_embeddings=True).mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for culture in args.cultures:
        print(f"\n=== {culture.upper()} ===")
        names = CULTURE_NAMES[culture]

        # 1. candidates = 1-hop links from all hubs (both wikis), deduped
        candidates: dict[str, str] = {}  # title -> wiki it came from (es preferred)
        for wiki in ("es", "en"):
            for hub in HUBS[culture].get(wiki, []):
                links = get_links(wiki, hub)
                print(f"  {wiki}:{hub}: {len(links)} links")
                for t in links:
                    candidates.setdefault(t, wiki)
        # hubs themselves are part of the bank
        for wiki in ("es", "en"):
            for hub in HUBS[culture].get(wiki, []):
                candidates[hub] = wiki
        print(f"  {len(candidates)} unique candidates")

        # 2. fetch summaries for ALL candidates, then apply the two-rule filter.
        # (No pre-rank cap before fetching: rule 1 needs the extract, and
        # native-language titles like "Ñandutí" embed poorly against Spanish
        # keyword centroids — a title-similarity cap silently drops exactly the
        # artifact articles this harvest exists to find.)
        titles = sorted(candidates)
        sims = dict(zip(titles, encoder.encode(titles, normalize_embeddings=True) @ centroid))
        rule1_kept, rule2_kept, dropped = [], [], 0
        for title in titles:
            wiki = candidates[title]
            summ = get_summary(wiki, title)
            if summ is None:
                dropped += 1
                continue
            summ["culture"] = culture
            extract_l = summ["extract"].lower()
            if any(n in extract_l for n in names):
                summ["kept_by"] = "name-match"
                rule1_kept.append(summ)
            elif sims[title] >= args.min_sim:
                summ["kept_by"] = f"centroid={sims[title]:.2f}"
                rule2_kept.append((sims[title], summ))
            else:
                dropped += 1
            time.sleep(0.05)
        # Bank = every rule-1 match, then best rule-2 matches up to the cap.
        rule2_kept.sort(key=lambda x: -x[0])
        kept = rule1_kept + [s for _, s in rule2_kept][: max(0, args.cap - len(rule1_kept))]
        print(f"  kept {len(kept)} articles "
              f"({len(rule1_kept)} name-matched, "
              f"{len(kept) - len(rule1_kept)} centroid, {dropped} dropped)")

        # 3a. text bank
        text_path = OUT_DIR / f"{culture}_text.jsonl"
        with text_path.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {text_path}")

        # 3b. Commons seeds — from articles matching the NARROW seed names only.
        # Seed categories multiply into hundreds of scraped images, so they need
        # the high-precision rule: broad rule-1 matches ("mentions Paraguay")
        # and centroid-kept articles ("Crafts of Oaxaca") would poison the image
        # bank with pan-regional material (the Talamanca-homonym lesson).
        seed_names = SEED_NAMES[culture]
        by_wiki: dict[str, list[str]] = {}
        for row in rule1_kept:
            if any(n in row["extract"].lower() for n in seed_names):
                by_wiki.setdefault(row["lang"], []).append(row["title"])
        seeds: set[str] = set()
        for wiki, ts in by_wiki.items():
            seeds |= set(get_commons_category(wiki, ts).values())
        seeds_path = OUT_DIR / f"{culture}_commons_seeds.txt"
        seeds_path.write_text("\n".join(sorted(seeds)) + "\n", encoding="utf-8")
        print(f"  wrote {seeds_path} ({len(seeds)} Commons categories)")

        # 4. human-review list
        print(f"\n  --- REVIEW: Commons seeds for {culture} (eyeball before scraping) ---")
        for s in sorted(seeds):
            print(f"    {s}")


if __name__ == "__main__":
    main()
