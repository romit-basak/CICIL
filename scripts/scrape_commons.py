"""Scrape culturally-relevant, freely-licensed images from Wikimedia Commons.

Feeds the Stage 1 distillation set: for each CICIL culture we walk a curated set
of Commons categories (BFS over subcategories, bounded depth), download images
that pass a strict license allowlist (PD / CC0 / CC BY / CC BY-SA only), and
record per-image provenance (source URL, license, author, description) in
``provenance.csv`` — the attribution record that satisfies BY/BY-SA terms.
Images land under gitignored ``data/external/commons/`` and are never
redistributed via the repo, same policy as the CC BY-NC shared-task data.

Every download is deduplicated against the CICIL images (sha1 + dHash via
``src.stage1.dedup``) so no shared-task image — above all no gold-eval pilot
image — can re-enter through the back door, and against the scrape itself
(Commons hosts many re-uploads).

Category descriptions (often encyclopedic) are kept: they are later fed to the
*teacher* VLM as context during silver captioning, which is where their extra
cultural signal pays off.

Run:
  uv run python scripts/scrape_commons.py --per-culture 10          # dry run
  uv run python scripts/scrape_commons.py --per-culture 300         # full
  uv run python scripts/scrape_commons.py --cultures bribri maya
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import time
from collections import deque
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage1 import config  # noqa: E402
from src.stage1.dedup import DedupIndex, ImageRef, cicil_index, dhash, sha1_file  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = ("CICIL-research-scraper/0.1 "
              "(Northeastern NLP course project; basak.r@northeastern.edu)")
PAUSE = 1.0          # seconds between HTTP requests (API etiquette)
RETRIES = 4          # on 429/5xx: exponential backoff starting at BACKOFF s
BACKOFF = 20.0
THUMB_WIDTH = 1024   # server-side downscale; plenty for VLM input
MIN_DIM = 300        # skip icons/thumbnails
BATCH = 50           # imageinfo titles per API call (API max for anon)

# Verified seed categories per CICIL culture (probed 2026-07-17; counts in
# parentheses are direct files/subcats at probe time). Depth-2 BFS reaches the
# richer subtrees, e.g. Maya peoples -> Yucatec (350 files) — the task's Maya
# variant — and Guaraní -> Mbya / Guarani-Kaiowá.
SEED_CATEGORIES: dict[str, list[str]] = {
    "wixarika": ["Huichol people", "Huichol art", "Vochol", "Wirikuta"],
    "maya": ["Maya peoples", "Yucatec"],
    "guarani": ["Guaraní"],
    # NOTE: NOT bare "Category:Talamanca" — that is Talamanca in Catalonia
    # (medieval castle, Romanesque churches) and it poisoned the first scrape
    # with ~200 Spanish-village images. "Talamanca Canton" is the Costa Rican
    # canton that is the Bribri heartland. Thin on Commons; expect < cap.
    "bribri": ["Bribris", "Talamanca Canton", "Puerto Viejo de Talamanca",
               "Hone Creek", "Cordillera de Talamanca"],
    "nahuatl": ["Nahua people"],
}

# Licenses we accept, by Commons LicenseShortName. Tight by design: PD/CC0/
# CC BY/CC BY-SA only; anything with NC/ND (shouldn't exist on Commons) or an
# unrecognized name (GFDL-only, "Copyrighted free use", ...) is skipped.
_LICENSE_ALLOW = re.compile(
    r"^(public domain|pd[- ]|pd$|cc0\b|no restrictions"
    r"|cc[- ]by(?:[- ]sa)?[- ]\d)", re.IGNORECASE)
_LICENSE_DENY = re.compile(r"\b(nc|nd|non[- ]?commercial|no[- ]?deriv)", re.IGNORECASE)

# Files/categories that are almost never culturally useful photographs.
_TITLE_BLOCK = re.compile(
    r"\b(map|maps|mapa|locator|flag|flags|bandera|coat[s]? of arms|escudo"
    r"|logo|seal|chart|diagram|genetic|distribution|linguistic maps"
    r"|named after)\b", re.IGNORECASE)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

PROVENANCE_FIELDS = [
    "culture", "local_file", "commons_title", "page_url", "license",
    "license_url", "artist", "credit", "description", "width", "height",
    "sha1", "dhash",
]


def license_ok(short_name: str) -> bool:
    s = (short_name or "").strip()
    return bool(s) and not _LICENSE_DENY.search(s) and bool(_LICENSE_ALLOW.match(s))


def clean_html(value: str, limit: int = 1200) -> str:
    """Strip tags/entities from an extmetadata HTML value; collapse whitespace."""
    text = _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value or ""))).strip()
    return text[:limit]


class Commons:
    """Thin, rate-limited MediaWiki API client."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def _request(self, url: str, params: dict | None = None) -> requests.Response:
        """One rate-limited GET with exponential backoff on 429/5xx.

        Wikimedia explicitly asks bots to slow down on 429 rather than retry
        hot; honoring Retry-After (or a growing default) keeps us welcome.
        """
        for attempt in range(RETRIES + 1):
            time.sleep(PAUSE)
            r = self.session.get(url, params=params, timeout=120)
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == RETRIES:
                r.raise_for_status()
                return r
            wait = float(r.headers.get("Retry-After") or BACKOFF * (2 ** attempt))
            print(f"  [throttle] HTTP {r.status_code}; backing off {wait:.0f}s")
            time.sleep(wait)
        raise RuntimeError("unreachable")

    def get(self, **params) -> dict:
        params.update(format="json", maxlag=5)
        return self._request(API, params).json()

    def category_members(self, cat: str, cmtype: str) -> list[str]:
        """All members of one type ('file' or 'subcat'), following pagination."""
        titles, cont = [], {}
        while True:
            r = self.get(action="query", list="categorymembers",
                         cmtitle=f"Category:{cat}", cmtype=cmtype,
                         cmlimit=500, **cont)
            titles += [m["title"] for m in r["query"]["categorymembers"]]
            cont = r.get("continue")
            if not cont:
                return titles

    def imageinfo(self, titles: list[str]) -> list[dict]:
        """imageinfo + extmetadata for up to BATCH file titles."""
        r = self.get(action="query", titles="|".join(titles), prop="imageinfo",
                     iiprop="url|extmetadata|size|sha1|mime",
                     iiurlwidth=THUMB_WIDTH)
        out = []
        for page in r["query"].get("pages", {}).values():
            infos = page.get("imageinfo")
            if infos:
                out.append({"title": page["title"], **infos[0]})
        return out

    def download(self, url: str, dest: Path) -> None:
        dest.write_bytes(self._request(url).content)


def walk_file_titles(api: Commons, seeds: list[str], depth: int) -> list[str]:
    """BFS over categories; return file titles in discovery order, deduped."""
    seen_cats, seen_files, ordered = set(), set(), []
    queue = deque((s, 0) for s in seeds)
    while queue:
        cat, d = queue.popleft()
        if cat in seen_cats or _TITLE_BLOCK.search(cat):
            continue
        seen_cats.add(cat)
        for t in api.category_members(cat, "file"):
            if t not in seen_files:
                seen_files.add(t)
                ordered.append(t)
        if d < depth:
            for sub in api.category_members(cat, "subcat"):
                queue.append((sub.removeprefix("Category:"), d + 1))
    return ordered


def load_provenance(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scrape_culture(api: Commons, culture: str, cap: int, depth: int,
                   out_root: Path, cicil: DedupIndex,
                   provenance: list[dict], writer: csv.DictWriter,
                   prov_file) -> int:
    img_dir = out_root / culture / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Resume state: already-scraped titles (all cultures) and this culture's
    # accepted count; intra-scrape dedup seeded from prior rows' hashes.
    done_titles = {row["commons_title"] for row in provenance}
    accepted = sum(1 for row in provenance if row["culture"] == culture)
    counter = accepted
    local = DedupIndex([], max_hamming=cicil.max_hamming)
    local._by_sha = {row["sha1"]: ImageRef(Path(row["local_file"]),
                                           row["culture"], "commons", row["local_file"])
                     for row in provenance}
    local._hashes = [(int(row["dhash"], 16),
                      ImageRef(Path(row["local_file"]), row["culture"],
                               "commons", row["local_file"]))
                     for row in provenance if row["dhash"]]

    if accepted >= cap:
        print(f"[{culture}] already have {accepted}/{cap} — skipping")
        return accepted

    print(f"[{culture}] walking categories (depth {depth}) ...")
    titles = [t for t in walk_file_titles(api, SEED_CATEGORIES[culture], depth)
              if t not in done_titles and not _TITLE_BLOCK.search(t)]
    print(f"[{culture}] {len(titles)} candidate files; need {cap - accepted} more")

    skipped = {"license": 0, "format": 0, "size": 0, "dup": 0, "error": 0}
    for i in range(0, len(titles), BATCH):
        if accepted >= cap:
            break
        for info in api.imageinfo(titles[i:i + BATCH]):
            if accepted >= cap:
                break
            meta = {k: clean_html(v.get("value", "")) for k, v in
                    info.get("extmetadata", {}).items()
                    if k in {"LicenseShortName", "LicenseUrl", "Artist",
                             "ImageDescription", "Credit"}}
            if not license_ok(meta.get("LicenseShortName", "")):
                skipped["license"] += 1
                continue
            if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
                skipped["format"] += 1
                continue
            if min(info.get("width", 0), info.get("height", 0)) < MIN_DIM:
                skipped["size"] += 1
                continue

            url = info.get("thumburl") or info["url"]
            ext = Path(url.split("?")[0]).suffix.lower() or ".jpg"
            counter += 1
            dest = img_dir / f"{culture}_c{counter:04d}{ext}"
            try:
                api.download(url, dest)
            except requests.RequestException as exc:
                print(f"  [warn] download failed: {info['title']}: {exc}")
                skipped["error"] += 1
                counter -= 1
                continue

            # sha1 + dHash against CICIL data and against this scrape.
            col = cicil.match(dest) or local.match(dest)
            h = dhash(dest)
            if col or h is None:
                dest.unlink()
                skipped["dup" if col else "error"] += 1
                counter -= 1
                if col:
                    print(f"  [dedup] {info['title']} == "
                          f"{col.match.split}/{col.match.lang}/{col.match.id} ({col.kind})")
                continue
            sha = sha1_file(dest)
            ref = ImageRef(dest, culture, "commons", dest.name)
            local._by_sha[sha] = ref
            local._hashes.append((h, ref))

            row = {
                "culture": culture,
                "local_file": str(dest.relative_to(out_root)),
                "commons_title": info["title"],
                "page_url": info.get("descriptionurl", ""),
                "license": meta.get("LicenseShortName", ""),
                "license_url": meta.get("LicenseUrl", ""),
                "artist": meta.get("Artist", ""),
                "credit": meta.get("Credit", ""),
                "description": meta.get("ImageDescription", ""),
                "width": info.get("width", ""),
                "height": info.get("height", ""),
                "sha1": sha,
                "dhash": f"{h:016x}",
            }
            writer.writerow(row)
            prov_file.flush()
            provenance.append(row)
            accepted += 1
            if accepted % 25 == 0:
                print(f"  [{culture}] {accepted}/{cap}")

    print(f"[{culture}] done: {accepted}/{cap} accepted; skipped {skipped}")
    return accepted


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape Commons images per culture.")
    ap.add_argument("--cultures", nargs="+", default=list(SEED_CATEGORIES),
                    choices=list(SEED_CATEGORIES))
    ap.add_argument("--per-culture", type=int, default=300)
    ap.add_argument("--depth", type=int, default=2, help="subcategory BFS depth")
    ap.add_argument("--out", type=Path,
                    default=config.ROOT / "data" / "external" / "commons")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "provenance.csv"
    provenance = load_provenance(csv_path)
    print(f"Building CICIL dedup index (pilot+dev+test) ...")
    cicil = cicil_index()

    api = Commons()
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS)
        if new_file:
            writer.writeheader()
        totals = {}
        for culture in args.cultures:
            totals[culture] = scrape_culture(
                api, culture, args.per_culture, args.depth, args.out,
                cicil, provenance, writer, f)

    print("\n=== scrape summary ===")
    for culture, n in totals.items():
        print(f"  {culture:<10} {n:>4} images")
    print(f"provenance -> {csv_path}")


if __name__ == "__main__":
    main()
