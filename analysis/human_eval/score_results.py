"""Un-blind and aggregate human-eval results.

Reads every results/*.csv (exported from human_eval.html), joins with
sample_key.csv (the ONLY place the A/B -> arm mapping is used), and prints:

  * per-arm (generic vs cultural) means for the three rubric dimensions
  * preference rates (cultural wins / generic wins / ties)
  * per-category cultural-accuracy breakdown (feeds RQ3)
  * with >= 2 annotators: quadratically-weighted Cohen's kappa per dimension
    (ordinal 0-2 scores) and plain % agreement for preference

    uv run python -m analysis.human_eval.score_results
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
KEY_FILE = HERE / "sample_key.csv"

DIMS = ["cultural_accuracy", "faithfulness", "fluency"]


def load_key(suffix: str = "") -> dict[str, dict]:
    with (HERE / f"sample_key{suffix}.csv").open(encoding="utf-8") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def load_results(suffix: str = "") -> list[dict]:
    """Round-suffixed results only: round-2 exports end in {suffix}.csv, and
    round-1 files (no suffix) must not be scored against a round-2 key."""
    rows = []
    for path in sorted(RESULTS_DIR.glob(f"*{suffix}.csv")):
        if not suffix and path.stem.endswith("_round2"):
            continue
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("sample_id"):
                    rows.append(r)
    return rows


def arm_scores(rows: list[dict], key: dict[str, dict]):
    """Yield (annotator, sample_id, arm, dim, score) with slots resolved to arms."""
    for r in rows:
        k = key[r["sample_id"]]
        for slot in ("A", "B"):
            arm = k[f"slot_{slot}_arm"]  # "generic" | "cultural"
            for dim in DIMS:
                v = r.get(f"{slot}_{dim}", "")
                if v != "" and v is not None:
                    yield r["annotator"], r["sample_id"], arm, dim, int(v)


def weighted_kappa(a: list[int], b: list[int], n_cat: int = 3) -> float:
    """Quadratically-weighted Cohen's kappa for ordinal categories 0..n_cat-1."""
    obs = [[0.0] * n_cat for _ in range(n_cat)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    n = len(a)
    if n == 0:
        return float("nan")
    row = [sum(obs[i]) for i in range(n_cat)]
    col = [sum(obs[i][j] for i in range(n_cat)) for j in range(n_cat)]
    w = [[(i - j) ** 2 / (n_cat - 1) ** 2 for j in range(n_cat)] for i in range(n_cat)]
    num = sum(w[i][j] * obs[i][j] for i in range(n_cat) for j in range(n_cat))
    den = sum(w[i][j] * row[i] * col[j] / n for i in range(n_cat) for j in range(n_cat))
    return float("nan") if den == 0 else 1.0 - num / den


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Score human-eval results.")
    ap.add_argument("--suffix", default="",
                    help="round suffix (e.g. _round2): matching key + results files")
    args = ap.parse_args()
    key = load_key(args.suffix)
    rows = load_results(args.suffix)
    if not rows:
        raise SystemExit(f"No results CSVs found in {RESULTS_DIR}/ -- export from "
                         f"human_eval.html and move the file there.")
    annotators = sorted({r["annotator"] for r in rows})
    print(f"Loaded {len(rows)} item-rows from {len(annotators)} annotator(s): "
          f"{', '.join(annotators)}\n")

    # Arm names vary by round (round 1: generic/cultural; round 2/3: whatever
    # backend tags build_sample.py --arms used) -- read them from the key file
    # rather than hardcoding, or every round but the first silently prints NaN.
    arm_names = sorted({k["slot_A_arm"] for k in key.values()}
                       | {k["slot_B_arm"] for k in key.values()})
    if len(arm_names) != 2:
        print(f"WARNING: expected 2 arms in {key and 'sample_key'+args.suffix+'.csv'!r}, "
              f"found {arm_names}")
    arm1, arm2 = (arm_names + ["?", "?"])[:2]

    # ---- per-arm dimension means -------------------------------------------
    sums: dict = defaultdict(lambda: [0, 0])  # (arm, dim) -> [total, n]
    for _, _, arm, dim, v in arm_scores(rows, key):
        sums[(arm, dim)][0] += v
        sums[(arm, dim)][1] += 1
    print("== Per-arm means (0-2) ==")
    print(f"{'dimension':<22} {arm1:>16} {arm2:>16} {'delta':>7}")
    for dim in DIMS:
        g = sums[(arm1, dim)]
        c = sums[(arm2, dim)]
        gm = g[0] / g[1] if g[1] else float("nan")
        cm = c[0] / c[1] if c[1] else float("nan")
        print(f"{dim:<22} {gm:>16.2f} {cm:>16.2f} {cm - gm:>+7.2f}")

    # ---- preference ---------------------------------------------------------
    pref = defaultdict(int)
    for r in rows:
        p = r.get("preference_A_B_tie", "")
        if p in ("A", "B"):
            pref[key[r["sample_id"]][f"slot_{p}_arm"]] += 1
        elif p == "tie":
            pref["tie"] += 1
    total = sum(pref.values())
    print("\n== Preference ==")
    for k_ in (arm1, arm2, "tie"):
        n = pref[k_]
        print(f"{k_:<20} {n:>3}  ({n / total:.0%})" if total else f"{k_:<20} 0")

    # ---- per-category cultural accuracy (RQ3) ------------------------------
    cat_sums: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for annot, sid, arm, dim, v in arm_scores(rows, key):
        if dim != "cultural_accuracy":
            continue
        cats = key[sid]["categories_present"]
        for cat in [c.strip() for c in cats.split(";")] if cats else []:
            cat_sums[cat][arm][0] += v
            cat_sums[cat][arm][1] += 1
    print("\n== Cultural accuracy by category (RQ3) ==")
    print(f"{'category':<22} {arm1:>16} {arm2:>16}")
    for cat in sorted(cat_sums):
        g, c = cat_sums[cat][arm1], cat_sums[cat][arm2]
        gm = g[0] / g[1] if g[1] else float("nan")
        cm = c[0] / c[1] if c[1] else float("nan")
        print(f"{cat:<22} {gm:>16.2f} {cm:>16.2f}")

    # ---- CBIR retrieval relevance (scored once per item, not per-arm) ------
    cbir_vals = [int(r["cbir_relevance"]) for r in rows
                if r.get("cbir_relevance", "") != ""]
    if cbir_vals:
        dist = Counter(cbir_vals)
        print("\n== CBIR retrieval relevance (0=unrelated, 1=partial, 2=related) ==")
        print(f"mean = {sum(cbir_vals) / len(cbir_vals):.2f}  (n={len(cbir_vals)})")
        for v in (0, 1, 2):
            print(f"  {v}: {dist.get(v, 0):>3}  ({dist.get(v, 0) / len(cbir_vals):.0%})")

    # ---- inter-annotator agreement -----------------------------------------
    if len(annotators) >= 2:
        print("\n== Inter-annotator agreement ==")
        by_annot: dict = defaultdict(dict)
        for r in rows:
            by_annot[r["annotator"]][r["sample_id"]] = r
        for a1, a2 in combinations(annotators, 2):
            shared = sorted(set(by_annot[a1]) & set(by_annot[a2]))
            print(f"\n{a1} vs {a2} ({len(shared)} shared items):")
            for dim in DIMS:
                xs, ys = [], []
                for sid in shared:
                    for slot in ("A", "B"):
                        v1 = by_annot[a1][sid].get(f"{slot}_{dim}", "")
                        v2 = by_annot[a2][sid].get(f"{slot}_{dim}", "")
                        if v1 != "" and v2 != "":
                            xs.append(int(v1))
                            ys.append(int(v2))
                print(f"  {dim:<20} weighted kappa = {weighted_kappa(xs, ys):.2f}  (n={len(xs)})")
            cx, cy = [], []
            for sid in shared:
                v1 = by_annot[a1][sid].get("cbir_relevance", "")
                v2 = by_annot[a2][sid].get("cbir_relevance", "")
                if v1 != "" and v2 != "":
                    cx.append(int(v1)); cy.append(int(v2))
            if cx:
                print(f"  {'cbir_relevance':<20} weighted kappa = {weighted_kappa(cx, cy):.2f}  (n={len(cx)})")
            agree = sum(
                by_annot[a1][sid].get("preference_A_B_tie") ==
                by_annot[a2][sid].get("preference_A_B_tie")
                for sid in shared)
            if shared:
                print(f"  {'preference':<20} agreement = {agree}/{len(shared)} "
                      f"({agree / len(shared):.0%})")
    else:
        print("\n(Single annotator so far -- kappa needs a second results CSV.)")


if __name__ == "__main__":
    main()
