import json
import os
import matplotlib.pyplot as plt
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────
PILOT_DIR = "data/pilot"
DEV_DIR   = "data/dev"

# Map language code → (display name, jsonl path to try)
LANGUAGE_PATHS = {
    "Wixárika":     [f"{PILOT_DIR}/wixarika.jsonl"],
    "Guaraní":      [f"{DEV_DIR}/guarani/guarani.jsonl",   f"{DEV_DIR}/guarani.jsonl"],
    "Bribri":       [f"{DEV_DIR}/bribri/bribri.jsonl",     f"{DEV_DIR}/bribri.jsonl"],
    "Yucatec Maya": [f"{DEV_DIR}/maya/maya.jsonl",         f"{DEV_DIR}/maya.jsonl"],
}

# ── LOAD ─────────────────────────────────────────────────────────────────────
def load_jsonl(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return [json.loads(l) for l in f], p
    return None, None

data = {}
for lang, paths in LANGUAGE_PATHS.items():
    examples, found_path = load_jsonl(paths)
    if examples:
        data[lang] = examples
        print(f"✓ Loaded {len(examples):>3} examples for {lang}  ({found_path})")
    else:
        print(f"✗ Not found: {lang}  (tried: {paths})")

if not data:
    print("\nNo data loaded — check your folder structure.")
    exit()

# ── SAMPLE ENTRY ─────────────────────────────────────────────────────────────
print("\n── SAMPLE ENTRY ────────────────────────────────────────────────────")
first_lang = list(data.keys())[0]
print(json.dumps(data[first_lang][0], indent=2, ensure_ascii=False))

# ── TABLE 1: PILOT / DEV SET SIZES ───────────────────────────────────────────
print("\n── TABLE 1: LOADED SET SIZES ───────────────────────────────────────")
print(f"{'Language':<20} {'# Examples':>12}")
print("-" * 34)
for lang, examples in data.items():
    print(f"{lang:<20} {len(examples):>12}")

# ── TABLE 2: RETRIEVAL CORPUS SIZES ──────────────────────────────────────────
retrieval = {
    "Guaraní":      (53183, "AmericasNLP 2023 + MultiScript30k"),
    "Bribri":       (7506,  "AmericasNLP 2021"),
    "Wixárika":     (8967,  "AmericasNLP 2021"),
    "Yucatec Maya": (None,  "TBD"),
    "Nahuatl":      (None,  "Py-Elotl 2025"),
}
print("\n── TABLE 2: RETRIEVAL CORPUS SIZES (from papers) ───────────────────")
print(f"{'Language':<20} {'# Pairs':>12}   Source")
print("-" * 65)
for lang, (size, source) in retrieval.items():
    print(f"{lang:<20} {str(size) if size else 'unknown':>12}   {source}")

# ── HELPERS ──────────────────────────────────────────────────────────────────
def get_caption(ex, field="target"):
    """Try common field names for the indigenous caption."""
    for f in ["target_caption", "target", "caption", "indigenous_caption"]:
        if f in ex:
            return ex[f]
    return None

def get_spanish(ex):
    for f in ["spanish_caption", "spanish", "es_caption", "reference"]:
        if f in ex:
            return ex[f]
    return None

langs  = list(data.keys())
n      = len(langs)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── FIGURE 1: CAPTION LENGTH DISTRIBUTION ────────────────────────────────────
fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=(n > 1))
if n == 1:
    axes = [axes]
fig.suptitle("Figure 1: Indigenous Caption Length Distribution (# words)", fontsize=13)

for ax, lang, col in zip(axes, langs, colors):
    captions = [get_caption(ex) for ex in data[lang] if get_caption(ex)]
    lengths  = [len(c.split()) for c in captions]
    ax.hist(lengths, bins=12, color=col, edgecolor="white")
    ax.axvline(np.mean(lengths), color="red", linestyle="--", linewidth=1.2,
               label=f"mean={np.mean(lengths):.1f}")
    ax.set_title(lang, fontsize=11)
    ax.set_xlabel("# words")
    ax.legend(fontsize=8)

axes[0].set_ylabel("# captions")
plt.tight_layout()
plt.savefig("figure1_caption_lengths.png", dpi=150)
print("\nSaved → figure1_caption_lengths.png")

# ── FIGURE 2: RETRIEVAL CORPUS SIZE BAR CHART ────────────────────────────────
known_langs = [l for l, (s, _) in retrieval.items() if s]
known_sizes = [retrieval[l][0] for l in known_langs]

fig2, ax2 = plt.subplots(figsize=(8, 4))
bars = ax2.barh(known_langs, known_sizes, color=colors[:len(known_langs)])
ax2.set_xlabel("# Parallel Pairs")
ax2.set_title("Figure 2: Retrieval Corpus Size by Language")
for bar, val in zip(bars, known_sizes):
    ax2.text(bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
             f"{val:,}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("figure2_corpus_sizes.png", dpi=150)
print("Saved → figure2_corpus_sizes.png")

# ── FIGURE 3: SPANISH vs INDIGENOUS LENGTH SCATTER ───────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
plotted = False
for lang, col in zip(langs, colors):
    sp_lens, ig_lens = [], []
    for ex in data[lang]:
        sp = get_spanish(ex)
        ig = get_caption(ex)
        if sp and ig:
            sp_lens.append(len(sp.split()))
            ig_lens.append(len(ig.split()))
    if sp_lens:
        ax3.scatter(sp_lens, ig_lens, label=lang, color=col, alpha=0.7)
        plotted = True

if plotted:
    ax3.set_xlabel("Spanish caption length (# words)")
    ax3.set_ylabel("Indigenous caption length (# words)")
    ax3.set_title("Figure 3: Spanish vs. Indigenous Caption Lengths")
    ax3.legend()
    plt.tight_layout()
    plt.savefig("figure3_length_scatter.png", dpi=150)
    print("Saved → figure3_length_scatter.png")
else:
    print("Figure 3 skipped — no Spanish captions found.")

# ── BASIC STATS SUMMARY ───────────────────────────────────────────────────────
print("\n── CAPTION STATS SUMMARY ───────────────────────────────────────────")
print(f"{'Language':<20} {'# captions':>12} {'mean len':>10} {'min':>6} {'max':>6}")
print("-" * 58)
for lang in langs:
    captions = [get_caption(ex) for ex in data[lang] if get_caption(ex)]
    lengths  = [len(c.split()) for c in captions]
    print(f"{lang:<20} {len(captions):>12} {np.mean(lengths):>10.1f} "
          f"{min(lengths):>6} {max(lengths):>6}")

print("\n── DONE ────────────────────────────────────────────────────────────")