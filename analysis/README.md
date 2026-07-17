# `analysis/` — Data analysis & evaluation

Bucket-2 evaluation deliverables: the RQ1 comparison figure, the RQ3 per-category
harness, and the human-evaluation kit. Everything reads from the committed
`outputs/` and `predictions/` files; the parts that need the gold references degrade
gracefully because `data/` is gitignored (CC BY-NC).

Run from the repo root. System Python works for the figures and the sampler; the
RQ3 **scoring** path additionally needs `sacrebleu` (already a project dep — use
`uv run` on a machine with the venv, or `pip install sacrebleu`).

## 1. RQ1 comparison figure — `fig_rq1_scores.py`

Per-language ChrF++, official baseline vs generic vs cultural-VQA (the paper's
headline, as a figure — the prelim had only tables).

```bash
python -m analysis.fig_rq1_scores      # -> figures/rq1_chrf_comparison.{pdf,png}
python -m analysis.results_data        # print the numbers table
```

Numbers live in `results_data.py` (single source of truth, from the prelim paper /
`STAGE1_HANDOFF.md`). They are hard-coded because the end-to-end scores can't be
recomputed without `data/`; update that one file if a run changes and the figure
follows.

## 2. RQ3 per-category harness — `rq3_category.py`

"Which cultural categories are hardest to ground?" Labels each dev image by which
categories (ceremony / material culture / landscape / kinship) the Stage-1 VQA found
present, then reports ChrF++ per category.

```bash
python -m analysis.rq3_category --all              # labeling + presence figure (now)
python -m analysis.rq3_category --lang guarani     # one language
```

Outputs:
- `rq3_labels_<lang>.csv` — **auditable** per-image labels + the absence cues that
  drove each 0/1. Edit the cells and feed back with `--override <file>`.
- `rq3_category_summary.csv` — per-language × category counts (+ ChrF++ when scored).
- `figures/rq3_category_presence.png` — category-presence distribution (runs now).
- `figures/rq3_category_chrf_heatmap.png` — mean ChrF++ per category (**only when
  `data/` is mounted**; verified working via a synthetic-reference smoke test).

**The category labels are a heuristic first pass** (negation-cue detection over the
Spanish annotations) — deliberately a scaffold for Nandita (taxonomy owner) to audit
via the labels CSV / override, not a black box.

## 3. Human-evaluation kit — `human_eval/`

Measures what ChrF++ can't: cultural accuracy, image faithfulness, fluency.

```bash
python -m analysis.human_eval.build_sample     # 5 langs x 3 images x 2 arms = 30 captions/sheet
```

- `human_eval/RUBRIC.md` — the 3-point cultural-accuracy rubric (+ faithfulness,
  fluency, preference), anchors and protocol (blind A/B, double annotation, κ).
- `human_eval/sample_spanish.csv` — Stage-1 Spanish descriptions (team-annotatable now).
- `human_eval/sample_target.csv` — target-language captions (native/heritage speakers).
- `human_eval/sample_key.csv` — A/B → arm un-blinding map (analysis only; do not show
  annotators).

Sampling is stratified by cultural category (reuses the RQ3 labels) and blinded
(arm hidden, order randomized), deterministic under `--seed`.

## Known caveats (read before reporting numbers)

- **No `data/` here.** Per-language and per-category ChrF++ against gold targets need
  the dataset mounted at `data/americasnlp2026/`. Until then, tasks 1 & 3 are fully
  runnable and task 2 runs its labeling/distribution half; the scoring half switches
  on automatically once `data/` is present.
- **Cultural annotation version.** The dev `*_cultural-vqa_ollama.jsonl` in `outputs/`
  are the **v1 (verbose)** annotations; the paper's reported cultural *scores* use the
  **v2 (concise)** regeneration (see `STAGE1_HANDOFF.md`). RQ3 category *presence* is
  robust to this, but re-run the harness on the v2 files before quoting per-category
  ChrF++ so labels and scores come from the same generation.
- **Target-language human eval needs speakers.** The team can annotate the Spanish
  sheet now; the target sheet is for recruited native/heritage annotators.
