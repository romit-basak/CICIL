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
Annotation is **English-assisted** (no team member reads Spanish — see RUBRIC.md's
protocol section and citations) and happens in a local, zero-backend HTML tool.

```bash
python -m analysis.human_eval.build_sample        # (already run) 5 langs x 3 images x 2 arms
uv run python -m analysis.human_eval.translate_english   # (already run) ES→EN via Vertex, once
python -m analysis.human_eval.build_interface     # -> human_eval.html
# each annotator: open human_eval.html, score, Export CSV -> results/
python -m analysis.human_eval.score_results       # un-blind + aggregate + weighted κ
```

- `human_eval/RUBRIC.md` — rubric anchors + protocol (blind A/B, double annotation,
  κ), the English-pivot rationale/caveats, and the target-language scope decision.
- `human_eval/human_eval.html` — annotation interface (image + English/Spanish/target
  panels + rubric form; autosaves to localStorage; exports results CSV). Requires the
  dataset at `data/americasnlp2026/` for images; no server.
- `human_eval/sample_key.csv` — A/B → arm un-blinding map (analysis only; never shown
  to annotators, never embedded in the HTML).
- **Target-language captions are display-only by scope decision** — no native/heritage
  speakers were available; `sample_target.csv` is ready for future recruited
  annotators (see RUBRIC.md).

Sampling is stratified by cultural category (reuses the RQ3 labels) and blinded
(arm hidden, order randomized), deterministic under `--seed`.

## Known caveats (read before reporting numbers)

- **No `data/` here.** Per-language and per-category ChrF++ against gold targets need
  the dataset mounted at `data/americasnlp2026/`. Until then, tasks 1 & 3 are fully
  runnable and task 2 runs its labeling/distribution half; the scoring half switches
  on automatically once `data/` is present.
- **Cultural annotation version.** The dev `*_cultural-vqa_ollama.jsonl` in `outputs/`
  are the **v2 (concise)** annotations (verified ~25 words/answer) — the same generation
  the paper's reported cultural *scores* use. Labels and scores therefore already come
  from one generation; no regeneration is needed. (The `cultural_annotations` are in any
  case prompt-invariant between v1/v2 — only the synthesized Spanish description changed —
  so category *presence* would be identical regardless.)
- **Category buckets overlap.** Each caption counts toward *every* category present in its
  image, so per-category mean ChrF++ columns are not independent partitions; a low bucket
  can't be cleanly attributed to one category. Report the heatmap as indicative, not causal.
- **Target-language human eval needs speakers.** The team can annotate the Spanish
  sheet now; the target sheet is for recruited native/heritage annotators.
