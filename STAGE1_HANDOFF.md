# Stage 1 → Stage 2 Handoff (Romit → Mehek)

Stage 1 produces the Spanish intermediate descriptions; Stage 2 translates them to
the target language and scores them. This note is the interface between the two.

## 🔎 2026-07-26 — RAG pilot: Wikipedia retrieval into Stage 1 (Guaraní + Wixárika)

**What it is.** Stage 1's root problem (established by the human eval + the 7%
culture-term statistic) is that no cultural knowledge source exists at inference —
the 2B student can't memorize the encyclopedia, and even the 7B teacher names a
culture in only 5/50 (Guaraní) / 1/50 (Wixárika) outputs. This pilot bolts a
retrieval bank onto Stage 1: an **image channel** (SigLIP CBIR over ~500
license-filtered Commons images per culture, harvested systematically from
Wikipedia hub articles — `scripts/harvest_wikipedia.py` + `scrape_commons.py
--extra-seeds`) injected as context into every VQA question, and a **text channel**
(MiniLM retrieval over Wikipedia lead extracts, queried with the image's own VQA
answers) injected into the synthesis step with calibrated-hedging instructions
("posiblemente" for uncertain matches; never name a concept without visual
support). Module: `src/stage1/rag_context.py`; new backend tags `smolvlm-rag` /
`ollama-rag`; per-image retrievals recorded in the output JSONL
(`cbir_context`, `text_rag_snippets`) for auditability.

**Results (dev, cultural-vqa, k=5, temp-0.7 Stage 2 decoding everywhere):**

| lang | arm | ChrF++ | culture-term | hedged | degen |
|---|---|---|---|---|---|
| guarani | smolvlm (no RAG, re-run) | 19.35 | 7/50 | 13/50 | 0/50 |
| guarani | **smolvlm-rag** | 19.10 | **40/50** | 5/50 | 0/50 |
| guarani | ollama-rag (teacher) | 20.26 | 20/50 | 27/50 | 0/50 |
| wixarika | smolvlm (no RAG, re-run) | 13.18 | 2/50 | 3/50 | 15/50 |
| wixarika | **smolvlm-rag** | 12.85 | **20/50** | 0/50 | 9/50 |
| wixarika | ollama-rag (teacher) | 14.06 | 17/50 | 23/50 | 10/50 |

(Reference: bare teacher without RAG names a culture in 5/50 / 1/50 — so retrieval,
not model scale, is the binding constraint on cultural naming.)

**Reading:**
- **Culture-term rate is the headline**: 14%→80% (Guaraní) and 4%→40% (Wixárika)
  for the same 2B student. ChrF++ stays flat — the established metric blindness:
  a single reference can't reward correct naming it doesn't contain.
- **Audit wins**: hch_021 (bare hills, scored "no cultural content" by every prior
  arm) now reads *"paisaje natural en Wirikuta, San Luis Potosí"* (student) /
  *"posiblemente relacionado con la cultura wixárika, que considera Wirikuta uno de
  los cinco lugares más sagrados"* (teacher). grn_025 (teacher): *"celebración
  cultural del chamamé en Argentina con... typói"* — correct festival, country, and
  garment.
- **The hedging-capability gap**: the teacher follows the calibration instruction
  (hedges 27/50, 23/50); the 2B student hedges *less* than its own baseline (5/50,
  0/50) — it converts retrieved concepts into confident assertions, sometimes
  wrong (grn_025 student: "mujer paraguaya... Carrozas del Ñandutí" — right
  artifact family, wrong country; grn_019: "piel de jaguar... La Recova, Brasil" —
  retrieval-induced fabrication). Calibrated uncertainty appears to be a
  capability, not a prompt.
- **Scope/confounds (state in the paper)**: 2-language pilot; the RAG synthesis
  prompt differs from the no-RAG v2 prompt (it must permit hedging), so prompt and
  context change together in the RAG arms; Wixárika degeneration improves
  (15→9/50) but the arms differ upstream, so don't attribute that to RAG alone.

Nothing here changes Mehek's sweep — the RAG tags are extra backends, and the
five-language sweep below remains the paper's main table.

---

## 🚀 2026-07-25 — Banks landed: run the sweep now (nothing left to build)

**Framing up front: your part has no construction work left.** The Phase2.zip merge
integrated all your Stage 2 code (`--query-arm`, `run_sweep.py`, the new
`run_ablations.py` tables); the retrieval banks for **all 5 languages** are committed
in `Dataset/` and wired into `build_index.py` (including a brand-new Yucatec Maya bank
— YUA-ES-CCC, CC-BY-4.0, 14,332 pairs; provenance in `DATA_LICENSES.md` and
`STAGE2_HANDOFF.md` §2). Your entire remaining job is a one-time GCP setup plus
running four commands in order and forwarding the tables. Total API cost ≈ $2 of
Gemini Flash calls.

### One-time GCP/Vertex setup (before the real sweep)

1. Claim your GCP education grant (redeem the course coupon link — the credits attach
   to a billing account), create a project, and link that billing account to it in the
   console.
2. Enable the Vertex AI API on the project: `gcloud services enable
   aiplatform.googleapis.com` (or via console → Vertex AI → Enable).
3. Install the gcloud CLI, then authenticate:
   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project <YOUR-PROJECT-ID>
   ```
4. **Mandatory, not optional:** `export GOOGLE_CLOUD_PROJECT=<YOUR-PROJECT-ID>` (add it
   to your shell profile or the repo `.env`). If unset, `translate.py` falls back to
   Romit's project id (`cicil-501318`) — you don't have IAM access to it, so every call
   will fail with a `403 PERMISSION_DENIED` naming a project you don't recognize. The
   fix for that confusing error is this export, set up front.
5. **Why Vertex and not AI Studio:** the education grant covers Vertex only, and the AI
   Studio free tier trains on submitted data — incompatible with the CC BY-NC CICIL
   data. This is a license requirement, not a preference.

### ⚠️ Decoding config changed 2026-07-25 — pull before running the sweep

`translate.py` now samples at `temperature=0.7` with a pinned seed (was greedy /
temp 0). Reason: greedy decoding degenerated into repetition loops on 64% of
Bribri and 86% of Wixárika dev captions; an A/B showed frequency penalties don't
fix it but temperature does (+1.7/+5.0 ChrF++, degeneration cut to ~30%), with no
regression on Guaraní (+0.64). Full table in `analysis/human_eval/paper_notes.md`.
Your sweep inherits this automatically — just make sure you've pulled. Don't
re-run the prelim bare-filename predictions; they stay temp-0 as the prelim
record.

### The commands (in order)

```bash
git pull
uv run python -m src.stage2.build_index        # ~10-30 min, one-time, builds indices/ locally
uv run python -m src.stage2.run_sweep --dry-run  # wiring check, no auth/API needed — can run before GCP setup
uv run python -m src.stage2.run_sweep          # the real thing: 45 runs ≈ 2,250 Gemini calls ≈ 2.5-5 h
uv run python -m src.stage2.run_ablations      # scores everything, prints Table 2 + k-ablation tables
```

Then send Table 2 + the k-ablation tables to Tisha and Romit.

### What not to worry about

- Your sweep only writes `_culturalquery_`/`_textquery_`-tagged prediction files. The
  prelim bare-filename predictions (and Table 1's numbers) are untouched — they remain
  the prelim record. Do **not** re-run bare `--query-arm auto` runs over them.
- You are **not blocked on Romit** for any of the above. His distilled-adapter
  (`--backend smolvlm`) rows are **already done** (table below, committed 2026-07-25)
  and don't touch any file your sweep produces. **Stage 1 is complete** — all 5
  languages × both modes exist under both backends (ollama + distilled smolvlm).
- **Human-eval tooling is ready (2026-07-25)** — Tisha is unblocked: annotation runs
  in a local HTML tool with English pivot translations (no Spanish needed, no GCP
  needed for annotators — just the repo + dataset checked out). Workflow and the
  target-language scope decision: `analysis/human_eval/RUBRIC.md`.

### ⚠️ First real-bank result (2026-07-25) — read before interpreting your sweep

The two distilled Wixárika arms were re-run against the real 9,940-pair bank as the
first test of the new indices:

| Arm | Placeholder bank (20 in-domain pilot pairs) | Real bank (9,940 pairs) |
|---|---|---|
| distill_full (`smolvlm`) | 9.17 | **8.22** |
| distill_noctx (`smolvlm-noctx`) | 10.34 | **8.23** |

Two takeaways. **(1) The context-ablation "gap" evaporated under a real bank** — the
two adapters are now indistinguishable (Δ 0.01), confirming the earlier +1.17 was a
placeholder-bank artifact, not a real effect. **(2) Both scores *dropped* when the bank
got 500× bigger.** Best hypothesis: domain mismatch — the old placeholder was 20
*in-domain* CICIL pilot captions, while `Dataset/wixarika.jsonl` is Mager et al.'s
corpus (largely narrative/fairy-tale register: Cinderella, spinning wheels), so k=5 now
retrieves fluent-but-irrelevant examples. This matches the winning 2026 submission's
finding that retrieval helps mainly with large *in-domain* corpora. Repetition loops
are also visible in several outputs (the known low-resource degeneration).

**Implication for your sweep:** if per-language results look flat or negative, check
the retrieved examples' register before concluding retrieval doesn't work — bank
domain match, not bank size, may be the binding variable. Worth an explicit
paragraph in the paper either way. (A quick follow-up worth considering if time
allows: append the 20 pilot pairs to each bank so in-domain examples can win the
similarity search when they're genuinely closest.)

### Distilled-adapter rows — all 5 languages, real banks (2026-07-25, complete)

The distilled SmolVLM (`distill_full`) Stage 1 outputs now exist for **all 5
languages × both modes** (`outputs/{lang}_dev_{mode}_smolvlm.jsonl`, 50 records each,
id-aligned with the ollama files), translated through Stage 2 with the **real** banks
(k=5, query-arm auto):

| Language | Generic | Cultural-VQA | Δ (cultural − generic) |
|---|---|---|---|
| Guaraní | 18.64 | 18.78 | +0.14 |
| Bribri | 5.01 | 5.09 | +0.08 |
| Yucatec Maya | 19.80 | 19.97 | +0.17 |
| Wixárika | 9.82 | 8.22 | **−1.60** |
| Nahuatl | 15.29 | 16.01 | +0.72 |

**Do not compare these against the prelim Table 1 ollama numbers directly** — those
were run zero-shot (no banks) / 20-pair placeholder, so backend AND bank changed at
once. The clean ollama-vs-distilled comparison exists only after your sweep produces
ollama rows under the same real banks (your `generic+text` arm is the matching
control). Within this table (same bank, same backend), cultural-VQA is +0.1–0.7 on
four languages (mostly within noise; Nahuatl again the largest) and −1.60 on Wixárika
— consistent with the domain-mismatch problem above hitting the culturally-keyed
retrieval hardest.

---

## ✅ Stage 1 update — distillation + context ablation (2026-07-19)

> **⚠️ Superseded (2026-07-25, re-run complete):** every end-to-end number in this
> section was measured against the 20-pair Wixárika placeholder bank. The real-bank
> re-run is done and recorded in the 2026-07-25 section above: the context-ablation
> gap was a placeholder artifact (arms tied at 8.22/8.23), and both arms scored lower
> under the bigger-but-out-of-domain real bank. Treat the gold-20 proxy numbers here
> as final and the end-to-end numbers as historical.

**Why:** the pilot LoRA fine-tune (19 gold Spanish pairs) came back flat —
17.55 ± 6.52 leave-one-out vs. 16.72 ± 3.60 base, i.e. no real signal. The
binding constraint was data, not hyperparameters, so Stage 1 pivoted to
**deployment-aligned knowledge distillation**: Qwen2.5-VL-7B (the same model
used as the generic-VLM baseline) generates silver Spanish targets on many more
images, and SmolVLM-2B is trained to imitate it — using the *exact* deployment
prompts (4 cultural-category questions + synthesis + generic), so the LoRA
adapter stays usable by the real pipeline rather than a proxy task. Gold
pilot captions are held out purely for eval, never trained on.

**What was built:**
- Wikimedia Commons scrape (license-filtered: PD/CC0/CC BY/CC BY-SA only) of
  ~1,328 additional culturally-relevant images across the 5 languages, sha1+dHash
  contamination-guarded against the gold pilot and dev/test images.
- `src/stage1/distill_data.py`: builds (image, prompt, target) triples matching
  deployment prompts exactly — 9,468 triples over 1,578 images (250 dev + 1,328
  Commons), 6 tasks/image.
- A hardened LoRA trainer (`finetune_smolvlm.py --distill`): step-level
  checkpointing + resume (survives spot preemptions mid-epoch), gradient
  checkpointing (required for the ~2k-token synthesis prompts on a 16GB T4).

**Headline result — full distill vs. baseline vs. teacher (gold-20 Spanish proxy):**

| Model | Generic | Cultural-VQA |
|---|---|---|
| SmolVLM-2B, off-the-shelf | 16.72 ± 3.60 | 14.09 ± 4.17 |
| **SmolVLM-2B, distilled (final)** | **20.33 ± 3.96** | **19.24 ± 2.23** |
| Qwen2.5-VL-7B teacher (proxy) | 21.0 | — |

Distillation closes nearly all of the gap to the 7B teacher at 2B parameters —
a real, reportable Stage 1 contribution.

**Important caveat — proxy and end-to-end disagree:** the same final adapter's
end-to-end Wixárika ChrF++ through Stage 2 is **9.17** — essentially tied with
the un-distilled teacher pipeline (9.19), and *below* an earlier, smaller
dev-only distill run (9.79), despite scoring higher on the Spanish proxy.
Checked for a generation bug (repetition, empty outputs, truncation) — none
found. Best read: **Spanish-proxy gains don't reliably transfer through Stage
2** on this single 50-image, single-reference metric. State this as a
limitation, not a second headline win.

**Context-injection ablation (does feeding Commons descriptions to the teacher
help?):** ran a controlled second arm — identical pipeline, but the teacher
sees *only* the image during Commons silver-captioning (no Wikimedia text).

| Metric | With teacher-context | No context | Δ |
|---|---|---|---|
| Gold-20 generic | 20.33 ± 3.96 | 20.49 ± 2.97 | +0.16 (noise) |
| Gold-20 cultural-VQA | 19.24 ± 2.23 | 19.69 ± 3.85 | +0.45 (noise) |
| End-to-end Wixárika (k=5) | 9.17 | 10.34 | +1.17 (real) |

**No measured benefit from context injection** — proxy deltas sit inside the
noise band, and end-to-end actually favors dropping it. This is a negative
result worth stating plainly rather than oversold: the enrichment idea added
real pipeline complexity (a full second Commons scrape + train + eval cycle)
without a demonstrated payoff.

**Caveat that bounds *all* of the end-to-end numbers above:** Wixárika's FAISS
index is still the **20-pair pilot placeholder** built 2026-07-03
(`indices/wixarika_pairs.jsonl`, unchanged since) — the real Mager et al. 2018
Wixárika-Spanish corpus hasn't landed. With k=5 over only 20 candidates,
retrieval pulls ~25% of the entire bank on every query, so the few-shot
context Gemini sees barely changes between a better and worse Stage 1
description. **We cannot yet distinguish "Stage 1 gains don't survive Stage
2" from "Stage 2's retrieval isn't built out enough to show them."** Do not
report the flat/negative end-to-end deltas above as a settled verdict on
distillation or the context ablation — they're a provisional read under a
placeholder retrieval bank. The other four languages have no bank at all
(zero-shot), so this is the best-case condition we currently have, not a
special Wixárika weakness.

**Still open:** only Wixárika has been re-run end-to-end with the new adapter.
Deploying the winning (with-context, final) adapter across all 5 languages'
dev sets and re-running Stage 2 for a full per-language table is the next
Stage 1 → Stage 2 handoff item — but re-running end-to-end evals once a real
retrieval bank lands should be considered higher-priority than either of those,
since it may change the ablation conclusions above.

---

## ✅ Stage 2 RESULTS — ran end-to-end 2026-07-03 (covering for Mehek)

Both arms were translated (Gemini 2.5 Flash **via Vertex AI**, ADC auth — the
Education grant covers Vertex but *not* the AI Studio paid tier) and scored with
`src/stage1/evaluate.py`. Stage 2 code now lives in **`src/stage2/`** (relocated
from `Mehek/`): `build_index → translate → run_ablations`; reproduce with
`uv run python -m src.stage2.run_ablations`.

**See `STAGE2_HANDOFF.md`** for the retrieval-arm ablation (`--query-arm`),
the missing-retrieval-bank corpus sources, and the NLLB-200 stretch scaffold.

**End-to-end dev ChrF++ (k=5):**

| Language | Official | Generic | Cultural-VQA | Δ |
|---|---|---|---|---|
| Guaraní | 20.82 | 21.26 | 20.70 | −0.56 |
| Bribri | 7.57 | 4.91 | 4.41 | −0.50 |
| Yucatec Maya | — | 19.25 | 19.56 | +0.30 |
| Wixárika | 17.77 | 8.97 | 9.19 | +0.22 |
| Nahuatl | 11.53 | 13.96 | 15.52 | **+1.56** |
| **Mean Δ** | | | | **+0.20** |

**Honest read (RQ1):** cultural-VQA shows **no consistent end-to-end gain** — four
of five languages are within ±0.6 ChrF++ (noise for n=50, indistinguishable);
only Nahuatl moves clearly (+1.56). Sanity checks pass (generic Guaraní 21.26 ≥
official 20.82; generic Nahuatl 13.96 ≥ 11.53).

**Caveats that bound these numbers:**
- Retrieval bank is real **only for Wixárika** (20 pilot pairs); the other four
  ran **zero-shot** (banks not yet integrated) → the k-ablation is a single point
  (k=5), retrieval contribution untested for 4/5 langs.
- Greedy decoding degenerated into repetition loops on the lowest-resource langs
  (Bribri ~5 both arms; Nahuatl *generic* looped); bounded with an output cap.
- **Nahuatl's +1.56 is confounded** — cultural-VQA partly wins by avoiding a
  degenerate generic baseline, not by demonstrably better grounding. Human eval +
  category breakdown (RQ3) are needed to disentangle.

Deliverable paper compiles at `acl2023/cicil_prelim.pdf` (integrates lit review +
these results). Predictions in `predictions/*.txt`.

---


## What Stage 1 produced

Two conditions, per dev language (5 languages × 50 images = **250 records each**),
generated with **Qwen2.5-VL-7B via Ollama**. Languages: guarani, bribri, maya,
wixarika, nahuatl.

| Condition | File | Role |
|---|---|---|
| Generic baseline | `outputs/<lang>_dev_generic_ollama.jsonl` | RQ1 **control** (no cultural prompting) |
| Cultural-VQA | `outputs/<lang>_dev_cultural-vqa_ollama.jsonl` | RQ1 **treatment** (+ cultural annotations) |

## Record schema (one JSON object per line)

```json
{
  "id": "grn_001",
  "filename": "grn_001.jpg",
  "language": "Guaraní",
  "iso_lang": "grn",
  "mode": "generic",
  "backend": "ollama",
  "generated_spanish": "<Spanish description — this is your Stage 2 SOURCE text>",
  "cultural_annotations": {"ceremony": "...", "material_culture": "...",
                            "landscape": "...", "kinship": "..."}
}
```
`cultural_annotations` is `{}` for the generic files; populated for cultural-VQA.

## What I need from Stage 2

1. **Translate `generated_spanish` → target language** with Gemini, for **both** conditions:
   - *Generic:* translate the Spanish (this is the baseline arm).
   - *Cultural-VQA:* use `cultural_annotations` as the retrieval keys (the Stage 1↔2 coupling), then translate.
2. **Score with the existing scorer — please don't rebuild it.** `src/stage1/evaluate.py`
   reproduces the official `baseline/eval.py` exactly (verified: 20.82 on Guaraní):
   ```bash
   uv run python -m src.stage1.evaluate --lang <lang> --translations <your_preds>.txt
   ```
3. **Report per-language ChrF++** for generic vs cultural-VQA, against the official baseline:

   | Guaraní | Wixárika | Nahuatl | Bribri | Maya |
   |---|---|---|---|---|
   | 20.82 | 17.77 | 11.53 | 7.57 | — (no MT baseline) |

   The **generic-vs-cultural delta is RQ1** — that's the headline result.

## ⚠️ Alignment contract (important)

`evaluate.py` aligns predictions to references **by line order**, and my JSONL files are
already in **dev-JSONL order** (same order as `data/americasnlp2026/data/dev/<lang>/<lang>.jsonl`).
So:
- Keep your predictions in that same order.
- Emit them as **plain text, one target caption per line, 50 lines per language**.
- Prefer matching by `id` instead? Tell Romit — he'll add id-based alignment to `evaluate.py`.

## ⚠️ Version note — cultural-VQA v2 (validated on the pilot proxy)

The synthesis prompt was tightened to **v2** (one ≤30-word sentence, no filler/hedging).
On the Wixárika pilot (n=20, scored against gold Spanish — the only split with Spanish
references), this flipped Stage 1 from a negative to a positive result:

| Stage 1 config (Wixárika pilot, n=20) | ChrF++ |
|---|---|
| Generic baseline | 20.68 |
| Cultural-VQA **v1** (verbose, ~105 tok) | 16.2 (Δ −4.8) |
| Cultural-VQA **v2** (concise, ~23 tok) | **21.69 (Δ +1.02)** |

**Status of the dev files you'll score:**
- **Generic** dev files — final, ready now.
- **Cultural-VQA** dev files in the repo are still **v1**. The full **v2 regeneration of all
  five dev languages is planned overnight (~9 h)** and lands tomorrow morning. Score the
  **v2** files for the reported cultural result — do **not** score the v1 dev files.

So: start on the **generic** arm now; the **cultural-VQA v2** dev files arrive tomorrow AM.

## Your Stage 2 time estimates & the plan

You are **not blocked** on building the pipeline: the FAISS index (over Nandita's parallel
corpora), the Gemini call, and the scorer (`evaluate.py`, already done) are all independent of
my outputs — and you can dry-run the whole pipeline on the pilot gold Spanish.

Rough Stage 2 runtimes (API-bound, not compute-bound):
- One-time FAISS index build (sentence-transformers over the corpora): **~10–30 min**, reused after.
- Translation over 250 records: ~250 Gemini calls/condition at ~1–3 s ≈ **~15–30 min per condition**
  (up to ~40 min if you hit free-tier rate limits); cost is pennies on Flash.
- Scoring: seconds.

So each condition is **well under an hour** end-to-end once you have the files.

**Suggested plan:**
1. **Today:** build the index + run the **generic** arm (files are ready) → first end-to-end number.
2. **Overnight:** Romit's v2 cultural-VQA regen (~9 h).
3. **Tomorrow AM:** Romit pushes v2 files → you `git pull` → re-run the **cultural** arm (~30–60 min)
   → the generic-vs-cultural comparison (RQ1), before the Jul 3 deadline.

The tight link is tomorrow morning's handoff (regen finishes → push → pull → run), so having the
index + generic arm done today makes it a quick swap-in.

## Minimum for the Jul 3 prelim

Even *generic → target* with simple/no retrieval gives the team's first end-to-end number
and proves the pipeline. Cultural-VQA is the headline comparison if time allows.

## Getting the files

The prediction JSONLs are **versioned in the repo** under `outputs/` (`.gitignore` now tracks
the `*.jsonl` files and comparison CSVs; logs and adapters stay out). Just `git pull`.
The `data/` dataset (CC BY-NC 4.0) stays out of the repo.
