# Stage 1 → Stage 2 Handoff (Romit → Mehek)

Stage 1 produces the Spanish intermediate descriptions; Stage 2 translates them to
the target language and scores them. This note is the interface between the two.

## ✅ Stage 2 RESULTS — ran end-to-end 2026-07-03 (covering for Mehek)

Both arms were translated (Gemini 2.5 Flash **via Vertex AI**, ADC auth — the
Education grant covers Vertex but *not* the AI Studio paid tier) and scored with
`src/stage1/evaluate.py`. Stage 2 code now lives in **`src/stage2/`** (relocated
from `Mehek/`): `build_index → translate → run_ablations`; reproduce with
`uv run python -m src.stage2.run_ablations`.

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
