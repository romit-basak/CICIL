# Stage 2 Handoff — retrieval-arm ablation, missing retrieval banks, NLLB scaffold

Evergreen reference material for whoever works on Stage 2 retrieval next (building the
missing corpora, running the retrieval-arm ablation, extending the NLLB scaffold).
Unlike `STAGE1_HANDOFF.md` (a dated, directional changelog), this doc isn't tied to a
specific date — update it in place as things change.

## 1. The retrieval-arm ablation (`--query-arm`)

`retrieval.build_query_from_record(record, query_arm=...)` takes:
- `"cultural"` — force the cultural-annotation query (join Stage 1's cultural_annotations
  values), so retrieval targets cultural relevance over surface lexical similarity.
- `"text"` — force the plain-Spanish query, regardless of annotations. The vanilla-
  retrieval control arm: same k, same index, same prompt template — only the query
  changes.
- `"auto"` (default) — legacy behavior (cultural if present, else text). Couples the
  query strategy to Stage 1's prompt mode, which is exactly the confound the `cultural`/
  `text` arms exist to break apart. **Leaving `--query-arm` at the default silently
  reproduces today's plain filename with no tag at all** — `run_ablations.py`'s Table 2
  and its k-ablation tables only look for `..._culturalquery_...` / `..._textquery_...`
  files, so an `auto` run won't show up there. Pass `--query-arm cultural` or
  `--query-arm text` explicitly for anything you want scored by the new ablation tables.

CLI:
```bash
uv run python -m src.stage2.translate --lang all --mode cultural-vqa --k 5 --query-arm cultural
uv run python -m src.stage2.translate --lang all --mode cultural-vqa --k 5 --query-arm text

# Or the whole k x query-arm grid in one command:
uv run python -m src.stage2.run_sweep --dry-run   # validate wiring, no API calls
uv run python -m src.stage2.run_sweep             # the real thing, once banks land
```

`run_ablations.py`'s **Table 2** is the headline result: cultural-query vs. text-query
ChrF++ at k=5, mode held fixed at cultural-vqa.

### How `--query-arm` coexists with `--backend`

`translate.py` also has `--backend` (which Stage 1 model produced the input JSONL —
`ollama`, `smolvlm`, `smolvlm-devonly`, `smolvlm-noctx`, etc.), independent of
`--query-arm`. The predictions filename (`translate.pred_filename`, the single source of
truth) composes both:

| backend | query_arm | Filename |
|---|---|---|
| `ollama` (default) | `auto` (default) | `{lang}_{mode}_k{k}_predictions.txt` — today's original, unchanged contract |
| non-default | `auto` | `{lang}_{mode}_{backend}_k{k}_predictions.txt` |
| `ollama` | non-default | `{lang}_{mode}_{query_arm}query_k{k}_predictions.txt` |
| non-default | non-default | `{lang}_{mode}_{backend}_{query_arm}query_k{k}_predictions.txt` |

`run_sweep.py --backend <name>` sweeps the whole retrieval-arm grid against one Stage 1
backend at a time (defaults to `ollama`; pass `--backend smolvlm` once the distilled
adapter's outputs cover all 5 languages, not just Wixárika).

## 2. Building the missing retrieval banks

`build_index.CORPORA` currently has exactly one entry (`wixarika`, the 20-pair pilot
placeholder). The other four languages translate zero-shot until real corpora are added.
Confirmed sources (verified real — repo/package existence checked directly):

| Language | Source | Size | How to get it |
|---|---|---|---|
| Bribri | `github.com/AmericasNLP/americasnlp2021`, `data/bribri-spanish/` | 7,506 pairs | `git clone`, direct download |
| Guaraní | `AmericasNLP/americasnlp2023` train data + MultiScript30k (Driggers-Ellis et al. 2025, ~30k **synthetic**, NLLB-generated) | ~53k | `git clone` + cite the synthetic-data caveat |
| Wixárika (real bank) | Same 2021 repo, `data/wixarika-spanish/`, or `github.com/pywirrarika/wixarikacorpora` (Mager et al. 2018) | ~8,966 pairs | `git clone` |
| Nahuatl | `pip install py-elotl` → `elotl.corpus.load('axolotl')` | ~12–16k pairs | pip package, no manual download |
| Yucatec Maya | **Not found.** Not in AmericasNLP 2021's language list; neither related paper below cites a retrieval source for it either. | — | Needs its own dedicated search — flag as a known open gap, don't assume it'll turn up. |

Once those land: reformat to `{"spanish": ..., "target": ...}` pairs, add 4 lines to
`build_index.CORPORA`, run `build_index.py`, then `run_sweep.py` — the "one command"
moment `run_sweep.py`/`run_ablations.py` were built for.

**On the two related papers below — read them directly, don't cite this summary:**
Two real, highly relevant sources for this exact shared task were found while preparing
this handoff — `arxiv.org/abs/2605.20626` (a submission using a near-identical
architecture: Qwen2.5-VL → Spanish → Gemini 2.5 Flash retrieval-augmented translation,
which **won** the competition) and `github.com/rmaacario/americasnlp2026-usp` (a
competing submission fine-tuning NLLB-200 across all 5 languages — directly relevant to
§3 below). Both are worth reading in full before writing up the retrieval-bank section —
but an earlier characterization of "their key finding" (retrieval helps Guaraní but hurts
Yucatec Maya) was checked directly against both sources and **does not match what either
one actually reports**. What they actually say, independently verified:
- The winning submission's finding: retrieval is highly language-dependent, beneficial
  mainly for large, in-domain corpora — it doesn't test Yucatec Maya at all. It also
  reports synthetic data augmentation drove ~28 ChrF++ of their Guaraní gain — worth
  comparing against our own much smaller (~1 point) Commons-augmentation result.
- The NLLB submission's finding: their base (non-fine-tuned) NLLB model scored *higher*
  on automatic ChrF++ than their fine-tuned version (19.49 vs. 17.57), yet the fine-tuned
  version won more human-eval votes — a direct precedent for our own proxy-vs-end-to-end
  disagreement (see `STAGE1_HANDOFF.md`).

Re-derive whatever gets cited in the paper from the sources themselves.

## 3. NLLB-200 fine-tune scaffold (`nllb_finetune_guarani.py`)

Untested scaffold (needs the real Guaraní parallel data from §2, reformatted the same way
as the retrieval bank). Core config verified directly against the real tokenizer in this
repo's environment: `grn_Latn` (Guaraní) and `spa_Latn` (Spanish) are both valid
FLORES-200 codes.

**If extended to Bribri or Maya:** NLLB-200's tokenizer does **not** error on an
unrecognized FLORES-200 code — confirmed empirically that `bzd_Latn` (Bribri) and
`yua_Latn` (Yucatec Maya) aren't in this model's vocabulary and silently resolve to the
`<unk>` token (id 3) — **not** `<s>`/BOS (id 0), a detail worth getting right if you cite
this anywhere. A silent `<unk>` as `forced_bos_token_id` makes `generate()` emit ordinary
(often English) text with no error. `nllb_finetune_guarani.py` now has an
`_assert_lang_in_vocab()` guard that raises before any real work starts if this happens —
extend to new languages by adding a new special token and retraining, not by assuming the
FLORES-200 code just works.

## 4. Misc

- `data/americasnlp2026/data/pilot/wixarika.jsonl` is the correct, already-present path
  for the Wixárika pilot data in this repo's layout (not `data/pilot/wixarika.jsonl` —
  don't create that directory).
