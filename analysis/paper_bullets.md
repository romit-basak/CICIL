# Final paper — main bullet points (drafting skeleton, 2026-07-26)

8 pages excl. references, ACL template (**check template before writing — wrong
template = zero**). Every number below has a pointer to its source file so nothing
gets cited from memory. `⏳ TBD` = lands after tonight's runs; see the checklist at
the bottom.

---

## Abstract (write last)

- Two-stage pipeline for cultural image captioning in 4 indigenous languages (+
  Nahuatl banks): culturally-prompted VLM → Spanish intermediate → retrieval-augmented
  LLM translation.
- Headline claims to plug in: culture-term rate jumps with Stage 1 retrieval
  (14%→80% Guaraní, 4%→40% Wixárika, pilot; ⏳ 5-language table tonight); retrieval,
  not model scale, is the binding constraint (bare 7B teacher: 5/50, 1/50).
- Honest frame: ChrF++ is largely insensitive to cultural correctness — we show
  *why* (single reference, metric blindness) and evaluate with term-rate + human eval.

## 1. Introduction

- Task: AmericasNLP 2026 CICIL; culturally situated images; ~50 dev/lang.
- Gap we attack: generic VLMs discard cultural information before translation begins.
- Core thesis (the arc sentence): **a 2B model cannot store the encyclopedia in its
  weights (distillation proved it), and even a 7B teacher names a culture in ≤10% of
  outputs unaided — so we supply the encyclopedia at inference time via retrieval,
  with hedging by design.**
- Culture-from-pixels is an **ill-posed inverse problem**: motifs are areal/global
  (radial lace = Tenerife family; the swastika pre-1930s), so attribution = visual
  evidence × provenance prior. The dataset supplies the prior (culture label);
  pipelines that withhold it from the vision stage are throwing away the posterior.
  (From `analysis/human_eval/paper_notes.md` — the ñandutí/bedsheet observation.)
- RQ1: does culturally-aware vision encoding beat generic VLM? RQ2: does benefit
  vary by resource level? RQ3: which cultural categories are hardest?

## 2. Related work (Romit's two; teammates add theirs)

- **CIC (2024)** — VQA-before-captioning; direct ancestor of our cultural-VQA module.
  We differ: their questions are culture-generic; we add retrieval + calibration.
- **CultureCLIP (Huang et al., 2025)** — synthetic contrastive fine-tuning for
  cultural awareness; informs why we fine-tune the decoder (LoRA) not the encoder.
- English-pivot human evaluation: Hendy et al. 2023 (arXiv:2302.09210), Kocmi et al.
  WMT23 (ES→EN reliability) — justifies the eval protocol (`analysis/human_eval/RUBRIC.md`).
- (Person D/team: retrieval-augmented MT, low-resource MT for Guaraní/Bribri, ChrF++.)

## 3. Method

### 3.1 Stage 1 — cultural VQA + distilled SmolVLM
- 4 cultural categories (ceremony, material culture, landscape, kinship/community);
  joint per-category questions → synthesis to one Spanish sentence (`src/stage1/vqa_prompts.py`).
- LoRA (r16/α32) on SmolVLM-2B decoder; SigLIP frozen. Distilled from Qwen2.5-VL-7B
  silver captions over ~1.6k license-filtered Commons images + dev teacher outputs
  (`src/stage1/distill_data.py`, 9,468 triples).
- Dev/test contamination guard: sha1 + dHash (Hamming ≤6) against all splits
  (`src/stage1/dedup.py`).

### 3.2 Stage 1 retrieval (the novel bit — lead with this)
- Two channels, harvested **systematically** from Wikipedia (hub articles → 1-hop
  links → two-rule filter; seed lists human-reviewed; `scripts/harvest_wikipedia.py`):
  - **Image channel (CBIR):** SigLIP embeddings over ~450–520 Commons images/culture;
    top-3 neighbors ≥0.55 cosine injected into every VQA prompt, tagged by confidence
    band: «coincidencia fuerte» ≥0.80 (rare by design: ~5% of dev neighbors),
    «coincidencia posible» otherwise (`src/stage1/rag_context.py`).
  - **Text channel:** MiniLM retrieval over Wikipedia lead extracts (543/150/⏳ per
    culture), queried with the image's own VQA answers, injected into synthesis only.
- **Calibration by construction** (v2 prompt): the culture is asserted as *given*
  (bank selection used the task's culture label — hedging the culture would hedge the
  one certain thing); strong image match → assert; possible match → «posiblemente»;
  text-only support → «podría ser». Verbal bands, not percentages (SigLIP cosine is
  not a probability; small models can't verbalize numeric confidence).
- **RAG-aware distillation** (⏳ tonight): teacher re-silver-captions the bank WITH
  deployment-style retrieval context (self-matches excluded from CBIR); student
  trains on prompts that byte-match inference. Tests whether the student's
  context-blindness (pilot finding) is fixable by training.

### 3.3 Stage 2 — culturally-indexed retrieval + Gemini
- FAISS over 5 real banks (Dataset/: guarani 15,494; nahuatl 16,119; maya 14,332
  [YUA-ES-CCC, CC-BY-4.0]; wixarika 9,940; bribri 8,297 pairs; licenses in
  `DATA_LICENSES.md`). k=5 few-shot into Gemini 2.5 Flash (Vertex, paid tier — the
  free tier trains on inputs, incompatible with CC BY-NC task data).
- Decoding: temperature 0.7 + fixed seed (ablation-backed, see 5.3).

## 4. Experimental setup
- ChrF++ via official script; per-language and per-category breakdowns.
- Human eval: English-pivot (nobody on the team reads Spanish — Stage 1 eval on
  ES→EN translations, cited reliability; **Stage 2 target-language eval is an
  explicit scope decision → future work with speakers**). 15 images, 3 dims, 0–2
  rubric. κ ⏳ pending second annotator (Tisha).
- Co-primary metrics for the RAG claims: culture-term rate (regex over Spanish
  outputs) + retrieval audits + spot checks, because ChrF++ is blind to naming
  (single reference).

## 5. Results

### 5.1 Main table (⏳ REBUILD TONIGHT: all arms at temp-0.7, 5 languages)
- Current distilled all-language rows (real banks, k=5) — `STAGE1_HANDOFF.md`:
  Guaraní 18.64/18.78, Bribri 5.01/5.09, Maya 19.80/19.97, Wixárika 9.82/8.22,
  Nahuatl 15.29/16.01 (generic/cultural-VQA). NOTE: pre-decoding-fix; the final
  table must use tonight's consistent temp-0.7 set (pilot re-runs already moved
  Guaraní cultural to 19.35, Wixárika to 13.18).
- Arms per language: smolvlm (no RAG) / smolvlm-rag / smolvlm-ragdistill ⏳ /
  vllm-rag (7B teacher upper bound) ⏳; plus Mehek's k∈{3,5,8} sweep ⏳.

### 5.2 The RAG result (5-language Stage-1 rates IN — 2026-07-27; ⏳ ChrF++ tonight)
- **Report the metric in two parts** (v2 prompt states the culture as given, so
  bare culture-name mentions are partly prompt echo — split them from concepts):
  - **Culture-NAME rate** (no-RAG student → RAG student → RAG teacher, /50):
    guaraní 3→37→37; wixárika 0→21→38; maya 1→32→38; nahuatl 1→18→29;
    bribri 0→34→37. Bare teacher without RAG: ≤10% everywhere.
  - **CONCEPT rate** (specific artifact/site, never in the prompt):
    wixárika 0→**24** (Wirikuta, peyote, Real de Catorce); bribri 5→**25**
    (cacao, Cahuita, Puerto Viejo); guaraní 1→5; maya 0→**0**; nahuatl 0→1.
    Concept grounding works where dev images show distinctive sites/artifacts;
    **maya's 0-across-all-arms (teacher included) is an RQ3 data point** — its
    dev set is daily-life imagery where no bank concept applies, not a retrieval
    failure. Verify against audits before writing.
- **Retrieval, not scale, is the binding constraint**: bare 7B teacher names a
  culture in ≤5/50; the 2B student WITH retrieval beats the 7B teacher WITHOUT.
- ChrF++ flat (19.35→19.10; 13.18→12.85 in the pilot) — metric blindness, argued
  not assumed (S003's fabricated year cost nothing; repetition loops scored +6).
- Case studies (`analysis/human_eval/paper_notes.md`):
  - hch_021 (bare hills): every prior arm "no cultural content" → "Wirikuta, San
    Luis Potosí" (student), "posiblemente... uno de los cinco lugares más sagrados"
    (teacher). Sacred geography works through retrieval.
  - grn_025 (chamamé): teacher gets festival/country/garment right; student-RAG
    misattributes to Paraguay (bank prior pulls attribution) — the cost side.
  - grn_019 (ñandutí): still missed by name in all arms; student-RAG fabricates
    ("piel de jaguar... Brasil"). Honest negative.
- **Hedging-inversion finding (now confirmed across all 5 langs, matched v2
  prompts):** teacher follows calibration everywhere (hedge 24–38/50); the 2B
  student inverts it everywhere (0–15/50, at/below its no-RAG baseline) — it
  converts retrieved concepts into confident assertions. Prompt engineering
  (culture-as-given + confidence bands) did NOT fix it, so it's not a pilot
  artifact or prompt bug. Calibrated uncertainty under retrieval looks like a
  capability, not a prompt. ⏳ smolvlm-ragdistill hedge rates = the decisive test:
  works → calibration is distillable; fails → emergent-with-scale, report as
  finding either way.

### 5.3 Decoding ablation (already final — `paper_notes.md`)
- Wixárika/Bribri degeneration (repetition loops): 86%→30% and 64%→28% with
  temp 0.7 + seed; ChrF++ +4.96 / +1.72; Guaraní +0.64 (no regression).
  frequency_penalty=0.7 alone: useless. Report the reversal honestly (temperature
  arm was added "for completeness" and overturned the initial conclusion).

### 5.4 Human eval (single annotator; ⏳ κ)
- Cultural accuracy at floor BOTH arms (0.00 generic / 0.13 cultural); small
  cultural gain trades against faithfulness (−0.20); 60% preference ties.
- Root cause (verified in code): Stage 1 never received the culture label → 7%
  culture-term rate corpus-wide → the RAG work is the direct fix (arc!).

### 5.5 Stage 2 findings (Person C/Mehek section, our data)
- Bank domain mismatch: real 9,940-pair Wixárika bank scored *worse* (8.22) than a
  20-pair in-domain placeholder — domain match beats scale for few-shot retrieval.
- Context ablation: teacher-context injection into silver captions — no effect
  (arms tied 8.22/8.23); the earlier apparent gap was an artifact.

## 6. Analysis / Discussion
- RQ1: cultural prompting alone ≈ no; + retrieval ≈ yes on cultural specificity,
  invisible to ChrF++. RQ2: resource gradient is stark — Bribri thin at *every*
  layer (7.5k pairs, 90 wiki extracts, 2 usable Commons seeds; see Limitations).
  RQ3: ⏳ per-category table (landscape was hardest pre-RAG; hch_021 suggests
  retrieval helps it most — check per-category term rates tonight).
- Proxy vs end-to-end disagreement (distillation: gold-20 win, end-to-end tie) —
  metric caution for the field.

## 7. Limitations
- ChrF++/single-reference blindness (fabrications, repetition, correct naming).
- Ill-posed culture attribution; hedging as *designed behavior*, not failure.
- Annotators: not speakers/members of the communities; English-pivot; Stage 2
  target-language eval deliberately out of scope → future work with speakers.
- Prompt+context confound in RAG arms (RAG prompt permits hedging the base bans).
- 2-language pilot for some ablations; Bribri bank thinness (see below).
- Wikipedia/Commons coverage is itself culturally skewed (Bribri: ~12–17k people,
  no state backing, no own-language Wikipedia — the resource gradient reproduces
  inside the retrieval source).

## 8. Ethics / Data statement
- All data public; per-source licenses + citations in `DATA_LICENSES.md` (YUA-ES-CCC
  CC-BY-4.0 + Zenodo DOI; Wikipedia CC BY-SA; Commons license-filtered at scrape
  time with per-file provenance; CICIL CC BY-NC honored via paid-tier Gemini).
- No community involvement in evaluation — stated, with the future-work commitment.

---

## ✅ FINAL RESULTS (2026-07-27) — table complete, plug into §5

Full table: `uv run python -m analysis.rag_pilot` (or STAGE1_HANDOFF.md 07-27
section). The four headline readings:
1. **Culture-naming fixed everywhere**: ≤3/50 → 18–39/50, all 5 languages.
2. **Concept grounding follows the dev set**: wixárika 0→24, bribri 5→25
   (Wirikuta, cacao/Cahuita); maya 0 in ALL arms incl. 7B teacher (daily-life
   imagery — the RQ3 answer: retrieval helps where images depict retrievable
   concepts).
3. **ChrF++**: teacher-RAG +1.7/+1.9 on grn/hch; **bribri student-RAG +1.2**
   (6.81→8.00 — the thinnest-resource language gains most, RQ2 answer inverts
   the naive expectation); maya/nahuatl RAG ≈ −1 (no concept payoff → retrieval
   noise). Report per-language, not averaged.
4. **Calibration IS distillable — but transfers with the teacher's conservatism**
   (the paper's second headline): ragdistill hedge 23–34/50 (teacher-like;
   prompting alone got 0–15) but concept rate collapses to teacher levels
   (wixárika 24→1, bribri 25→5). hch_021 contrast: prompt-RAG student names
   Wirikuta flatly; ragdistill hedges the culture but drops the site. A
   precision/recall trade on cultural specificity — specific-but-uncalibrated
   vs calibrated-but-conservative; only the 7B holds both.

Also note for §5.2: teacher grn_025 caption now reads the poster ("Mundial de
Chamamé 2019 en Corrientes, Argentina") — verify against the image before
quoting (the human eval found wrong years fabricated from posters before).

## ⏳ STILL PENDING (not blocked on our runs)
- **Mehek's sweep tables** (k ablation, query-arm; + smolvlm-rag sweep after pull).
- **κ** from Tisha's second annotation pass; until then single-annotator caveat.
- **Per-category (4 VQA categories) breakdown** — cheap script over the
  cultural_annotations fields if time permits (RQ3 depth).
- **Budget line**: ~$8 L4 (9h incl. debugging) + ~$5 Gemini total; ~$8 GCP credit
  remains. Wall-clock: harvest→final table ≈ 27h, one person + one L4.
