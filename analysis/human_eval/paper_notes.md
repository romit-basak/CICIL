# Human-eval findings & paper notes (qualitative analysis + limitations material)

Working notes from the first annotation pass (2026-07-25, annotator: Romit, ollama
arm, 15 images). For Tisha's qualitative-analysis and limitations sections. Every
claim below is verifiable from repo artifacts; gold captions quoted from
`data/americasnlp2026/data/dev/` (not redistributed here beyond fair-use quotation).

## Case studies (use as the qualitative-analysis backbone)

### 1. S001 (grn_019) — the artifact both arms missed

Gold: *"Peteĩ **ñandutí** mba'emi ojejapóva sa'y hetávagui... ojejapóva Paraguáipe"*
— the reference names the specific artifact (ñandutí lace) and its origin.
- Generic arm: concrete visual description, no cultural content.
- Cultural arm: "traditional textile art in cultures that use fabrics and threads to
  create symbolic or decorative designs" — culturally *flavored* but unfalsifiable
  (nearly every culture uses symbolic textiles).
Neither names ñandutí. The cultural-VQA arm's failure mode is **unfalsifiable
hedging**, not wrong claims. Note also: ñandutí itself descends from Canary Islands
"sun lace" (Tenerife lace), so even expert recognition is contextual inference, not
unique visual identification — cultural significance ≠ visual uniqueness.

### 2. S003 (grn_025) — OCR-parasitic culture + fabricated specifics ChrF++ cannot see

Image: chamamé dancer at "Mundial de Chamamé **2012**" (year visible on poster).
- Both arms name the event by reading the poster — but fabricate the year (A: 2013,
  B: 2019). Two different wrong years from the same model under different prompts =
  confabulation, not misreading.
- Gold caption mentions **no event, no year, no Corrientes**: it describes the
  **ñandutí adornments on the dress** ("Jerokyhára ijao ñande ypykuére, oñembojapava
  ñandutípe..."). Same artifact as S001, missed again.
- **ChrF++ is structurally blind to the fabricated dates here**: the entire event
  clause is off-reference content, so the metric doesn't penalize the fabrication as
  a fabrication at all. Only human faithfulness scoring catches it.
- Pattern: when text is visible in the image, the models anchor on transcription
  (cheap specificity) instead of the visual cultural evidence (the hard skill the
  pipeline was meant to add). Observed in 2 of 15 sampled images.

### 3. yua_001 — the culturally-empty image the rubric can't score

Image: a cat in a mass-produced pet bed. Gold: *"Juntúul miis ku weenel"* — "A cat
is sleeping." The Maya-speaking reference authors themselves assert **zero cultural
content**. The image is in the dataset by provenance (a Maya household), not by
visible cultural content.
- **Rubric gap discovered:** cultural accuracy has no N/A. A caption that correctly
  declines to name any culture scores 0 ("culturally generic") — the same as failing
  to see a ceremony. Correct restraint is penalized. Workaround this round: flag such
  items in the notes field; exclude/report separately at analysis time. Fix for any
  future protocol version: add an explicit "no cultural content present" option.

### 4. S007–S009, S013, S015 — what ChrF++ actually rewarded

10 of 12 Wixárika/Bribri target captions in the sample are **degenerate repetition
loops** (unique-token ratio 6–26%; e.g. "t+kame" × 32). Full-corpus rates (smolvlm
arm, real banks): **Bribri 32/50 (64%), Wixárika 43/50 (86%)** of dev captions.
These are the predictions behind the reported Wixárika ~8–9 / Bribri ~5 ChrF++:
character-overlap metrics award nontrivial scores to completely broken output.
(The target-language fluency dimension would score these 0; that evaluation is
future work, below.)

**Decoding ablation (2026-07-25) — a two-part result worth its own paragraph:**
all arms cultural-vqa, k=5, smolvlm backend, real banks, otherwise-identical config.

| Arm | Bribri ChrF++ | Bribri degen | Wixárika ChrF++ | Wixárika degen |
|---|---|---|---|---|
| temperature 0 (greedy, prelim config) | 5.09 | 32/50 | 8.22 | 43/50 |
| + frequency_penalty 0.7 | 4.86 | 34/50 | 8.54 | 43/50 |
| temperature 0.7, fixed seed | **6.81** | **14/50** | **13.18** | **15/50** |

Healthy-language regression check: Guaraní 18.78 → 19.42 (+0.64, no degeneration
in either arm) — no cost to the languages that were already fine.

Reading: **frequency penalties don't break the loops** (they mutate into token
variants and multi-token phrase cycles per-token penalties barely touch), but
**sampling does** — greedy decoding is specifically pathological here (the argmax
path enters the repetition attractor and cannot leave; cf. Holtzman et al. 2019),
and stochastic decoding escapes it, recovering +1.7 (Bribri) to +5.0 (Wixárika)
ChrF++ and cutting degeneration by ~two-thirds. The remaining gap to the official
baselines is the real competence limitation; roughly a third of Wixárika's
apparent deficit was decoding artifact. Adopted into `translate.py`
(`temperature=0.7`, pinned seed) as of 2026-07-25 — all subsequent runs (including
the retrieval-arm sweep) use it; prelim temp-0 numbers remain the prelim record.
Caveats: single seed, n=50/language; the degeneration-rate drop is mechanical and
robust, exact ChrF++ deltas carry sampling variance.

## Quantitative context

- **Culture-naming is near-absent by design-consequence:** 18/250 (7.2%) of
  cultural-VQA dev outputs contain any culture/region term at all, and most hits are
  just "indígena". In the evaluated 15-image sample: 1 image (S014). Root cause
  verified in `src/stage1/vqa_prompts.py`: **Stage 1 is never given the culture
  label**, although the task supplies it at inference time (the dev metadata carries
  `culture:`). Hedging is the model's rational response to an under-determined
  question.
- Framing that unifies everything: **culture-from-pixels is an ill-posed inverse
  problem**; motifs are areal/global (radial lace = Tenerife family; cf. the
  swastika across Eurasia/Americas pre-1930s), so correct cultural attribution is
  Bayesian — visual evidence × provenance prior. The dataset supplies the prior; the
  current pipeline withholds it from Stage 1. Deliberate-sounding defense: "we never
  prompt the model to assert a culture it cannot see." Obvious future work: condition
  Stage 1 on the known culture and measure grounding gain vs. hallucination cost
  (the risk: ñandutí confabulated onto mass-produced bedsheets).

## Results — annotator 1 (Romit, 2026-07-25, all 15 items)

`results/human_eval_results_Romit.csv`; recomputed via
`uv run python -m analysis.human_eval.score_results`.

| Dimension (0–2) | Generic | Cultural-VQA | Δ |
|---|---|---|---|
| Cultural accuracy | **0.00** | **0.13** | +0.13 |
| Faithfulness | 1.73 | 1.53 | −0.20 |
| Fluency (content coherence) | 1.93 | 1.87 | −0.07 |

Preference: ties 9/15 (60%), cultural 4 (27%), generic 2 (13%).

**Reading (single annotator — κ pending a second pass):**
- **Cultural accuracy is at floor for BOTH arms.** The generic arm scored 0 on all
  15 images; the cultural arm earned exactly two 1s (S001, S006). The human eval
  independently confirms the corpus statistic (7% culture-term rate): whatever the
  cultural-VQA prompting adds, it is not *verifiable cultural specificity* in the
  final Spanish description.
- **The small cultural gain trades against faithfulness** (−0.20): consistent with
  the case studies — hedged cultural gestures and fabricated specifics (S003's
  invented years) cost accuracy points.
- **60% ties on preference**: for most images the two arms are not meaningfully
  different to a human reader — consistent with the ±noise ChrF++ deltas on 4 of 5
  languages. The metric and the human agree that the arms are close; the human adds
  *why* (both culturally mute, differing mainly in verbosity/anchoring).
- Honest paper line: RQ1's human-side answer so far is "cultural prompting produces
  marginally more cultural gesture, slightly lower faithfulness, and mostly
  indistinguishable captions." Needs the second annotator before reporting.

- Annotator 2: TBD — κ needs a second pass (any teammate; English pivot means no
  Spanish required; ~30–45 min).

## RAG pilot (2026-07-26) — retrieval closes the naming gap the weights can't

This is the direct follow-up to the "obvious future work" line above, run as a
2-language pilot (Guaraní + Wixárika). Arc sentence for the paper: **a 2B model
cannot store the encyclopedia in its weights (distillation showed that), and even
a 7B teacher names a culture in ≤10% of outputs unaided — so we supply the
encyclopedia at inference time via retrieval, with hedging by design.** Two
channels, both harvested systematically from Wikipedia (no hand-picked
categories): SigLIP CBIR over ~500 Commons images/culture (context for every VQA
question) and MiniLM retrieval over Wikipedia lead extracts queried with the
image's own VQA answers (context for synthesis only, with "posiblemente" hedging
instructions).

| lang | arm | ChrF++ | culture-term | hedged | degen |
|---|---|---|---|---|---|
| guarani | smolvlm no-RAG | 19.35 | 7/50 (14%) | 13/50 | 0/50 |
| guarani | smolvlm-rag | 19.10 | **40/50 (80%)** | 5/50 | 0/50 |
| guarani | ollama-rag (7B teacher) | 20.26 | 20/50 | 27/50 | 0/50 |
| wixarika | smolvlm no-RAG | 13.18 | 2/50 (4%) | 3/50 | 15/50 |
| wixarika | smolvlm-rag | 12.85 | **20/50 (40%)** | 0/50 | 9/50 |
| wixarika | ollama-rag (7B teacher) | 14.06 | 17/50 | 23/50 | 10/50 |

Bare 7B teacher without RAG: 5/50 and 1/50 culture-terms — **retrieval, not model
scale, is the binding constraint on cultural naming.**

**Case-study updates (same images as above):**
- **hch_021 (bare hills)**: previously "sin evidencia de elementos culturales
  específicos" in every arm. Student-RAG: *"paisaje natural en Wirikuta, San Luis
  Potosí"* (CBIR image channel found it — all top-3 neighbors were Wirikuta).
  Teacher-RAG hedges properly: *"posiblemente relacionado con la cultura wixárika,
  que considera Wirikuta como uno de los cinco lugares más sagrados."* The
  sacred-geography category — the hardest, most novel case — works through
  retrieval.
- **grn_025 (chamamé festival)**: teacher-RAG gets festival, country, and garment
  right (*"chamamé en Argentina con... typói"*). Student-RAG shows the failure
  mode: *"mujer paraguaya... Carrozas del Ñandutí"* — the Paraguay-centric Guaraní
  bank pulled the attribution to the wrong country, asserted without hedging.
- **grn_019 (ñandutí)**: still missed by name in all arms; student-RAG fabricates
  (*"piel de jaguar... La Recova, Brasil"* — a retrieval-induced error, worse than
  the baseline's vague honesty). RAG's cost side, use alongside the win.

**FINAL UPDATE 2026-07-27 — the distillation verdict (both directions at once):**
RAG-aware distillation (student retrained on 13,765 teacher triples whose prompts
include the retrieval context, byte-matching deployment) CLOSED the hedging
inversion — ragdistill hedge rates 23–34/50 across all 5 languages, teacher-like,
where prompting alone got 0–15/50. **Calibration is distillable.** But it
transferred the teacher's conservatism with it: ragdistill's concept rate
collapses to teacher levels (wixárika 24→1, bribri 25→5 vs the prompt-only RAG
student). hch_021 makes the trade visible: prompt-RAG student names Wirikuta
flatly (right, but uncalibrated); ragdistill hedges the culture properly but no
longer names the site. Framing for the paper: **you can distill the caution, but
specificity and calibration trade off at 2B — only the 7B teacher holds both.**
ChrF++ side: bribri student-RAG +1.19 (thinnest bank, biggest metric gain);
teacher-RAG +1.7/+1.9 on grn/hch; maya/nahuatl ≈ −1 where no concept grounded.

**UPDATE 2026-07-27 (all-culture extension, interim):** the finding below is now
confirmed on all 5 languages with matched v2 calibrated prompts (culture stated
as given; CBIR neighbors tagged «coincidencia fuerte/posible»; channel-ranked
hedging rules): teacher hedge rates 24–38/50 everywhere, student 0–15/50
everywhere (at/below its own no-RAG baseline). Prompting does not fix the
inversion. Also new: report culture-NAME rate and CONCEPT rate separately in the
final table — with culture-as-given, name mentions are partly prompt echo, while
concept naming (Wirikuta, cacao/Cahuita: wixárika 0→24/50, bribri 5→25/50) is
the real retrieval signal. Maya shows 0 concepts in every arm incl. teacher —
likely a dev-set property (daily-life imagery), RQ3 material.

**The hedging-capability finding (new, paper-worthy):** the calibration
instruction ("mark uncertain matches with posiblemente") is followed by the 7B
teacher (hedge rate 27/50, 23/50) and *inverted* by the 2B student, which hedges
less than its own no-RAG baseline (13→5, 3→0) — it converts retrieved concepts
into confident assertions. Calibrated uncertainty under retrieval appears to be a
capability, not a prompting fix. This reframes the earlier hedging observation:
the baseline student hedged because it lacked knowledge; given knowledge, it
over-commits.

**Confounds to state**: the RAG synthesis prompt necessarily permits uncertainty
expressions that the no-RAG v2 prompt bans (prompt and context change together);
ChrF++ is flat across arms — single-reference blindness to correct naming, which
is exactly why culture-term rate + audit are co-primary here; 2-language pilot
only; Wixárika degeneration improvement (15→9/50) is not attributable to RAG
alone.

## Draft limitations paragraphs (adapt freely)

> Our human evaluation of cultural accuracy is bounded by annotator expertise:
> annotators were neither speakers of the target languages nor members of the
> depicted cultures, so judgments capture whether a caption makes culturally
> specific claims, not whether those claims are verifiably correct — e.g., no
> annotator could be expected to recognize ñandutí lace, though the reference
> caption names it. More fundamentally, an image's cultural identity is supplied by
> dataset context rather than always being visually recoverable (visual motifs recur
> across unrelated traditions), so models face a choice between unfalsifiable
> cultural hedging and culturally silent description; our rubric scores the former
> above the latter, which annotator preferences did not always endorse, and offers
> no "no cultural content" option for images whose own reference captions are
> culturally neutral. We report per-dimension inter-annotator agreement so readers
> can weigh the cultural-accuracy scores accordingly.

> We deliberately did not human-evaluate the target-language captions: no native or
> heritage speakers of the five languages were available within the project window,
> and machine-pivoting an extremely low-resource language for evaluation would be
> circular. The blinded target-language sheet is constructed and ready
> (`sample_target.csv`); end-to-end human evaluation with community annotators is
> future work. Spanish-side annotation was conducted through Gemini ES→EN pivot
> translations — reliable for this high-resource direction (Hendy et al. 2023,
> arXiv:2302.09210; Kocmi et al. 2023, WMT23 findings) in exactly the way ES→target
> is not, which is this project's premise — with the caveats that Spanish-specific
> wording and grammar are outside what the pivot can expose.
