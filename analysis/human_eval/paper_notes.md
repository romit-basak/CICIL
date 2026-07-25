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
loops** (unique-token ratio 6–26%; e.g. "t+kame" × 32). These are the same
predictions behind the reported Wixárika ~9 / Bribri ~5 ChrF++: character-overlap
metrics award nontrivial scores to completely broken output. (Known issue — see
STAGE1_HANDOFF "greedy decoding degenerated" — but the human-eval sample makes it
concrete and citable. The target-language fluency dimension would score these 0;
that evaluation is future work, below.)

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
