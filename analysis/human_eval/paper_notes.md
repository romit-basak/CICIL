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

**Round 2 update (2026-07-27, RAG arms, same image) — retrieval displaced a
grounding strategy that was already working.** CBIR retrieval here is genuinely
good (`cbir_relevance` = 2): the top neighbor ("Nanduti 3py") is a real ñandutí-
adorned typói dress, visually close to the actual garment (embroidered floral
medallions, wide open skirt, similar pose). But neither RAG-arm caption mentions
Corrientes, chamamé, or the festival poster at all (visible, legible text the
round-1 non-RAG arms both read correctly, fumbling only the year). Both instead
assert **"Itaguá/Itauguá, Paraguay"** — the retrieved neighbor's own location,
copied verbatim onto a different photograph. This is not a vague wrong-country
miss; it is a specific, checkable place name imported from retrieved-reference
metadata. Net effect of retrieval on this image: traded "right event, wrong
year" (poster-OCR grounding, still working pre-RAG) for "no event mentioned,
wrong country" (retrieval-metadata leakage) — retrieval did not just fail to
help, it discarded a strategy that already worked in favor of a wrong one that
was easier to copy.
- Scoring note: split this across dimensions rather than let one error swallow
  both. `cultural_accuracy` should credit ñandutí where a caption names it
  correctly (verified real, specific, present in the garment) independent of the
  location error; the wrong city is a `faithfulness` problem, not a
  cultural_accuracy one — the two failures are unrelated and shouldn't be
  conflated into a single low score.

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
- **Round 2 flip side (2026-07-27, RAG arms, same image):** smolvlm-rag's caption —
  *"A sleeping cat on a bench made of fur with a geometric design in shades of blue,
  red, and white, possibly in a Yucatecan Mayan domestic setting"* — properly hedges
  ("possibly") and names the correct culture, which under a naive reading of the
  cultural_accuracy rubric earns a 2 ("accurate, correct culturally specific content").
  But the claim is **unfalsifiable from the image**: there is no diagnostic content to
  verify it against (confirmed by both the gold caption and the round-1 case above).
  The model isn't grounding this in anything it sees — it's restating the culture the
  `culture-as-given` prompt handed it, hedged just enough to resemble calibrated
  uncertainty. Scored as 1 ("right gist but vague"), not 2, on the reasoning that
  "accurate" should require content verifiable against the image, not just correct
  restatement of metadata already known before the model looked at anything.
  **Why this matters beyond one image:** it's direct behavioral evidence for exactly
  why `analysis/rag_pilot.py` reports culture-NAME rate and CONCEPT rate separately —
  name rate is "partly prompt echo" under culture-as-given. Maya's concept rate sits
  at 0 across every arm including the 7B teacher (§RAG results below) while its name
  rate jumps to 32-39/50; this image is the mechanism made visible: the model doesn't
  fail to find Maya content, it substitutes the given label for it. The rubric's 0/1/2
  scale has no clean slot for "correctly hedged but ungrounded" as distinct from
  "genuinely grounded" — a second concrete rubric gap alongside the one above.

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
- **hch_021 (bare hills) — ⚠️ CORRECTED 2026-07-28, do not cite as a clean win.**
  Previously "sin evidencia de elementos culturales específicos" in every arm.
  Student-RAG: *"paisaje natural en Wirikuta, San Luis Potosí"* — originally
  framed here as "the sacred-geography category works through retrieval." It
  doesn't, at least not verifiably. Direct comparison of the actual top-1 CBIR
  neighbor image (0.67, "posible" — never "fuerte") against hch_021: the
  neighbor is a ceremonial gathering on a reflective salt-flat/shallow water
  surface with a crowd in ceremonial dress; hch_021 is an empty rocky canyon
  with a river, no people. Different scene types entirely, sharing only coarse
  SigLIP-similarity features (earthy tones, rocky/hilly terrain, water
  reflection). The "Wirikuta" claim is lifted verbatim from the neighbor's own
  caption text, not independently recognized — same mechanism as the confirmed-
  wrong grn_025 Itauguá error below, except unfalsifiable here (Wirikuta is a
  large region, not a checkable specific place, and dev has no geolocation
  field). All 3 of hch_021's top-3 neighbors mention Wirikuta/San Luis Potosí,
  but none are canyon/river photos either (a cactus close-up, a town panorama,
  the ceremony above) — of the wixárika bank's 450 images, only 8 (2%) mention
  Wirikuta at all, so this isn't pure base-rate noise, but it does mean the
  "Wirikuta-tagged" pool is small and none of its members actually resemble this
  query image. Net: we cannot confirm or deny the claim is true; we CAN confirm
  the retrieval channel provides no evidence for it. The smolvlm-rag caption
  states it as flat fact, no hedge — the least justified version of any arm.
  **Reframe for the paper**: not "sacred geography grounding works," but "broad,
  hard-to-falsify regional claims pass every check we built — regex, ChrF++,
  even casual human read — in a way narrow checkable claims (a named town)
  don't," itself a metric-validity limitation worth its own sentence.
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
student). hch_021 illustrates the trade mechanically — prompt-RAG student names
Wirikuta flatly (unhedged), ragdistill hedges the culture but drops the site
name — but per the 2026-07-28 correction above, neither version of this
particular claim is independently verified. (Checked bribri's bzd_042/"Cahuita
tree" as a substitute illustration before using it — it's ALSO a visual
mismatch, see the CBIR reliability finding below; do not cite that one either
without the same caveat. No individual image in this pair has been confirmed
clean; treat the aggregate rate collapse as the reliable part of this finding,
not any single caption.) Framing for the paper: **you can distill the caution,
but specificity and calibration trade off at 2B — only the 7B teacher holds
both.**
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

### CBIR retrieval-quality spot checks (human eval round 2, 2026-07-27)

Manually inspecting the top-1 retrieved Commons neighbor against the eval image
(new `cbir_relevance` rating, `analysis/human_eval/build_cbir_refs.py`):
- **grn_019 (ñandutí): genuine retrieval failure.** Top neighbor ("Artesanía
  Textil en Atyra," score 0.66) is a market stall selling striped blankets,
  leather bags, and baskets — no ñandutí lace visible. Matched at the *category*
  level (textile craft ↔ textile craft) on generic visual similarity (color
  variety, stall composition), not the specific weave. SigLIP's known limit:
  same-type-of-scene ≠ same artifact.
- **grn_023 (Jesuit ruins): unverifiable, not confirmable either way.** Retrieved
  a different ruin photo (dry vs. vegetated — could be season, could be a
  different site entirely). Neither annotator nor the interface can confirm
  same-site identity from pixels alone. Worth stating plainly: CBIR's "posible"/
  "fuerte" bands are a similarity heuristic, not a same-referent guarantee, and
  this is a limitation of human verification too, not just the retrieval — a
  domain expert or geotagged metadata would be needed to settle it.
- **grn_025 (chamamé dancer): good retrieval, badly used.** See the case study
  above for the full analysis — genuinely close visual match (`cbir_relevance` =
  2), but the caption imports the retrieved neighbor's *location* wholesale
  rather than reading the correct one off the photo's own visible poster. The
  spectrum across these three: bad retrieval (grn_019), unverifiable retrieval
  (grn_023), good retrieval + bad application (grn_025) — three distinct failure
  modes, only one of which is actually "the retrieval was wrong."
- **hch_021 (bare hills, "Wirikuta"): genuine retrieval failure, our former
  flagship example.** Full correction in the case-study section above. Top
  neighbor (0.67, "posible") is a ceremonial-gathering photo on a reflective
  salt flat; hch_021 is an empty canyon with a river. No shared content beyond
  coarse tone/terrain-type similarity. The "Wirikuta" claim in every RAG
  caption for this image is copied from the neighbor's caption text, not
  independently grounded — do not cite this as evidence retrieval grounds
  sacred geography.
- **bzd_042 (wooden bowl/nutcracker artifact, "Cahuita tree"): also a
  mismatch.** Checked as a candidate replacement example for the paragraph
  above — retrieved neighbor is a black-and-white forest-path photo with tree
  roots, no artifacts. Shares only monochrome tone and vegetation/dirt-ground
  texture with the actual image (a close-up of a carved wooden implement in a
  cleared yard). Another coarse-tone match, not a content match.

**Aggregate reliability check on the "posible" confidence band (0.55–0.80,
which covers the large majority of all neighbors — "fuerte" ≥0.80 is only
~5% of dev neighbors per the original score-distribution check): of 4
manually spot-checked "posible"-tier retrievals across this session
(ñandutí, hch_021, bzd_042, grn_025), 3 are visual mismatches and only 1
(grn_025's dress) is a genuine match.** n=4 is too small to quote a
percentage in the paper, but it is large enough to warrant real caution:
**do not treat any individual "posible"-tier concept-rate hit as verified
without checking the actual retrieved image.** The aggregate concept-rate
*trends* (e.g. the ragdistill collapse, wixárika/bribri's higher rates than
maya/nahuatl) may still be real patterns even if individual instances are
unverified — a rate computed over 50 images is more robust than any single
caption — but the paper should state this as an explicit limitation on what
"concept rate" is actually measuring: partly genuine grounding, partly
coarse-similarity-triggered label copying that happens to name something
plausible. Future work: raise the "fuerte" threshold, add a stricter
same-object check (e.g. a second, class-aware similarity model), or simply
report this uncertainty rather than resolve it before the deadline.

Both are concrete illustrations for a "retrieval quality" paragraph: even where
the eventual caption is inert (the market-stall neighbor was never named in the
generated text), a similarity-only retrieval channel will sometimes hand the
model a same-category-wrong-object reference, and there is no free way to catch
this other than eyes-on spot-checking.

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
