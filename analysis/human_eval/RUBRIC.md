# Human-Evaluation Rubric — Cultural Image Captioning (CICIL)

**Purpose.** ChrF++ is a character-overlap metric: it can reward a caption that
copies the source or repeats itself, and it cannot tell whether the *culturally
specific* content is right. This human evaluation exists to measure what the metric
cannot — whether a caption is **culturally accurate**, **faithful to the image**,
and **fluent** — and to test RQ1 (does cultural-VQA help?) and RQ3 (which cultural
categories are hardest?) with human judgement rather than ChrF++ alone.

Design follows the transparent, rubric-piloted protocol of THumB
(Kasai et al., 2022): fixed dimensions, explicit anchors, blind A/B presentation.

---

## What annotators see

Each row is one **image** with **two captions, labelled A and B**, in randomized
order. Annotators do **not** know which is the generic control and which is the
cultural-VQA arm (the mapping lives in `sample_key.csv`, used only at analysis
time). Annotators score A and B independently on every dimension, then record a
preference.

Annotation happens in **`human_eval.html`** (built by `build_interface.py`): the
image on top, and for each caption its English translation (judge this), the
original Spanish (Stage 1 output), and the target-language caption (Stage 2
output — display only, never scored).

### English-assisted annotation (protocol change, 2026-07-25)

No team member reads Spanish well enough to annotate directly, so annotators judge
each Spanish caption **via a Gemini ES→EN translation** displayed alongside it
(`translate_english.py`, run once, temperature 0). ES→EN is a high-resource
direction where LLM translation is state-of-the-art: Hendy et al. (2023,
arXiv:2302.09210) find GPT-class models "achieve very competitive translation
quality for high resource languages, while having limited capabilities for low
resource languages," and the WMT23 findings (Kocmi et al., 2023,
aclanthology.org/2023.wmt-1.1) ranked GPT-4 top across most high-resource
directions. The pivot therefore adds far less noise than the 0–2 dimensions it
supports — and note the same asymmetry is this project's whole premise: ES→EN is
reliable in exactly the way ES→Maya/Bribri/Wixárika is not.

**Honest caveats to state in the paper:** cultural-accuracy judgments of *Spanish
word choice* are limited by the pivot (a mistranslated cultural term could mask or
invent an error); and the **fluency dimension measures the content's coherence**
(repetition loops, truncation, incoherence survive translation), **not Spanish
grammar**, which is out of reach through a pivot.

### Scope decision: target-language evaluation is future work

The target-language sheet (`sample_target.csv`) is built, blinded identically, and
ready — but **unscored by design**: no native/heritage speakers of the five
languages were available within the project window, and machine-pivoting an
extremely-low-resource language for evaluation would be circular (judging MT output
through more MT of the same weak direction). This is a deliberate scope decision,
reported as such: end-to-end target-language human evaluation is future work with
recruited speakers, for whom the sheet is ready to hand over. The interface shows
the target captions for reference only.

---

## Dimensions (score each caption, A and B)

### 1. Cultural accuracy — **the primary dimension** (0–2)

How well the caption names the culturally specific content actually present in the
image (objects, dress, practices, setting), without inventing or mislabelling it.

| Score | Meaning | Example signal |
|---|---|---|
| **0 — wrong / absent** | Culturally generic *or* names the wrong culture. Misidentifies a culturally specific object, or omits obvious cultural content and describes only generic shapes/colors. | "two wooden cups on a striped cloth" for a *mate* set; or attributing an item to the wrong tradition. |
| **1 — partial** | Gets the general cultural gist but is vague, incomplete, or mixes one correct and one incorrect cultural detail. | "a traditional drink container" (right idea, not named); names *mate* but invents the material. |
| **2 — accurate** | Correctly identifies the culturally specific content, with the right terms and no cultural errors. | "a *mate* gourd and *bombilla* for drinking yerba mate." |

> **Wrong-culture rule:** confidently attributing content to the wrong culture is a
> **0**, even if fluent — a confident cultural error is worse than a vague-but-safe
> caption.

### 2. Image faithfulness / adequacy (0–2)

Independent of culture: does the caption describe **what is actually in the image**?

| Score | Meaning |
|---|---|
| **0** | Contradicts the image or hallucinates major content not present. |
| **1** | Mostly right but with a notable wrong or missing element. |
| **2** | Faithful to the salient content of the image. |

### 3. Fluency (0–2)

Is the text well-formed in its language?

| Score | Meaning |
|---|---|
| **0** | Broken: repetition loops, truncation, or ungrammatical throughout. |
| **1** | Understandable but awkward, with grammar or word-choice errors. |
| **2** | Natural and grammatical. |

> Fluency catches the **greedy-decoding degeneration** the paper flags (repetition
> loops on low-resource languages) — a caption stuck in a loop is a fluency **0**
> regardless of its ChrF++.

### 4. Preference (A / B / tie)

After scoring both, which caption is the better overall description of this image?
Choose **A**, **B**, or **tie**. This gives a metric-free RQ1 signal (does the
cultural arm win head-to-head?).

### 5. Notes (free text)

Anything the scores miss — a specific cultural error, a hallucinated object, the
category the caption got right or wrong (ceremony / material culture / landscape /
kinship). Category notes feed RQ3.

---

## Protocol notes (for a clean, reportable result)

- **Blind.** Never reveal the A/B → arm mapping to annotators.
- **Independent then compare.** Score A and B on all dimensions *before* choosing a
  preference, so the preference doesn't bias the dimension scores.
- **Double annotation + agreement.** Have ≥2 annotators score each item and report
  inter-annotator agreement (Cohen's κ for the ordinal scores, % agreement for
  preference). Adjudicate disagreements of ≥2 points.
- **Anchor first.** Before scoring the real sample, both annotators score the same
  3 warm-up items and reconcile, so the anchors are shared.
- **Reporting.** Per arm: mean cultural accuracy / faithfulness / fluency, the A-vs-B
  preference rate, and cultural accuracy broken down by category (RQ3). Compare
  against the ChrF++ deltas to show where the metric and humans disagree.

## Files in this kit

| File | Role |
|---|---|
| `RUBRIC.md` | this document — give it to every annotator |
| `build_sample.py` | draws the stratified, blinded sample and writes the sheets |
| `sample_spanish.csv` | the sampled Stage-1 Spanish captions (source of truth for the interface) |
| `sample_english.csv` | Gemini ES→EN translations of the sample (`translate_english.py`) |
| `sample_target.csv` | target-language captions — display-only now; ready for future speaker annotators |
| `sample_key.csv` | A/B → arm un-blinding map (analysis only — never shown to annotators, never embedded in the HTML) |
| `translate_english.py` | one-shot ES→EN via Vertex Gemini (needs GCP ADC; run once) |
| `build_interface.py` | generates `human_eval.html` from the three sample CSVs |
| `human_eval.html` | the annotation interface — open locally, no server needed |
| `results/` | exported results CSVs go here (one per annotator) |
| `score_results.py` | un-blinds, aggregates per-arm means/preference/per-category, computes weighted κ |

## Annotation workflow

1. One-time (already done): `uv run python -m analysis.human_eval.translate_english`
   then `uv run python -m analysis.human_eval.build_interface`.
2. Each annotator: open `human_eval.html` in a browser (repo checked out with the
   dataset at `data/americasnlp2026/` for the images), enter your name, score all 15
   items, click **Export results CSV**, move the download into
   `analysis/human_eval/results/`, commit it.
3. Analysis: `uv run python -m analysis.human_eval.score_results` (needs ≥2
   annotators for κ). Progress autosaves in the browser (localStorage) per
   annotator name — closing the tab loses nothing.
