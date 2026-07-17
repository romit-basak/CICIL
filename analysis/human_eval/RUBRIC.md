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

Two sheets are produced, because the team reads Spanish but not the target
languages:

| Sheet | Text judged | Who annotates | Answers |
|---|---|---|---|
| `sample_spanish.csv` | Stage-1 **Spanish** description | the team, now | isolates Stage 1 (RQ1/RQ3) |
| `sample_target.csv` | final **target-language** caption | native/heritage speakers | end-to-end quality |

Judge the Spanish sheet against the image. The target sheet needs a speaker of the
language; where none is available, leave it for recruited annotators and report the
Spanish-side results as the primary human evaluation (as the paper already does for
Stage 1 intrinsic quality).

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
| `sample_spanish.csv` | annotation sheet, Stage-1 Spanish (fill the score columns) |
| `sample_target.csv` | annotation sheet, target-language captions |
| `sample_key.csv` | A/B → arm un-blinding map (for analysis only — do not show annotators) |
