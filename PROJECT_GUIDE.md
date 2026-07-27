# CICIL, Explained Plainly

This is a plain-English walkthrough of the whole project: what we're building, why the
obvious/simple approach doesn't work, what we're doing instead and why, and the actual
problems we've run into along the way. It's meant to be readable by a teammate (or
future you) who wants the full picture without reading every commit and Slack thread.

---

## 1. What we're actually trying to do

There's an academic competition (AmericasNLP 2026's "CICIL" shared task) with a simple-
sounding prompt: **look at a photo of something culturally specific to an Indigenous
community in the Americas, and write a caption for it — not in English or Spanish, but in
that community's own language.** The languages in scope are Guaraní, Yucatec Maya,
Bribri, and Wixárika (Huichol) — all real languages spoken today, all dramatically
under-represented in the data that AI models are normally trained on.

The photos aren't generic stock images. They show things like traditional ceremonies,
handmade crafts, specific agricultural practices, or culturally loaded objects — a photo
that means something different, and more, to someone from that culture than it would to
a generic image-captioning model.

Two things make this hard at the same time:
1. **The images need cultural understanding**, not just object recognition. A model that
   says "a woman near a fire" when the real answer is "a woman preparing traditional
   ceremonial food during [specific ritual]" has technically described the pixels but
   missed the entire point.
2. **The target languages have almost no training data.** These aren't languages like
   Spanish or French with billions of web pages to learn from. Guaraní has a moderate
   amount of digitized text; Wixárika and Bribri have very little. No off-the-shelf model
   speaks any of these languages well.

This splits into two halves. **Stage 1** involves turning the image into a rich,
culturally-grounded description. **Stage 2** involves turning that description into
the actual target-language caption, plus the retrieval infrastructure and paper
analysis that support it.

---

## 2. Why the "obvious" approach doesn't work

If you had to build this in an afternoon, you'd probably do:

```
[Image] → [some off-the-shelf vision-language model] → [Spanish caption]
        → [some off-the-shelf translation model] → [target-language caption]
```

This is a real thing people have already tried (see §3), and it has two structural
problems:

**Problem A — generic captioning throws away the cultural signal before translation
even starts.** A general-purpose vision-language model (VLM) was trained to answer "what
is this a picture of" for a huge, generic distribution of internet images. It has no
special reason to notice that a woven bag is a *specific traditional Wixárika craft* with
its own name and significance, rather than "a colorful bag." Once that detail is lost in
the Spanish caption, no downstream translation step can ever recover it — you can't
translate cultural nuance that was never written down.

**Problem B — the target languages are too low-resource for translation models (or
even large LLMs) to just know them.** Ask Gemini or GPT to translate a sentence into
Wixárika cold, and it's translating a language it has seen only a tiny amount of during
training. It will produce *something*, but it's often generic, sometimes wrong, and
occasionally degenerates into repeating itself (a known failure mode of low-confidence
generation — more on this in §6).

Both problems need a different fix, which is why the pipeline has two stages, not one.

---

## 3. What existing architectures actually look like (and their limits)

This isn't a hypothetical "existing systems are generic" — we found two real submissions
to this exact 2026 shared task and read what they actually did and found. Both are worth
knowing about specifically:

**A near-identical architecture that won the competition.** One winning submission uses
almost exactly our shape of pipeline: a vision-language model (Qwen2.5-VL) generates a
Spanish intermediate caption, then Gemini 2.5 Flash translates it using retrieval-
augmented "many-shot" prompting (showing the model real example translations before
asking it to translate the new sentence). It placed 2nd in human evaluation. Its key
finding: **retrieval only helps when there's a large, in-domain corpus of real examples
to retrieve from** — for languages where the example bank is small or off-topic,
retrieval doesn't help and can even hurt. It also found that a huge chunk of their
Guaraní improvement (~28 ChrF++, a large jump) came specifically from *synthetic data
augmentation* — generating extra training-like data rather than only using what
already existed. Both points are directly relevant to decisions we've made (§7, §8).

**A competing submission that fine-tuned NLLB-200 instead of prompting an LLM.** Instead
of showing a general-purpose LLM examples at translation time, this team fine-tuned
NLLB-200 (a dedicated, smaller translation model that already knows 200 languages,
including Guaraní) directly on parallel data for each target language. Interesting
twist: on the automatic metric (ChrF++), the **plain, non-fine-tuned NLLB model actually
scored higher** than their fine-tuned version — yet the fine-tuned version won more votes
in blind human evaluation. This is a real, documented case of the exact tension we keep
running into ourselves: automatic metrics and actual human-judged quality don't always
point the same direction (§7 has our own version of this).

**The common thread in both:** neither treats "ask culturally-targeted questions about
the image before describing it" as a first step — they go straight from image to a
single caption. That's the specific gap our Stage 1 cultural-VQA module is designed to
close, and it's the main novel piece of what I'm contributing.

---

## 4. Our architecture, and why each piece exists

```
[Image]
   ↓
[Stage 1: culturally fine-tuned vision-language model]
   → asks itself a set of targeted cultural questions about the image first
   → then writes a grounded Spanish description informed by those answers
   ↓
[Stage 2: culturally-indexed retrieval + LLM translation]
   → looks up real example translations that are culturally similar to this one
   → shows those examples to Gemini 2.5 Flash before asking it to translate
   ↓
[Caption, in the target Indigenous language]
```

### Stage 1, piece by piece

**The cultural VQA module — ask before you describe.** Before generating any caption,
the model is prompted with a fixed set of targeted questions: what ceremony (if any) is
shown, what traditional objects are visible, what's happening with the landscape or
setting, what kinship/social relationships are implied. Only after answering those does
it synthesize a final description. The idea (borrowed and adapted from a paper called
CIC, which pioneered VQA-first captioning) is that **forcing the model to notice specific
cultural categories up front prevents it from defaulting to a generic, surface-level
description.** This is the actual novel contribution of Stage 1 — not the model
architecture itself, but the structured questioning that happens before generation.

**Fine-tuning a small vision-language model (SmolVLM, 2 billion parameters) via LoRA.**
We don't train a model from scratch — we take an existing small VLM and adapt it using
LoRA (a technique that trains a small set of extra parameters instead of the whole
model, much cheaper and faster). Why fine-tune at all, and why this specific model? See
§8 for the reasoning.

**Retrieval for Stage 1 too — an encyclopedia at the model's elbow (added late July).**
The human evaluation (§6) revealed that even the culturally-prompted model almost never
names an actual culture, artifact, or sacred site — because it has nowhere to *learn*
them from: a 2B model can't memorize the encyclopedia, and it turned out even the 7B
teacher names a culture in ≤10% of outputs unaided. The fix mirrors how a human
annotator actually works: look at the image, then look things up. Two lookup channels,
both harvested systematically from Wikipedia/Wikimedia Commons per culture:
- **Similar-image search (CBIR):** the dev image is compared against ~220–520
  license-clean Commons photos of that culture; the nearest matches' titles and
  descriptions ("Ñandutí detalle.jpg", "Wirikuta desert...") are injected into every
  question as context, each tagged with a confidence band («coincidencia fuerte» for
  near-certain matches, «posiblemente»-grade otherwise).
- **Encyclopedia text search:** the model's own visual answers query ~90–540 Wikipedia
  lead paragraphs for that culture; retrieved snippets feed the final synthesis step.
Crucially, the prompt asserts the image's culture as *given* (we know it from the task
metadata — that's how the right bank was chosen), and reserves hedging for the concept
identification, which is the genuinely uncertain part.

### Stage 2, piece by piece

**Retrieval-augmented translation.** Rather than asking Gemini to translate cold, Stage
2 first searches a database of real, previously-known Spanish↔target-language sentence
pairs for the ones most similar to the current caption, and shows Gemini those as
examples before asking it to translate the new one. This is the standard fix for
"translation model doesn't really know this language" — you compensate with in-context
examples rather than hoping the model already knows the language.

**Culturally-indexed, not just text-indexed, retrieval.** The twist specific to our
project: instead of searching for examples that are merely *textually* similar to the
Spanish caption, we search using the *cultural annotations* Stage 1 produced (the
answers to "what ceremony," "what objects," etc.). The idea is that two captions can be
culturally similar (same kind of ceremony, same category of object) even if their exact
wording is quite different — and cultural similarity is what should drive which examples
help translation, not surface lexical overlap.

---

## 5. How this addresses the two problems from §2

- **Problem A (culture lost before translation)** → fixed by the cultural-VQA module:
  the model is forced to notice and record cultural detail *before* it ever writes the
  Spanish caption that Stage 2 will translate, so there's something for Stage 2 to work
  with.
- **Problem B (target languages too low-resource)** → fixed by retrieval-augmented
  translation: instead of expecting Gemini to already know Wixárika, we show it real
  examples every time, culturally matched rather than just textually matched.

---

## 6. The actual story: problems we've hit, in order

This project did **not** go in a straight line. Here's what actually happened, roughly
chronologically, and it's worth knowing because several of these problems are still
open and shape what we can honestly claim.

**Fine-tuning on the tiny gold dataset came back flat.** The only images with real
Spanish reference captions (for training/eval purposes) number just 19. A LoRA fine-tune
directly on those 19 examples produced no real improvement over the untuned baseline —
17.55 vs. 16.72 on our proxy metric, well within noise. The lesson: **the bottleneck was
data, not model capacity or hyperparameters.** No amount of hyperparameter tuning fixes
19 training examples.

**The fix: distillation, not more fine-tuning.** Instead of training directly on the
tiny gold set, we had a much bigger vision-language model (Qwen2.5-VL, 7B) generate
"silver" (model-written, not human-written) Spanish descriptions on many more images,
and trained the small SmolVLM student to imitate that larger teacher. Crucially, the
training examples used the *exact* prompts the model will see at real deployment time
(the same cultural-category questions, not a simplified proxy task) — otherwise the
fine-tuned adapter would get good at a task it's never actually asked to do. This
worked: the distilled model went from 16.72 to matching the 7B teacher (~20-21) at 2B
parameters, on our held-out proxy metric.

**We tried to make the training set bigger with outside images (Wikimedia Commons).**
Since the shared task itself only provides a small number of images, we scraped roughly
1,300 additional, license-clean, culturally-relevant photos from Wikimedia Commons to
give the student model more to learn from. Two real problems came up here:
- A scraping category we used for Bribri ("Talamanca") turned out to also be the name
  of an unrelated village in Catalonia, Spain — so a large fraction of our "Bribri"
  images were actually pictures of a Spanish castle. Caught by spot-checking a sample
  caption, fixed by re-scraping with more specific category names.
- After all that work, the honest measured benefit was small: it helped the Spanish-
  language proxy metric by about 1 point, but didn't clearly help the final,
  translated-language metric. (That comparison was later re-run through the real
  retrieval banks once they existed — the arms came out tied, confirming the null.)
  The scraped images did, however, become the backbone of the Stage 1 retrieval bank
  (§4), which grew to ~2,100 images across the five cultures — so the scrape paid off,
  just not for the purpose it was built for.

**For weeks, the Stage 2 retrieval bank didn't actually exist.** (Since resolved —
see "The banks got built" below — but it shaped everything measured in this period.)
To do "look up similar real examples" (§4), Stage 2 needs a real bank of
Spanish↔target-language sentence pairs to search. For most of the project, for 4 of 5
languages, that bank **didn't exist at all** — Gemini translated completely blind. For
the 5th (Wixárika), the bank had only 20 entries — the same 20 images used elsewhere
as our tiny gold-evaluation set, not a real corpus. This meant every end-to-end
(translated-language) score measured in that period ran through an unfinished,
placeholder Stage 2 — we genuinely couldn't tell whether Stage 1 improvements "didn't
survive translation" or whether Stage 2's retrieval was just too thin to show any
difference. This was flagged directly by a teammate mid-project, and any pre-bank
end-to-end number quoted from old notes still carries that caveat.

**A parallel teammate deliverable (an experimental Stage 2 upgrade) arrived built on a
different, older version of the code than what we'd since evolved to.** It added a
genuinely good new capability (testing culturally-indexed retrieval against plain-text
retrieval as an independent variable) but, because it was developed without visibility
into concurrent changes on our side, it would have silently deleted an unrelated but
essential feature (the ability to score outputs from different fine-tuned model
versions) if merged without reconciling the two first. Caught before merging by diffing
line-by-line rather than trusting the "just drop these files in" instructions at face
value. (Since reconciled and merged — the cultural-vs-text retrieval ablation is part
of the Stage 2 sweep now running.)

**Infrastructure problems, repeatedly.** Training runs on cloud GPUs got interrupted by
preemption (cheap "spot" instances can be reclaimed at any time), by out-of-memory
crashes from a subtle memory-fragmentation issue, and once by a crashed server process
that silently kept holding onto GPU memory in the background even after the visible
process was killed. None of these are exotic — they're the normal cost of doing
real GPU work on a tight budget — but each one needed a genuine fix (automatic
checkpoint-and-resume logic, memory-allocator tuning, learning to check *actual* GPU
memory usage rather than trusting "is the process still running").

**Automatic metrics and real translation quality don't always agree.** This showed up
three separate times: the fully-distilled model scored *better* on the Spanish-language
proxy than an earlier, smaller version, but scored about the *same or slightly worse* on
the final translated-language metric. This isn't unique to us — the competing NLLB
submission in §3 found the same pattern (worse automatic score, better human-judged
result). The honest conclusion isn't "our numbers are wrong" — it's that a single
automatic metric, scored against one reference caption per image, on only ~50 images, is
a genuinely noisy way to measure "did this actually get better," and we should say so
plainly rather than overclaim a win every time a number goes up.

**The banks got built — and the "real" bank made things worse.** (Late July.) All five
retrieval banks now exist for real: 8k–16k genuine Spanish↔target sentence pairs per
language, with per-source licenses documented. The surprise: on Wixárika, the real
9,940-pair bank scored *worse* end-to-end than the 20-pair placeholder it replaced —
because the placeholder pairs, tiny as they were, were caption-style and in-domain,
while the real corpus is general text. Domain match beats scale for few-shot retrieval.
This also retroactively cleaned up an earlier confusion: an apparent gap between two
Stage 1 ablation arms vanished once both were re-run through the same real bank.

**Wixárika and Bribri captions were coming out as word-salad loops.** The human eval
surfaced that most captions in the two hardest languages degenerated into repetition
("kek tso tso tso..."). An A/B ablation found the fix was not the obvious knob
(frequency penalty did nothing) but plain sampling temperature: greedy-ish decoding at
temperature 0 makes a low-confidence model loop, while temperature 0.7 with a fixed
seed (still reproducible) cut degeneration by roughly two-thirds and *raised* ChrF++ —
+5.0 on Wixárika. A reminder that "deterministic = more scientific" isn't free.

**The human eval said the quiet part out loud: nobody names a culture.** Scoring 15
images by hand (via English pivot translations, since nobody on the team reads Spanish
— itself a documented protocol decision) found cultural accuracy at floor for BOTH the
generic and cultural arms, and only 7% of all outputs named any culture at all. Root
cause, verified in code: Stage 1 was never told which culture the image came from, even
though the task metadata says so. Recognizing a culture purely from pixels is an
ill-posed problem (the same lace pattern exists on three continents) — the pipeline was
withholding the one piece of information that disambiguates it.

**Retrieval into Stage 1 fixed the naming problem — with one big caveat each way.**
(The current work.) With both retrieval channels on, the rate at which the *same 2B
student* names the culture jumped from ~2–14% to 36–92% across all five languages, and
on the languages whose images show distinctive sites/artifacts it started naming the
*specific* ones (Wirikuta for bare desert hills that every earlier arm called "no
cultural content"; Cahuita and cacao for Bribri). The caveats: (1) ChrF++ barely moves
— the reference captions can't reward naming they don't contain, so we measure this
with term-rate audits and human spot checks instead; (2) retrieval can misattribute —
the Paraguay-heavy Guaraní bank pulled an Argentine chamamé festival across the border,
confidently.

**A capability gap we didn't expect: the small model can't hedge — and the fix
worked, with a catch.** The retrieval prompts instruct the model to mark uncertain
matches with "posiblemente." The 7B teacher does (roughly half its captions hedge
appropriately). The 2B student did the opposite — it hedged *less* than without
retrieval, converting every retrieved concept into a confident assertion, right or
wrong, and prompt engineering didn't fix it. Retraining the student on teacher
outputs *with* the retrieval context in the training prompts (RAG-aware
distillation) **did** fix it: the retrained student hedges at teacher-like rates in
all five languages. The catch: it inherited the teacher's conservatism along with
its caution — it now hedges the culture properly but stopped naming the specific
sites and artifacts the un-retrained student had been (correctly, if overconfidently)
naming. At 2B parameters, specificity and calibration appear to trade off; only the
7B teacher holds both at once. That trade-off is itself one of the paper's cleanest
findings.

---

## 7. Rationale behind the key decisions (the "why," collected in one place)

- **Why distillation instead of just fine-tuning harder?** Because the real constraint
  was the *size* of the human-labeled dataset (19 images), not model capacity. Making a
  bigger teacher model generate more training examples sidesteps the actual bottleneck;
  tuning hyperparameters on 19 examples does not.
- **Why train on the exact deployment prompts, not a simplified version?** A model
  fine-tuned on a different (simpler) task than what it will actually be asked to do at
  inference time may improve on the training task while staying bad at the real one.
  Matching them exactly avoids finding out too late that the two diverged.
- **Why a small model (SmolVLM, 2B) at all, instead of just using the big teacher
  directly in production?** Compute budget and latency — a 2B model fine-tuned to nearly
  match a 7B model's quality is far cheaper to run at scale, and "can a small distilled
  model match a big teacher" is itself an interesting research question (RQ1).
- **Why retrieval-augmented translation instead of fine-tuning a dedicated translation
  model (like NLLB) for Stage 2?** Both are legitimate; we chose retrieval + a strong
  general LLM as the primary approach because it doesn't require enough parallel
  training data to fine-tune a translation model well, and it's the same design that won
  this exact competition last cycle (§3). NLLB fine-tuning is kept as a stretch-goal
  alternative to compare against, not abandoned.
- **Why culturally-indexed retrieval instead of plain text-similarity retrieval?** The
  hypothesis is that cultural relevance, not surface wording, is what makes a retrieved
  example actually useful as a translation guide. This is a testable but still
  *unconfirmed* hypothesis — the real banks now exist, and the Stage 2 sweep
  (cultural-vs-text query arms × retrieval depth) is the experiment that answers it;
  results merge into the final paper table.
- **Why Vertex AI instead of just calling the Gemini API directly?** Licensing. The
  dataset's license (CC BY-NC) forbids feeding it into a service that trains on the
  data. The free/direct Gemini API tier does train on requests; Vertex AI (covered by
  our education grant) does not.
- **Why does all our code have to run on three different kinds of hardware (NVIDIA GPU,
  Apple Silicon, plain CPU)?** Practical necessity: development happens on a Mac, but
  the actual training budget is a small cloud-GPU grant. Code that only works on one
  has to be debugged for the first time on the expensive, time-limited resource — every
  script needs a cheap, local way to catch bugs before spending real GPU-hours on them.

---

## 8. Where things honestly stand right now (updated 2026-07-27)

- Stage 1 distillation is done and is a real, positive result: base model → distilled
  model closed nearly the whole gap to a 7B teacher, at 2B parameters.
- The Commons-image augmentation and the "does teacher context help" ablation are both
  done, and both came back as **honest negative/null results** on end-to-end quality —
  reported as such, not spun into a win.
- **The retrieval banks are real now** (all 5 languages, licenses documented), the old
  placeholder-bank caveat is resolved — and it resolved *interestingly*: domain match
  beats corpus size (§6).
- **The decoding fix is in** (temperature 0.7 + seed): Wixárika/Bribri degeneration is
  down by ~two-thirds and their scores up; all final numbers use this setting.
- **Stage 1 retrieval is the current headline**: culture-naming went from near-zero to
  the strong majority of captions across all five languages, with genuine site/artifact
  grounding on the languages whose images support it, verified by audits and spot
  checks rather than ChrF++ (which is blind to it). Its cost side (confident
  misattribution by the small model) is measured, not hidden.
- **In flight right now**: a RAG-aware re-distillation of the student (does calibration
  transfer from teacher to student?), the final all-language Stage 2 re-runs at the new
  decoding, and the merged results table. The team's Stage 2 sweep (retrieval-k and
  query-arm ablations) runs in parallel on the same fixed inputs.

If you only remember one thing from this document: **the story matured from "several
plausible improvements don't clearly help" to "we found the load-bearing missing
ingredient — cultural knowledge has to be *supplied* at inference, not hoped for from
the weights — and we can show both what that fixes and what it breaks."** The honest
accounting of the caveats (metric blindness, misattribution, the hedging capability
gap) is as much a part of the contribution as the win.
