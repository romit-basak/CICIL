# Stage 1 Related-Work — Paragraph Outline (Romit)

Plan for writing the Stage 1 (Vision & VLM) related-work section as prose, matching the teammate sections' style: one paragraph per idea, each stating *what the paper does* then *how our work relates or differs*. Author-year inline; full refs with links at section end.

**Target length:** ~400–550 words (this is one of four contributor sections in a 4-page paper). That's ~6 tight paragraphs. **Formally discuss 5–6 papers** (rubric); the rest are cited inline as support/method refs.

**Coordination rules (do not violate):**
- **CIC** = cite for its *VQA-before-captioning mechanism* only. Nandita already owns CIC's *category set* (Architecture/Clothing/Food&Drink/Dance&Music/Religion). Don't rewrite her angle.
- Don't restate the **ChrF++-not-comparable-across-languages** caveat — Mehek/EDA already make it. Defer to it.
- Papers that are yours alone (no overlap): CultureCLIP, RAVENEA, SmolVLM, LoRA, SigLIP, CLIP, Qwen2.5-VL, CulturalVQA, CVQA, MaRVL, GIMMICK.

---

## ¶1 — The bottleneck: generic VLMs discard culture *(sets up RQ1)*
- **Papers:** CLIP `radford2021clip` (foundational, cited inline) + evidence: MaRVL `liu2021marvl`, CVQA `romero2024cvqa`, optionally GPT-4V probing `2024gpt4vculture`.
- **Claim to state:** Web-scale contrastive pretraining (CLIP) yields Western-centric encoders; MaRVL shows cross-cultural/cross-lingual transfer collapses relative to English, and CVQA shows SOTA MLLMs still fail on culturally-grounded VQA in low-resource languages.
- **Our-work hook:** This is precisely the failure the CICIL shared-task winner flagged but didn't fix — generic VLMs strip cultural content *before* translation. Stage 1 targets it directly (RQ1).
- **Discuss fully?** MaRVL and CVQA get a sentence each; CLIP is an inline anchor. (Counts toward 1 formal paper: CVQA.)

## ¶2 — Fixing culture at the vision side: CultureCLIP *(1 formal paper)*
- **Paper:** CultureCLIP `huang2025cultureclip` (COLM 2025).
- **Claim to state:** Builds CulTwin (visually similar / culturally different concept–caption–image triplets), fine-tunes CLIP with tailored contrastive learning; **+5.49%** on fine-grained cultural concept recognition.
- **Our-work hook:** Same problem, *different intervention layer*. They retrain the **encoder**; we keep SmolVLM's SigLIP encoder frozen and inject cultural signal at the **decoder** via VQA prompting + LoRA — cheaper, and viable on 50-example pilot sets where contrastive encoder retraining is not.

## ¶3 — VQA-before-captioning: CIC as our method ancestor *(1 formal paper — CORE)*
- **Paper:** CIC `yun2024cic` (IJCAI 2024) — **method angle only**.
- **Claim to state:** Three-step pipeline — generate cultural questions → extract cultural elements via VQA → caption with an LLM — produces more culturally descriptive captions than VLP baselines (human eval, 45 participants / 4 cultures).
- **Our-work hook:** This is the conceptual ancestor of our cultural VQA module. We differ by (i) *fine-tuning* the VLM rather than using it off-the-shelf, (ii) targeting indigenous languages of the Americas, and (iii) producing a Spanish intermediate whose cultural annotations *drive downstream retrieval* rather than ending at the caption.
- **Connection:** the categories our VQA questions probe — ceremony, material culture, landscape, kinship — come from Nandita's ethnographically-derived taxonomy (§data), not CIC's generic set.

## ¶4 — Coupling vision to retrieval: RAVENEA *(1 formal paper — the novel-contribution bridge)*
- **Paper:** RAVENEA `yuan2025ravenea` (EMNLP 2025 Findings — *verify venue*).
- **Claim to state:** Defines culture-informed image captioning (cIC); culture-aware retrieval improves lightweight VLMs by **≥+3.2% (cVQA) / +6.2% (cIC)** absolute.
- **Our-work hook:** Strongest external evidence for our central design bet — that cultural grounding should *drive* retrieval. We extend it: our retrieval keys are the cultural concept annotations emitted by Stage 1's VQA, not generic text.
- **Connection:** hands off to Mehek's Stage 2 section (FAISS + Gemini) — cite as the shared hinge between our two components.

## ¶5 — Model and fine-tuning choices: SmolVLM + baseline *(1 formal paper + inline method refs)*
- **Papers:** SmolVLM `marafioti2025smolvlm` (formal); SigLIP `zhai2023siglip` + LoRA `hu2022lora` (inline); Qwen2.5-VL `qwen2025qwen25vl` (baseline); LLaVA-1.5 `liu2024llava` (one-clause backup).
- **Claim to state:** SmolVLM (2B) runs in <1GB and rivals far larger models — feasible under the $50 / T4 budget. SigLIP encoder frozen; decoder adapted with LoRA (rank 16, α=32, lr 2e-4). Qwen2.5-VL is the generic off-the-shelf VLM baseline.
- **Our-work hook:** The frozen-encoder + LoRA-decoder split is *why* cultural signal must enter through VQA prompting (encoder can't be retrained on 50 examples). Qwen2.5-VL with no cultural fine-tuning is the ablation control that isolates Stage 1's contribution (RQ1).
- **Connection:** Nandita's EDA (Wixárika = 20 pilot examples) motivates the parameter-efficient choice and frames RQ2 (data-starved languages).

## ¶6 — What's hard to ground, and how we'll measure it: CulturalVQA (+ GIMMICK) *(1 formal paper, feeds RQ3)*
- **Papers:** CulturalVQA `nayak2024culturalvqa` (formal); GIMMICK `schneider2025gimmick` (support); optionally Yadav `yadav2025cultural` for rubric grounding.
- **Claim to state:** CulturalVQA shows VLM cultural competence varies sharply by *facet* (clothing/rituals > food/drink) and *region*; GIMMICK finds models handle *tangible > intangible* culture.
- **Our-work hook:** These give us (i) a template for the per-cultural-category ChrF++ heatmap and (ii) a prior for which categories Stage 1 will miss (intangible/ritual) — both directly serve RQ3. Yadav's cultural-theory framing backs how we define "cultural accuracy" in the human-eval rubric.
- **Connection:** defer cross-language ChrF++ comparability to Mehek/EDA; here we only justify the *category-level* breakdown.

---

## Which 6 to formally discuss (if space is tight)
CVQA (¶1) · CultureCLIP (¶2) · CIC (¶3) · RAVENEA (¶4) · SmolVLM (¶5) · CulturalVQA (¶6).
Everything else (CLIP, MaRVL, SigLIP, LoRA, Qwen2.5-VL, LLaVA, GIMMICK, Yadav, GPT-4V probing) is cited inline — plenty to lean on if a reviewer wants more depth, and useful for the 8-page final where the section expands.

## Suggested topic sentences (starting points, rewrite in your own voice)
- ¶1: "Generic vision-language models are trained on predominantly Western web data, and their cultural blind spots are now well documented."
- ¶2: "One line of work attacks this at the encoder…" (CultureCLIP)
- ¶3: "A complementary line asks cultural questions before captioning…" (CIC)
- ¶4: "Recent work shows these cultural annotations are most useful when they drive retrieval…" (RAVENEA)
- ¶5: "For the vision backbone we prioritize models that fine-tune cheaply…" (SmolVLM)
- ¶6: "Finally, prior benchmarks tell us which cultural content is hardest to ground…" (CulturalVQA/GIMMICK)
