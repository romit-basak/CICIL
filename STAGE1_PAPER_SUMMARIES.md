# Stage 1 Paper Summaries — Citation Guide for the Preliminary Paper

Companion to [STAGE1_READING_LIST.md](STAGE1_READING_LIST.md). For each paper: what it does, **why it's relevant to our pipeline**, and **what to cite it for**. Citation keys are suggestions for the `.bib`.

> Reminder for the related-work section: every paper must be discussed *in the context of our work* — how it's similar/different, and whether it's a foundation, a baseline, or a motivation.

---

## A. Culturally-aware vision & captioning — the heart of Stage 1

### 1. CultureCLIP (Huang et al., COLM 2025) — `huang2025cultureclip` **[ASSIGNED]**
arXiv: https://arxiv.org/abs/2507.06210
- **What it does:** Builds *CulTwin*, a synthetic dataset of concept–caption–image triplets where items look visually similar but are culturally different, then fine-tunes CLIP with tailored contrastive learning to separate these "cultural twins."
- **Key numbers:** Up to **+5.49%** on fine-grained cultural concept recognition while preserving general-domain generalization.
- **Relevance to us:** Direct evidence that generic CLIP-style encoders **fail to distinguish culturally-distinct but visually-similar concepts** — exactly the bottleneck we attack in Stage 1. Their fix is contrastive fine-tuning at the *encoder*; ours is VQA prompting + LoRA at the *decoder* (SmolVLM's SigLIP encoder stays frozen). Good contrast point: same problem, different intervention layer.
- **Cite for:** the claim that generic VLMs lose cultural nuance; precedent for culturally-targeted fine-tuning; framing of "similar-looking, culturally-different" concepts.

### 2. CIC: A Framework for Culturally-Aware Image Captioning (Yun & Kim, IJCAI 2024) — `yun2024cic` **[ASSIGNED]**
arXiv: https://arxiv.org/abs/2402.05374 · IJCAI: https://www.ijcai.org/proceedings/2024/180
- **What it does:** Three-step pipeline — (1) generate questions from **cultural categories**, (2) extract cultural visual elements via **VQA**, (3) generate culturally-aware captions with an **LLM** prompted on those elements.
- **Evaluation:** Human eval with **45 participants across 4 cultural groups**; CIC produces more culturally descriptive captions than VLP captioning baselines (e.g., BLIP).
- **Relevance to us:** This is the **direct conceptual ancestor of our cultural VQA module.** Our Stage 1 is essentially CIC's "ask cultural questions first, then caption," specialized to indigenous-American cultures and feeding a Spanish intermediate. Key difference: we *fine-tune* the VLM (LoRA) and couple the cultural annotations into downstream retrieval (Stage 2) rather than stopping at the caption.
- **Cite for:** justification of the VQA-before-captioning design; precedent that VLPs "lack detailed descriptive captions for cultural elements"; the human-eval rubric idea (motivates our 3-point cultural-accuracy rubric).

### 3. RAVENEA (Yuan et al., 2025) — `yuan2025ravenea`
arXiv: https://arxiv.org/abs/2505.14462
- **What it does:** Benchmark for retrieval-augmented visual culture understanding; defines two tasks — culture-focused VQA (cVQA) and **culture-informed image captioning (cIC)**. Extends datasets with **11,396** human-ranked Wikipedia docs over 8 countries / 11 cultural categories; evaluates 7 retrievers and 15 VLMs.
- **Key numbers:** Culture-aware retrieval improves lightweight VLMs by **≥+3.2% absolute on cVQA** and **≥+6.2% absolute on cIC** (published ICLR 2026 figures; the earlier preprint reported +6%/+11%).
- **Relevance to us:** Strongest external validation of our **core novel contribution** — coupling cultural annotations to retrieval. They show culture-aware retrieval beats non-augmented VLMs on captioning; we extend this by sourcing the cultural index keys from Stage 1's VQA output rather than generic text.
- **Cite for:** evidence that culture-aware retrieval helps captioning; definition of culture-informed captioning as a task; bridge between my Stage 1 and Mehek's Stage 2.

---

## B. Models & efficient fine-tuning — methodology grounding

### 4. SmolVLM (Marafioti et al., 2025) — `marafioti2025smolvlm`
arXiv: https://arxiv.org/abs/2504.05299
- **What it does:** Family of compact VLMs (256M–2.2B) for resource-efficient inference; smallest variant uses **<1GB GPU memory** and beats the 300× larger Idefics-80B. Fully open (Apache 2.0): checkpoints, data, recipes.
- **Relevance to us:** The model I fine-tune. Justifies the **2B choice under a $50 / T4 budget**; its SigLIP-encoder + SmolLM-decoder design is what lets us freeze vision and LoRA the decoder.
- **Cite for:** model description; the feasibility/efficiency argument for small VLMs in low-resource settings.

### 5. LoRA (Hu et al., ICLR 2022) — `hu2022lora`
arXiv: https://arxiv.org/abs/2106.09685
- **What it does:** Freezes pretrained weights and injects trainable low-rank matrices, cutting trainable params and GPU memory with no inference-latency penalty.
- **Relevance to us:** Our fine-tuning method (rank 16, α=32, lr=2e-4). Foundational justification for adapting the decoder cheaply on ~50 pilot examples/language.
- **Cite for:** the LoRA hyperparameter setup; the parameter-efficiency rationale.

### 6. SigLIP — Sigmoid Loss for Language Image Pre-Training (Zhai et al., ICCV 2023) — `zhai2023siglip`
arXiv: https://arxiv.org/abs/2303.15343
- **What it does:** Replaces CLIP's softmax contrastive loss with a pairwise sigmoid loss; trains efficiently at small and very large batch sizes.
- **Relevance to us:** SmolVLM's **frozen vision encoder**. Lets me state precisely what is *not* updated and argue the cultural signal must enter via the decoder + VQA prompting.
- **Cite for:** describing the frozen encoder; the architecture section.

### 7. LLaVA-1.5 — Improved Baselines with Visual Instruction Tuning (Liu et al., CVPR 2024) — `liu2024llava`
arXiv: https://arxiv.org/abs/2310.03744
- **What it does:** Shows a simple CLIP-ViT + MLP projector + instruction data reaches SOTA across 11 benchmarks with 1.2M public examples, ~1 day on one 8-A100 node.
- **Relevance to us:** Our **Stage 1 backup model** if SmolVLM underperforms; the reference visual-instruction-tuning recipe our VQA prompting extends.
- **Cite for:** backup-model justification; visual instruction tuning as the paradigm behind prompting a VLM with structured questions.

### 8. Qwen2.5-VL Technical Report (Qwen Team, 2025) — `qwen2025qwen25vl`
arXiv: https://arxiv.org/abs/2502.13923
- **What it does:** Strong open VLM family with dynamic-resolution ViT and window attention; 72B variant rivals GPT-4o on document/diagram tasks.
- **Relevance to us:** Our **generic off-the-shelf VLM baseline** for the Stage 1 ablation — the "no cultural fine-tuning" control that isolates Stage 1's contribution (RQ1).
- **Cite for:** baseline model description; the ablation design.

### 9. CLIP — Learning Transferable Visual Models from Natural Language Supervision (Radford et al., ICML 2021) — `radford2021clip`
arXiv: https://arxiv.org/abs/2103.00020
- **What it does:** Contrastive pretraining on 400M web image–text pairs enabling zero-shot transfer; the ancestor of SigLIP/CLIP-based encoders.
- **Relevance to us:** Foundational citation explaining *why* generic encoders trained on Western-centric web data discard cultural specificity — the gap our pipeline targets.
- **Cite for:** background on contrastive VLMs; root cause of the cultural bottleneck.

---

## C. Cultural / multilingual benchmarks & evaluation — supports RQ2 (resource level) and RQ3 (hard categories)

### 10. CulturalVQA — Benchmarking VLMs for Cultural Understanding (Nayak et al., EMNLP 2024) — `nayak2024culturalvqa`
arXiv: https://arxiv.org/abs/2407.10920 · ACL: https://aclanthology.org/2024.emnlp-main.329/
- **What it does:** Geo-diverse cultural VQA over 11 countries and **5 facets** (traditions, rituals, food, drink, clothing). GPT-4V/Gemini show strong North-America understanding but weak Africa; performance varies by facet (clothing/rituals/traditions > food/drink).
- **Relevance to us:** Methodological template for our **per-cultural-category ChrF++ breakdown (RQ3)** and the regional-disparity framing (RQ2). Their facet taxonomy is a useful comparison for Nandita's taxonomy.
- **Cite for:** per-category/per-region evaluation methodology; evidence of uneven cultural competence across facets.

### 11. CVQA (Romero et al., NeurIPS 2024 D&B, Oral) — `romero2024cvqa`
arXiv: https://arxiv.org/abs/2406.05967
- **What it does:** Culturally-driven VQA over **30 countries, 31 languages, 13 scripts, ~10k questions**, built with native speakers and cultural experts; challenging for SOTA MLLMs.
- **Relevance to us:** Demonstrates current MLLMs underperform on culturally-grounded, **low-resource-language** VQA — the exact failure mode our cultural VQA module addresses, and supports RQ2's data-richness hypothesis.
- **Cite for:** evidence of cultural bias in MLLMs; importance of pairing culturally-specific imagery with low-resource languages (not just translating text).

### 12. MaRVL — Visually Grounded Reasoning across Languages and Cultures (Liu et al., EMNLP 2021, Best Long Paper) — `liu2021marvl`
ACL: https://aclanthology.org/2021.emnlp-main.818/
- **What it does:** Native-speaker-elicited true/false visual reasoning over 5 typologically diverse languages; cross-lingual transfer lags far behind English.
- **Relevance to us:** Early, well-cited evidence that VLMs trained on Western-centric data **transfer poorly across cultures** — foundational motivation for RQ1/RQ2.
- **Cite for:** the cross-cultural transfer gap; motivation that Western-centric pretraining is the problem.

### 13. GIMMICK (Schneider et al., 2025) — `schneider2025gimmick`
arXiv: https://arxiv.org/abs/2502.13766
- **What it does:** 144-country, 6-region multimodal cultural benchmark (728 cultural facets, 6,857 UNESCO-ICH images); 20 LVLMs + 11 LLMs evaluated. Finding: models know **tangible > intangible** culture and broad origins > nuanced understanding.
- **Relevance to us:** Lets me **predict which cultural categories Stage 1 will struggle on (RQ3)** — intangible/ritual concepts likely hardest — before we even run the breakdown.
- **Cite for:** the tangible-vs-intangible finding; prior on category-level difficulty patterns.

### 14. Evaluation of Cultural Competence of VLMs (Yadav et al., 2025) — `yadav2025cultural`
arXiv: https://arxiv.org/abs/2505.22793
- **What it does:** Position paper arguing cultural evaluation of VLMs should draw on visual-culture theory (semiotics, cultural/visual studies); proposes 5 frameworks of cultural dimensions for annotating images.
- **Relevance to us:** Theoretical backing for how Nandita's **cultural taxonomy** and my **VQA question set** are constructed and what "cultural accuracy" means in our human eval.
- **Cite for:** justification of taxonomy/rubric design; framing cultural annotation as principled rather than ad hoc.

### 15. Exploring Visual Culture Awareness in GPT-4V (2024) — `2024gpt4vculture` *(optional)*
arXiv: https://arxiv.org/abs/2402.06015
- **What it does:** Probing study of cultural blind spots in a frontier VLM.
- **Relevance to us:** Supplementary support that even the strongest generic VLMs miss cultural cues.
- **Cite for:** extra evidence in the intro motivation, if space allows.

---

## How to deploy these in the preliminary paper

**Intro / motivation (the bottleneck):** CLIP `radford2021clip` → generic encoders discard culture, shown concretely by CultureCLIP `huang2025cultureclip`, MaRVL `liu2021marvl`, CVQA `romero2024cvqa`, and GPT-4V probing `2024gpt4vculture`.

**Method — Stage 1 design:** VQA-before-captioning from CIC `yun2024cic`; visual instruction tuning from LLaVA `liu2024llava`; model = SmolVLM `marafioti2025smolvlm` (SigLIP encoder `zhai2023siglip`); fine-tuning = LoRA `hu2022lora`; baseline = Qwen2.5-VL `qwen2025qwen25vl`.

**Bridge to Stage 2 (novel contribution):** culture-aware retrieval helps captioning — RAVENEA `yuan2025ravenea`.

**Evaluation — RQ2/RQ3:** per-category/per-region methodology from CulturalVQA `nayak2024culturalvqa`; category-difficulty priors from GIMMICK `schneider2025gimmick`; taxonomy/rubric grounding from `yadav2025cultural`.

**Note:** Verify final venue/version for the 2025 preprints (CultureCLIP, RAVENEA, GIMMICK, Yadav) before the camera-ready — some are arXiv-only as of now.
