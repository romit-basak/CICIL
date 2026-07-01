# Stage 1 Reading List — Romit (Vision & VLM)

Curated for the preliminary paper's related-work section. My owned scope is **Stage 1: cultural VQA prompting module + LoRA fine-tuning of SmolVLM**, plus the generic-VLM baseline ablation.

The assignment requires **5–6 papers per person**. This list has **15 candidates** so I can read widely and pick the strongest 5–6 to actually write up. The two papers already assigned to me in the proposal are flagged **[ASSIGNED]**.

Each entry has a one-line note on *why it matters for Stage 1*. When I summarize a paper in the report, I must articulate how it relates to our work (similar/different, foundation/baseline).

---

## A. Culturally-aware vision & captioning (the core of my contribution)

1. **CultureCLIP: Empowering CLIP with Cultural Awareness through Synthetic Images and Contextualized Captions** — Huang et al., COLM 2025. **[ASSIGNED]**
   - arXiv: https://arxiv.org/abs/2507.06210
   - Direct blueprint for making the vision side culturally aware via contrastive fine-tuning on cultural image–caption pairs; informs why we don't just rely on a generic encoder.

2. **CIC: A Framework for Culturally-Aware Image Captioning** — Yun & Kim, IJCAI 2024. **[ASSIGNED]**
   - arXiv: https://arxiv.org/abs/2402.05374 · IJCAI: https://www.ijcai.org/proceedings/2024/180
   - The conceptual ancestor of our cultural VQA module: generate cultural-category questions → extract cultural elements via VQA → caption with an LLM. This *is* our Stage 1 design pattern.

3. **RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding** — Yuan et al., 2025.
   - arXiv: https://arxiv.org/abs/2505.14462
   - Defines culture-informed image captioning (cIC) and shows culture-aware retrieval gives +11% on cIC. Bridges my Stage 1 cultural annotations to Mehek's Stage 2 retrieval — strongest external evidence that our coupling should work.

---

## B. Models I use & efficient fine-tuning (methodology grounding)

4. **SmolVLM: Redefining Small and Efficient Multimodal Models** — Marafioti et al., 2025.
   - arXiv: https://arxiv.org/abs/2504.05299
   - The model I fine-tune. Needed to justify the 2B choice on a $50 / T4 budget and to cite architecture (SigLIP encoder + SmolLM decoder).

5. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR 2022.
   - arXiv: https://arxiv.org/abs/2106.09685
   - The fine-tuning method (rank 16, α=32). Foundational citation for why we can adapt the decoder cheaply while freezing the vision encoder.

6. **Sigmoid Loss for Language Image Pre-Training (SigLIP)** — Zhai et al., ICCV 2023.
   - arXiv: https://arxiv.org/abs/2303.15343
   - SmolVLM's frozen vision encoder. Lets me explain exactly what stays frozen and why the cultural signal must enter through the decoder + VQA prompting, not the encoder.

7. **Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)** — Liu et al., CVPR 2024.
   - arXiv: https://arxiv.org/abs/2310.03744
   - Our Stage 1 backup model if SmolVLM underperforms. Also the reference recipe for visual instruction tuning that our VQA prompting builds on.

8. **Qwen2.5-VL Technical Report** — Qwen Team, 2025.
   - arXiv: https://arxiv.org/abs/2502.13923
   - Our generic, off-the-shelf VLM baseline for the Stage 1 ablation (isolating the contribution of cultural fine-tuning).

9. **Learning Transferable Visual Models from Natural Language Supervision (CLIP)** — Radford et al., ICML 2021.
   - arXiv: https://arxiv.org/abs/2103.00020
   - Foundational contrastive VLM that everything above descends from; needed to frame *why* generic encoders discard culture (the bottleneck we target).

---

## C. Cultural / multilingual multimodal benchmarks & evaluation (supports RQ3 — which cultural categories are hardest)

10. **Benchmarking Vision Language Models for Cultural Understanding (CulturalVQA)** — Nayak et al., EMNLP 2024.
    - arXiv: https://arxiv.org/abs/2407.10920 · ACL: https://aclanthology.org/2024.emnlp-main.329/
    - Geo-diverse cultural VQA across 5 facets (traditions, rituals, food, drink, clothing); their per-facet/per-region gaps directly motivate our per-cultural-category ChrF++ breakdown.

11. **CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark** — Romero et al., NeurIPS 2024 (Datasets & Benchmarks, Oral).
    - arXiv: https://arxiv.org/abs/2406.05967
    - Shows current MLLMs struggle on culturally-grounded VQA in low-resource languages — the exact failure mode our cultural VQA module aims to mitigate.

12. **Visually Grounded Reasoning across Languages and Cultures (MaRVL)** — Liu et al., EMNLP 2021 (Best Long Paper).
    - ACL: https://aclanthology.org/2021.emnlp-main.818/
    - Early evidence that vision-language models trained on Western-centric data transfer poorly across cultures; foundational motivation for RQ1/RQ2.

13. **GIMMICK — Globally Inclusive Multimodal Multitask Cultural Knowledge Benchmarking** — Schneider et al., 2025.
    - arXiv: https://arxiv.org/abs/2502.13766
    - 144-country cultural benchmark finding models know "tangible > intangible" culture and broad-origin > nuance; useful prior for predicting which categories Stage 1 will struggle on (RQ3).

14. **Evaluation of Cultural Competence of Vision-Language Models** (a.k.a. *Cultural Evaluations… Have a Lot to Learn from Cultural Theory*) — Yadav et al., 2025.
    - arXiv: https://arxiv.org/abs/2505.22793
    - Position paper proposing cultural-theory frameworks for annotating cultural dimensions in images; helps justify how Nandita's taxonomy and my VQA question set are constructed and evaluated.

15. **Exploring Visual Culture Awareness in GPT-4V: A Comprehensive Probing** — 2024. *(optional / backup)*
    - arXiv: https://arxiv.org/abs/2402.06015
    - Probing study of cultural blind spots in a frontier VLM; supplementary support for the claim that generic VLMs discard cultural information.

---

## Suggested core 6 to write up first

If I only formally cover 6 in the preliminary paper, the strongest set covering motivation → method → evaluation is:
**#1 CultureCLIP, #2 CIC, #3 RAVENEA, #4 SmolVLM, #8 Qwen2.5-VL, #10 CulturalVQA.**
(LoRA #5 and SigLIP #6 are cited inline as method references even if not given full related-work paragraphs.)

---

## Published / canonical versions (cite these, not arXiv, where they exist)

| # | Paper | Venue | Official link |
|---|-------|-------|---------------|
| 1 | CultureCLIP | COLM 2025 | https://openreview.net/forum?id=cWVpXWARbt |
| 2 | CIC | IJCAI 2024 | https://www.ijcai.org/proceedings/2024/180 |
| 3 | RAVENEA | EMNLP 2025 Findings *(see note)* | https://wenyanli.org/publication/emnlp2025-revenea/ |
| 4 | SmolVLM | arXiv tech report (no peer-reviewed venue) | https://arxiv.org/abs/2504.05299 |
| 5 | LoRA | ICLR 2022 | https://openreview.net/forum?id=nZeVKeeFYf9 |
| 6 | SigLIP | ICCV 2023 | https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html |
| 7 | LLaVA-1.5 (Improved Baselines) | CVPR 2024 | https://cvpr.thecvf.com/virtual/2024/poster/29558 |
| 8 | Qwen2.5-VL | arXiv tech report (no peer-reviewed venue) | https://arxiv.org/abs/2502.13923 |
| 9 | CLIP | ICML 2021 (PMLR v139) | https://proceedings.mlr.press/v139/radford21a.html |
| 10 | CulturalVQA | EMNLP 2024 | https://aclanthology.org/2024.emnlp-main.329/ |
| 11 | CVQA | NeurIPS 2024 (Datasets & Benchmarks) | https://proceedings.neurips.cc/paper_files/paper/2024/hash/1568882ba1a50316e87852542523739c-Abstract-Datasets_and_Benchmarks_Track.html |
| 12 | MaRVL | EMNLP 2021 | https://aclanthology.org/2021.emnlp-main.818/ |
| 13 | GIMMICK | Findings of ACL 2025 | https://aclanthology.org/2025.findings-acl.500/ |
| 14 | Cultural Competence / Cultural Theory (Yadav et al.) | arXiv only (position paper) | https://arxiv.org/abs/2505.22793 |
| 15 | GPT-4V Visual Culture Probing | OpenReview (workshop) | https://openreview.net/forum?id=PCYP1Of3We |

**Notes / caveats:**
- **#3 RAVENEA:** the author's page tags it **EMNLP 2025 Findings**, but a later OpenReview PDF carries an **ICLR 2026** camera-ready header — confirm the canonical venue before citing. Its published numbers were also revised down to **+3.2% (cVQA) / +6.2% (cIC)** absolute from the earlier preprint's +6%/+11%.
- **#4 SmolVLM** and **#8 Qwen2.5-VL** are technical reports with no peer-reviewed venue — arXiv is the canonical citation.
- **#14** (`2505.22793`) is a position paper, arXiv-only. Do not confuse it with Yadav et al.'s separate NAACL 2025 Findings paper "Beyond Words: Exploring Cultural Value Sensitivity in Multimodal Models" — that's a different paper.
- For ACL Anthology and CVF entries, swap `/forum`/`/virtual` for the PDF/`.bib` link when generating BibTeX (the Anthology pages have a "Cite (ACL)" BibTeX button).
