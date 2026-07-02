# Stage 1 Related Work — Prose Draft (Romit)

## Vision and Cultural Grounding (Stage 1)

The systemic cultural blind spots of generic vision–language models (VLMs) trained on predominantly Western, web-scale image–text distributions are well documented. The contrastive pretraining paradigm introduced by Radford et al. (2021) with CLIP yields cross-modal representations that are inherently skewed; because supervision relies on English-dominated web scraping, concepts outside this linguistic and geographic distribution are marginalized.

Liu et al. (2021) formalized the consequences of this data bias through MaRVL, demonstrating that cross-lingual and cross-cultural transfer in visual reasoning falls drastically short of English-baseline performance. Similarly, Romero et al. (2024) introduced CVQA—a benchmark co-created with native speakers across 30 countries—confirming that culturally diverse visual question answering remains an open challenge for state-of-the-art models.

This representational bottleneck is precisely what the winning CICIL system (Dhawan et al., 2026) uncovers but leaves unresolved: its two-stage pipeline produces the Spanish intermediate with a generic Qwen2.5-VL encoder, so cultural semantics are stripped out before the downstream translation phase even begins. Stage 1 of our proposed pipeline addresses this limitation, allowing us to evaluate whether mitigating this initial information loss yields measurable gains in end-to-end caption quality (**RQ1**).

### Interventions at the Vision Layer

Prior attempts to remedy these representational gaps typically intervene at the vision encoder level. For instance, Huang et al. (2025) developed CulTwin, a synthetic dataset featuring visually similar but culturally distinct image–caption triplets, and fine-tuned CLIP using a tailored contrastive objective to achieve up to 5.49% improvement in fine-grained cultural concept recognition. While we share their diagnostic premise, our pipeline intervenes at a different architectural stratum.

Contrastive encoder retraining requires massive, curated image datasets. Given our constraint of roughly 50 pilot examples per target language, extensive visual pretraining is computationally impractical. Consequently, we freeze the vision encoder of SmolVLM and inject cultural signals into the autoregressive language decoder via structured prompting and parameter-efficient fine-tuning. This architecture trades encoder-level cultural alignment for data-efficient decoder-level adaptation.

### Cultural Question Answering and Taxonomies

A complementary line of research utilizes cross-examinations of visual content prior to generating captions. The CIC framework proposed by Yun and Kim (2024) generates category-specific queries, extracts cultural visual elements via visual question answering (VQA), and leverages a large language model to synthesize culturally descriptive output. Though human evaluations show that CIC outperforms generic pretraining baselines across diverse cultural cohorts, our work departs from this framework in three ways — Model Adaptation (We fine-tune the vision–language architecture rather than querying off-the-shelf models), Geographic and Linguistic Specialization (We focus on the endangered indigenous languages of the Americas), and Downstream Integration (Rather than treating the generated Spanish description as a terminal output, we leverage it as a descriptive vehicle for structured cultural annotations that drive downstream retrieval). Furthermore, instead of relying on a generalized visual inventory, our specific question categories—ceremony, material culture, landscape, and kinship—are derived directly from the ethnographically grounded taxonomy detailed in **[Section X]**.

### Retrieval-Augmented Visual Grounding

Recent literature suggests that cultural annotations yield the highest utility when coupled with retrieval mechanisms. The RAVENEA benchmark (Li et al., 2026) formalizes culture-informed image captioning, establishing that culture-aware retrieval improves lightweight VLMs by at least 3.2% on cultural VQA and 6.2% on cultural captioning.

This finding provides empirical justification for our design architecture: cultural grounding must actively steer the retrieval engine rather than run parallel to it. However, while RAVENEA retrieves information over curated, encyclopedic text corpora, our pipeline constructs retrieval queries from the cultural concepts emitted by Stage 1's VQA module. This design couples the vision and translation phases that prior architectures treat as disjoint operations (**[Section X]**).

### Architectural Efficiency under Low-Resource Constraints

To accommodate resource constraints, our vision backbone prioritizes parameter efficiency and low-overhead fine-tuning. SmolVLM (Marafioti et al., 2025) is a family of compact models (256M–2.2B) whose smallest variant runs in under 1GB; we fine-tune the 2B model, which remains tractable on a single T4 GPU within our $50 compute budget.

Its architecture pairs a SigLIP vision encoder (Zhai et al., 2023) with a compact language decoder. We freeze the SigLIP encoder and adapt the decoder using Low-Rank Adaptation (LoRA; Hu et al., 2022) configured at rank $r = 16$ and scaling factor $\alpha = 32$. This frozen-encoder, adapted-decoder division underscores why the cultural signal must be introduced via prompting—the underlying visual features cannot be robustly re-aligned on 50 examples.

As an experimental control, we evaluate an unadapted, off-the-shelf Qwen2.5-VL model (Qwen Team, 2025) to isolate Stage 1’s contribution for **RQ1**. The acute data scarcity detailed in our exploratory analysis—such as Wixárika offering a mere 20 pilot pairs—further justifies this parameter-efficient architecture and frames our investigation into low-resource scalability (**RQ2**).

### Mapping Structural and Conceptual Failure Modes

Finally, existing benchmarks offer a predictive framework for identifying which cultural dimensions present the steepest grounding hurdles. Using CulturalVQA, Nayak et al. (2024) demonstrated that model competence varies sharply across cultural facets, with clothing, rituals, and traditions grounded more reliably than food and drink.

Schneider et al. (2025) corroborated this asymmetric performance profile with GIMMICK, reporting that current models grasp material cultural artifacts more readily than intangible, lived practices.

These documented performance disparities provide us with both a structural template for reporting ChrF++ scores stratified by cultural category, and an empirical prior for where Stage 1 is vulnerable—namely, on intangible and ceremonial concepts (**RQ3**). These insights also shape our human-evaluation rubric, which formalizes its definitions of cultural accuracy by anchoring them in the cultural-theoretic frameworks proposed by Yadav et al. (2025).

---

## References (this section)

- Dhawan, A., et al. (2026). Retrieval-Augmented Long-Context Translation for Cultural Image Captioning: Gators Submission for AmericasNLP 2026 Shared Task. *Proceedings of the Sixth Workshop on NLP for Indigenous Languages of the Americas (AmericasNLP); arXiv:2605.20626.* https://arxiv.org/abs/2605.20626
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022.* https://openreview.net/forum?id=nZeVKeeFYf9
- Huang, Y., Fan, Z., He, Z., Polisetty, S., Li, W., and Fung, Y. R. (2025). CultureCLIP: Empowering CLIP with Cultural Awareness through Synthetic Images and Contextualized Captions. *COLM 2025.* https://openreview.net/forum?id=cWVpXWARbt
- Li, J., Yuan, Y., Li, W., et al. (2026). RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding. *ICLR 2026.* https://openreview.net/forum?id=4zAbkxQ23i
- Liu, F., Bugliarello, E., Ponti, E. M., Reddy, S., Collier, N., and Elliott, D. (2021). Visually Grounded Reasoning across Languages and Cultures (MaRVL). *EMNLP 2021.* https://aclanthology.org/2021.emnlp-main.818/
- Marafioti, A., et al. (2025). SmolVLM: Redefining Small and Efficient Multimodal Models. *arXiv:2504.05299.* https://arxiv.org/abs/2504.05299
- Nayak, S., Jain, K., Awal, R., Reddy, S., van Steenkiste, S., Hendricks, L. A., Stanczak, K., and Agrawal, A. (2024). Benchmarking Vision Language Models for Cultural Understanding (CulturalVQA). *EMNLP 2024.* https://aclanthology.org/2024.emnlp-main.329/
- Qwen Team (2025). Qwen2.5-VL Technical Report. *arXiv:2502.13923.* https://arxiv.org/abs/2502.13923
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP). *ICML 2021.* https://proceedings.mlr.press/v139/radford21a.html
- Romero, D., et al. (2024). CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark. *NeurIPS 2024 Datasets & Benchmarks.* https://proceedings.neurips.cc/paper_files/paper/2024/hash/1568882ba1a50316e87852542523739c-Abstract-Datasets_and_Benchmarks_Track.html
- Schneider, F., Holtermann, C., Biemann, C., and Lauscher, A. (2025). GIMMICK: Globally Inclusive Multimodal Multitask Cultural Knowledge Benchmarking. *Findings of ACL 2025.* https://aclanthology.org/2025.findings-acl.500/
- Yadav, S., Tilton, L., Antoniak, M., Arnold, T., Li, J., et al. (2025). Cultural Evaluations of Vision-Language Models Have a Lot to Learn from Cultural Theory. *arXiv:2505.22793.* https://arxiv.org/abs/2505.22793
- Yun, Y. and Kim, J. (2024). CIC: A Framework for Culturally-Aware Image Captioning. *IJCAI 2024.* https://www.ijcai.org/proceedings/2024/180
- Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP). *ICCV 2023.* https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html