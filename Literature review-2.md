# Stage 1 Related Work — Prose Draft (Romit)

## Vision and Cultural Grounding (Stage 1\)

The systemic cultural blind spots of generic vision–language models (VLMs) trained on predominantly Western, web-scale image–text distributions are well documented. The contrastive pretraining paradigm introduced by Radford et al. (2021) with CLIP yields cross-modal representations that are inherently skewed; because supervision relies on English-dominated web scraping, concepts outside this linguistic and geographic distribution are marginalized.  
Liu et al. (2021) formalized the consequences of this data bias through MaRVL, demonstrating that cross-lingual and cross-cultural transfer in visual reasoning falls drastically short of English-baseline performance. Similarly, Romero et al. (2024) introduced CVQA—a benchmark co-created with native speakers across 30 countries—confirming that culturally diverse visual question answering remains an open challenge for state-of-the-art models.  
This representational bottleneck is precisely what the winning CICIL system uncovers but leaves unresolved: generic visual encoders strip out nuanced cultural semantics before the downstream translation phase even begins. Stage 1 of our proposed pipeline addresses this limitation, allowing us to evaluate whether mitigating this initial information loss yields measurable gains in end-to-end caption quality (**RQ1**).

### Interventions at the Vision Layer

Prior attempts to remedy these representational gaps typically intervene at the vision encoder level. For instance, Huang et al. (2025) developed CulTwin, a synthetic dataset featuring visually similar but culturally distinct image–caption triplets, and fine-tuned CLIP using a tailored contrastive objective to achieve up to 5.49% improvement in fine-grained cultural concept recognition. While we share their diagnostic premise, our pipeline intervenes at a different architectural stratum.  
Contrastive encoder retraining requires massive, curated image datasets. Given our constraint of roughly 50 pilot examples per target language, extensive visual pretraining is computationally impractical. Consequently, we freeze the vision encoder of SmolVLM and inject cultural signals into the autoregressive language decoder via structured prompting and parameter-efficient fine-tuning. This architecture trades encoder-level cultural alignment for data-efficient decoder-level adaptation.

### Cultural Question Answering and Taxonomies

A complementary line of research utilizes cross-examinations of visual content prior to generating captions. The CIC framework proposed by Yun and Kim (2024) generates category-specific queries, extracts cultural visual elements via visual question answering (VQA), and leverages a large language model to synthesize culturally descriptive output. Though human evaluations show that CIC outperforms generic pretraining baselines across diverse cultural cohorts, our work departs from this framework in three ways — **Model Adaptation** (We fine-tune the vision–language architecture rather than querying off-the-shelf models), **Geographic and Linguistic Specialization** (We focus on the endangered indigenous languages of the Americas), and **Downstream Integration** (Rather than treating the generated Spanish description as a terminal output, we leverage it as a descriptive vehicle for structured cultural annotations that drive downstream retrieval). Furthermore, instead of relying on a generalized visual inventory, our specific question categories—ceremony, material culture, landscape, and kinship—are derived directly from the ethnographically grounded taxonomy detailed in **\[Section X\]**.

### Retrieval-Augmented Visual Grounding

Recent literature suggests that cultural annotations yield the highest utility when coupled with retrieval mechanisms. The RAVENEA benchmark (Li et al., 2026\) formalizes culture-informed image captioning, establishing that culture-aware retrieval improves lightweight VLMs by at least 3.2% on cultural VQA and 6.2% on cultural captioning.  
This finding provides empirical justification for our design architecture: cultural grounding must actively steer the retrieval engine rather than run parallel to it. However, while RAVENEA retrieves information over curated, encyclopedic text corpora, our pipeline constructs retrieval queries from the cultural concepts emitted by Stage 1's VQA module. This design couples the vision and translation phases that prior architectures treat as disjoint operations (**\[Section X\]**).

### Architectural Efficiency under Low-Resource Constraints

To accommodate resource constraints, our vision backbone prioritizes parameter efficiency and low-overhead fine-tuning. SmolVLM (Marafioti et al., 2025\) is a family of compact models (256M–2.2B) whose smallest variant runs in under 1GB; we fine-tune the 2B model, which remains tractable on a single T4 GPU within our $50 compute budget.  
Its architecture pairs a SigLIP vision encoder (Zhai et al., 2023\) with a compact language decoder. We freeze the SigLIP encoder and adapt the decoder using Low-Rank Adaptation (LoRA; Hu et al., 2022\) configured at rank *r \= 16* and scaling factor *α \= 32*. This frozen-encoder, adapted-decoder division underscores why the cultural signal must be introduced via prompting—the underlying visual features cannot be robustly re-aligned on 50 examples.  
As an experimental control, we evaluate an unadapted, off-the-shelf Qwen2.5-VL model (Qwen Team, 2025\) to isolate Stage 1’s contribution for **RQ1**. The acute data scarcity detailed in our exploratory analysis—such as Wixárika offering a mere 20 pilot pairs—further justifies this parameter-efficient architecture and frames our investigation into low-resource scalability (**RQ2**).

### Mapping Structural and Conceptual Failure Modes

Finally, existing benchmarks offer a predictive framework for identifying which cultural dimensions present the steepest grounding hurdles. Using CulturalVQA, Nayak et al. (2024) demonstrated that model competence varies sharply across cultural facets, with clothing, rituals, and traditions grounded more reliably than food and drink.  
Schneider et al. (2025) corroborated this asymmetric performance profile with GIMMICK, reporting that current models grasp material cultural artifacts more readily than intangible, lived practices.  
These documented performance disparities provide us with both a structural template for reporting ChrF++ scores stratified by cultural category, and an empirical prior for where Stage 1 is vulnerable—namely, on intangible and ceremonial concepts (**RQ3**). These insights also shape our human-evaluation rubric, which formalizes its definitions of cultural accuracy by anchoring them in the cultural-theoretic frameworks proposed by Yadav et al. (2025).

## References (this section)

* Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022\.* https://openreview.net/forum?id=nZeVKeeFYf9  
* Huang, Y., Fan, Z., He, Z., Polisetty, S., Li, W., and Fung, Y. R. (2025). CultureCLIP: Empowering CLIP with Cultural Awareness through Synthetic Images and Contextualized Captions. *COLM 2025\.* https://openreview.net/forum?id=cWVpXWARbt  
* Li, J., Yuan, Y., Li, W., et al. (2026). RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding. *ICLR 2026\.* https://openreview.net/forum?id=4zAbkxQ23i  
* Liu, F., Bugliarello, E., Ponti, E. M., Reddy, S., Collier, N., and Elliott, D. (2021). Visually Grounded Reasoning across Languages and Cultures (MaRVL). *EMNLP 2021\.* https://aclanthology.org/2021.emnlp-main.818/  
* Marafioti, A., et al. (2025). SmolVLM: Redefining Small and Efficient Multimodal Models. *arXiv:2504.05299.* https://arxiv.org/abs/2504.05299  
* Nayak, S., Jain, K., Awal, R., Reddy, S., van Steenkiste, S., Hendricks, L. A., Stanczak, K., and Agrawal, A. (2024). Benchmarking Vision Language Models for Cultural Understanding (CulturalVQA). *EMNLP 2024\.* https://aclanthology.org/2024.emnlp-main.329/  
* Qwen Team (2025). Qwen2.5-VL Technical Report. *arXiv:2502.13923.* https://arxiv.org/abs/2502.13923  
* Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP). *ICML 2021\.* https://proceedings.mlr.press/v139/radford21a.html  
* Romero, D., et al. (2024). CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark. *NeurIPS 2024 Datasets & Benchmarks.* https://proceedings.neurips.cc/paper\_files/paper/2024/hash/1568882ba1a50316e87852542523739c-Abstract-Datasets\_and\_Benchmarks\_Track.html  
* Schneider, F., Holtermann, C., Biemann, C., and Lauscher, A. (2025). GIMMICK: Globally Inclusive Multimodal Multitask Cultural Knowledge Benchmarking. *Findings of ACL 2025\.* https://aclanthology.org/2025.findings-acl.500/  
* Yadav, S., Tilton, L., Antoniak, M., Arnold, T., Li, J., et al. (2025). Cultural Evaluations of Vision-Language Models Have a Lot to Learn from Cultural Theory. *arXiv:2505.22793.* https://arxiv.org/abs/2505.22793  
* Yun, Y. and Kim, J. (2024). CIC: A Framework for Culturally-Aware Image Captioning. *IJCAI 2024\.* https://www.ijcai.org/proceedings/2024/180  
* Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP). *ICCV 2023\.* https://openaccess.thecvf.com/content/ICCV2023/html/Zhai\_Sigmoid\_Loss\_for\_Language\_Image\_Pre-Training\_ICCV\_2023\_paper.html

1\. Stage 1 output — 500 Spanish descriptions (the deliverable, not a score yet). For each of 5 languages × 50 dev images, we produced two versions:

* Generic (\*\_generic\_ollama.jsonl) — plain caption, no cultural prompting → the RQ1 control.  
* Cultural-VQA (\*\_cultural-vqa\_ollama.jsonl) — the ask-then-synthesize output, richer and culturally grounded, plus per-category annotations (ceremony / material culture / landscape / kinship) → the RQ1 treatment.

These are Spanish intermediates. They are not scored — there's no ChrF++ for our system yet, because ChrF++ is measured against the target-language captions, and translating Spanish → target is Stage 2's job. So we can't yet say whether cultural-VQA beats generic; that delta only appears after Stage 2\.  
2\. The only real metric we hold — the official baseline (reference point for the paper), recomputed with the official scorer:

| Guaraní | Wixárika | Nahuatl | Bribri | Maya |
| :---- | :---- | :---- | :---- | :---- |
| 20.82 | 17.77 | 11.53 | 7.57 | — (no MT baseline) |

That's the number our system has to beat.

**Stage 1: Preliminary Intermediate-Description Quality.**  
Stage 1: Preliminary Intermediate-Description Quality. Because the target languages are unknown to the authors, we evaluate Stage 1 on its Spanish intermediate output, which isolates the vision-and-grounding contribution from downstream translation error. We report ChrF++ (word\_order=2) against the gold Spanish reference on the Wixárika pilot (n=20), the only split with released Spanish captions, using deterministic decoding (temperature 0). An initial cultural-VQA implementation underperformed the generic baseline (16.2 vs. 21.0), which diagnostic analysis attributed to verbosity dilution: the synthesis step emitted long, hedged descriptions (mean 105 tokens vs. \~35) penalised by character-F against short references. Constraining synthesis to a single concise sentence (mean 25 tokens) closed this gap entirely, yielding parity with the baseline (20.9 vs. 21.0, Δ \= −0.15) while producing fewer internally inconsistent descriptions (0/20 vs. 2/20 naming multiple incompatible materials). We conclude that, on this proxy metric, concise cultural-VQA prompting matches the generic baseline in surface quality without degrading it; whether its structured cultural grounding yields end-to-end gains is deferred to the Stage 2 indigenous-language evaluation.  
Table:

| Stage 1 configuration | ChrF++ ↑ | mean length (tok) | material conflicts |
| :---- | :---- | :---- | :---- |
| Generic baseline | 21.0 | 35 | 2/20 |
| Cultural-VQA, verbose | 16.2 | 105 | — |
| Cultural-VQA, concise | 20.9 | 25 | 0/20 |

Wixárika pilot, n=20; qwen2.5-VL-7B, temperature 0 (deterministic). ChrF++ vs. gold Spanish.

**Data and Cultural taxonomy** 

A core contribution of our pipeline is the cultural taxonomy that drives Stage 1 of our system, the structured set of categories that tells the vision model what cultural elements to look for in an image before generating a Spanish description. We ground this taxonomy in three lines of prior work.

Hershcovich et al. (2022) provide the foundational framework for thinking about culture in NLP. They identify four broad elements where culture and language interact: linguistic form and style, objectives and values, common ground, and aboutness. Critically, they argue that cross-lingual NLP is insufficient in serving speakers of a language and requires also serving the cultural context those speakers inhabit. This directly motivates our work: generating captions in Guaraní or Bribri is not just a translation problem, but a cultural grounding problem. Unlike Hershcovich et al., who survey the problem space broadly across high-resource languages, our work applies these principles specifically to endangered indigenous languages of the Americas, where cultural stakes are particularly high given active revitalization efforts.

Liu, Gurevych & Korhonen (2025) build on this foundation with a fine-grained taxonomy of cultural elements grounded in anthropology and social-science literature, organized into three branches — ideational (concepts, knowledge, values, norms, and artifacts), linguistic (dialects, styles, and registers), and social (relationships, context, and communicative goals). Their survey of 127 \*CL papers organise existing NLP resources and methods against this taxonomy, revealing gaps in non-Western, endangered, and oral-language communities. While their survey spans text and multimodal resources alike, our contribution is to operationalize a cultural taxonomy as a set of visual question prompts that drive caption generation for indigenous languages — turning a descriptive framework into a generation component.

The CIC framework (2024) is the most direct precedent for our approach. CIC introduces a pipeline for culturally-aware image captioning that uses a VQA module to extract cultural elements from images across five categories: Architecture, Clothing, Food & Drink, Dance & Music, and Religion. These categories are validated through user surveys with participants from four cultural groups. CIC demonstrates that generic vision-language models miss cultural visual elements describing a ceremonial space as simply a "hallway" or "bed" exactly the bottleneck our Stage 1 module is designed to fix. Our work differs from CIC in two key ways: first, we target indigenous languages of the Americas rather than broadly defined global cultures; second, our cultural categories (ceremony, material culture, landscape, kinship) are derived from linguistic and ethnographic documentation of these specific communities rather than general visual taxonomies. This makes our taxonomy more precise but also more constrained in coverage.

Our pipeline's Stage 2 retrieval depends critically on the availability and quality of Spanish Indigenous parallel corpora. We draw on four data sources, each with distinct characteristics and limitations that directly affect system performance across languages.

Mager et al. (2021) present the findings of the AmericasNLP 2021 shared task, which introduced parallel datasets for ten indigenous languages including Bribri (\~7,500 pairs) and Wixárika (\~8,967 pairs). These corpora were assembled from diverse sources including governmental documents, educational materials, and linguistic fieldwork, and were used to train and evaluate machine translation systems across two tracks. This shared task is the primary source of our Bribri retrieval data. A key limitation relevant to our work is domain mismatch: the Bribri corpus draws heavily from formal and religious texts, while our task requires caption-style visual descriptions. As Mager et al. note, for the lowest-resource languages, corpus domain matters as much as corpus size. This directly informs RQ2 of our paper that the performance gap between Guaraní and Bribri may reflect domain mismatch as much as raw data quantity.

Ebrahimi et al. (2023) describe the AmericasNLP 2023 shared task, which expanded and updated parallel corpora for Guaraní and other languages, retaining the XNLI-based evaluation framework from 2021 while introducing new Chatino data. The Guaraní training set from this task forms the backbone of our \~53,000-pair retrieval corpus. Compared to Bribri and Wixárika, Guaraní benefits from substantially more parallel data, and some of its particularly community-sourced examples are closer in register to visual description. This corpus size advantage is why we expect Guaraní to show the strongest retrieval gains in our ablations over k ∈ {3, 5, 8}.

Data Analysis 

p2026-main/import json.py"

✓ Loaded  20 examples for Wixárika  (data/pilot/wixarika.jsonl)

✓ Loaded  50 examples for Guaraní  (data/dev/guarani/guarani.jsonl)

✓ Loaded  50 examples for Bribri  (data/dev/bribri/bribri.jsonl)

✓ Loaded  50 examples for Yucatec Maya  (data/dev/maya/maya.jsonl)

── SAMPLE ENTRY ────────────────────────────────────────────────────

{

  "id": "hch\_001",

  "filename": "images/wixarika/hch\_001.jpg",

  "split": "train",

  "culture": "wixarika",

  "spanish\_caption": "Los puentes colgantes estan en lugares de difícil acceso como cuando crecen los ríos y no haya riesgos en las comunidades pequeñas.",

  "language": "Wixárika",

  "iso\_lang": "hch",

  "target\_caption": "Ta kie muye háne ik+ pan+ w+ ta húyetá, kiekari yak+ m+ yeyeut+ tsie metá há me wayeneikak+ te m+ ka hé hauweni k+."

}

── TABLE 1: LOADED SET SIZES ───────────────────────────────────────

Language               \# Examples

\----------------------------------

Wixárika                       20

Guaraní                        50

Bribri                         50

Yucatec Maya                   50

── TABLE 2: RETRIEVAL CORPUS SIZES (from papers) ───────────────────

Language                  \# Pairs   Source

\-----------------------------------------------------------------

Guaraní                     53183   AmericasNLP 2023 \+ MultiScript30k

Bribri                       7506   AmericasNLP 2021

Wixárika                     8967   AmericasNLP 2021

Yucatec Maya              unknown   TBD

Nahuatl                   unknown   Py-Elotl 2025

Saved → figure1\_caption\_lengths.png

Saved → figure2\_corpus\_sizes.png

Saved → figure3\_length\_scatter.png

── CAPTION STATS SUMMARY ───────────────────────────────────────────

Language               \# captions   mean len    min    max

\----------------------------------------------------------

Wixárika                       20       13.7      6     26

Guaraní                        50       21.6      9     40

Bribri                         50       14.9      8     30

Yucatec Maya                   50       11.3      3     24

We conducted an exploratory analysis of the CICIL pilot and development sets across four target languages. Table 1 shows the number of available examples per language: Wixárika has only 20 pilot examples compared to 50 development examples for Guaraní, Bribri, and Yucatec Maya, reflecting its lower resource availability even within this already low-resource task setting.

Table 2 summarizes the retrieval corpus sizes available for Stage 2\. Guaraní is by far the best-resourced language with 53,183 parallel pairs (AmericasNLP 2023 \+ MultiScript30k), while Bribri (7,506) and Wixárika (8,967) are significantly more limited. Yucatec Maya and Nahuatl corpus sizes remain to be confirmed during data preparation.

Caption length analysis reveals notable variation across languages (Figure 1). Guaraní captions are longest on average (21.6 words, range 9–40), while Yucatec Maya captions are shortest (11.3 words, range 3–24). Because these languages differ substantially in morphological typology — a morphologically rich language may encode in a single word what another expresses in several — raw word counts are not directly comparable across languages, and these differences likely reflect tokenization and annotation conventions as much as caption content. This reinforces our use of character-level ChrF++ and cautions against comparing ChrF++ scores directly across languages.

**Retrieval, LLM prompting, Evaluation scripts**

Stage 2 of our pipeline depends on three bodies of prior work: retrieval-augmented generation for machine translation, in-context example selection strategies, and evaluation metrics for low-resource machine translation. We review each in turn.

Agrawal et al. (2023) provide the foundational study of in-context example selection for machine translation. They compare random sampling, BM25-based sparse retrieval, and a BM25 \+ n-gram-recall reranking strategy across multiple language pairs, finding that semantic similarity between the test sentence and retrieved examples correlates positively with translation quality. Critically, they also show that diversity among retrieved examples matters — a set of near-duplicate pairs does not help the model as much as a varied set. This directly motivates our design choice to index by cultural concept annotations from Stage 1 rather than by raw surface text similarity. Where Agrawal et al. work with high-resource language pairs and generic parallel corpora, our work extends this principle to indigenous languages of the Americas, where the retrieval corpus is far smaller and cultural relevance of examples matters as much as lexical similarity.

Merx et al. (2024) study few-shot LLM prompting for English-to-Mambai translation, a genuinely low-resource Austronesian language with roughly 200,000 speakers. They find that mixing TF-IDF-retrieved sentences with semantically embedded examples in the prompt substantially improves translation quality over random selection. However, they also document a sharp performance gap between in-domain and out-of-domain test sentences — a finding directly relevant to our pipeline, since the Bribri and Wixárika corpora drawn from formal and religious texts (Mager et al., 2021\) are likely out of domain relative to the visual caption style required by CICIL. Unlike Merx et al., who use a single retrieval corpus for all test instances, our pipeline indexes by cultural category, allowing us to retrieve examples that are both semantically and culturally close to each test image.

Reimers and Gurevych (2020) introduce the multilingual Sentence-BERT framework, extending monolingual sentence embeddings to 50+ languages via knowledge distillation. The paraphrase-multilingual-MiniLM-L12-v2 model used in our FAISS index is a direct product of this work, mapping sentences from any supported language into a shared 384-dimensional vector space optimized for semantic similarity. Shi et al. (2022) conduct a systematic study of cross-lingual retrieval with such multilingual encoders and find that dense retrieval consistently outperforms sparse BM25 for morphologically rich, low-resource language pairs — precisely the setting of our target languages. Their finding reinforces our decision to use embedding-based retrieval over keyword matching, which would fail entirely for languages with little lexical overlap with Spanish query text.

For evaluation, Kumar et al. (2026) conduct the most directly relevant comparative study, examining ChrF++ and BLEU on extremely low-resource Indic languages under both LLM and neural MT systems. They find that ChrF++ is more robust to common failure modes in low-resource MT — hallucination, source-copying, and morphological inflection variants — because it operates at the character level and rewards partial matches. However, they caution against relying solely on ChrF++, noting that BLEU catches a complementary set of errors. This informs both our choice of ChrF++ as the primary metric (consistent with the official CICIL task metric) and the design of our human evaluation rubric, which is calibrated to detect cases where character overlap inflates the ChrF++ score without reflecting genuine cultural accuracy.

**System Description**

Our Stage 2 system takes the culturally grounded Spanish description produced by Stage 1 as input and produces a caption in the target indigenous language. It has two components: a FAISS-based retrieval module and an LLM prompting module using Gemini 2.5 Flash.

*Retrieval Module.* We build a separate FAISS index for each target language using the parallel corpora described in Nandita's data section: \~53,183 pairs for Guaraní, \~7,506 for Bribri, and \~8,967 for Wixárika. Each Spanish sentence in the corpus is encoded using paraphrase-multilingual-MiniLM-L12-v2 and indexed using FAISS IndexFlatIP (inner product similarity, equivalent to cosine similarity over normalized vectors). The key novelty of our retrieval design is that queries are formed not from the raw Stage 1 Spanish description but from the cultural concept annotations extracted by Stage 1 — for example, the detected categories "ceremony" and "traditional textile" are used to construct the query embedding. This means retrieval targets cultural relevance rather than surface lexical similarity, which is the core design difference between our system and the Gators baseline (Dhawan et al., 2026\) that uses standard text-to-text retrieval. At retrieval time, we retrieve k nearest neighbors from the index, where k ∈ {3, 5, 8} is treated as an ablation hyperparameter. Retrieved pairs are ranked by cosine similarity and the top-k Spanish–indigenous pairs are passed to the prompting module.

*LLM Prompting Module.* We use Gemini 2.5 Flash via the Gemini API as the Stage 2 translator. The prompt is structured as follows: a system instruction specifying the target language and cultural grounding task, followed by the k retrieved Spanish–indigenous parallel pairs as few-shot demonstrations, followed by the Stage 1 Spanish description as the source to translate. The few-shot examples are ordered by similarity score, with the most similar example placed closest to the source sentence, following the findings of Agrawal et al. (2023) that proximity of the most relevant example to the query improves output quality. No fine-tuning of Gemini is performed; all adaptation is through in-context learning. As a stretch goal, we will additionally experiment with fine-tuned NLLB-200 (600M) as an alternative translator for Guaraní, where the larger parallel corpus makes fine-tuning feasible.

*Evaluation Scripts.* We evaluate all system outputs using ChrF++ as the primary metric, consistent with the official CICIL shared task. Our evaluation scripts compute per-language ChrF++ scores across all system variants: the full pipeline, a generic VLM baseline (no cultural fine-tuning in Stage 1), an ablation removing cultural annotation-based retrieval (replacing it with raw text similarity), and the retrieval depth ablation over k ∈ {3, 5, 8}. Results are visualized as a per-language ChrF++ bar chart comparing all variants and a retrieval depth ablation curve. Additionally, we produce a language-by-cultural-category heatmap showing ChrF++ broken down by the cultural categories detected in Stage 1, which directly addresses RQ3 — which cultural categories are hardest to ground correctly. For the human evaluation, we score \~30 randomly sampled captions on a 3-point cultural accuracy rubric (0 \= culturally incorrect or missing, 1 \= partially correct, 2 \= culturally accurate and complete), with annotation performed by native or heritage speakers where possible, supplemented by consultation with the linguistic documentation sources used in the taxonomy.

**References for this section**

Agrawal, S., Zhou, C., Lewis, M., Zettlemoyer, L., and Ghazvininejad, M. (2023). In-context examples selection for machine translation. In Findings of ACL 2023, pp. 8857–8873. [https://aclanthology.org/2023.findings-acl.564](https://aclanthology.org/2023.findings-acl.564)

Kumar, S. et al. (2026). Evaluating extremely low-resource machine translation: A comparative study of ChrF++ and BLEU metrics. [arXiv:2602.17425.](https://arxiv.org/abs/2602.17425)

Merx, R., Mahmudi, A., Langford, K., de Araujo, L. A., and Vylomova, E. (2024). Low-resource machine translation through retrieval-augmented LLM prompting: A study on the Mambai language. In Proceedings of EURALI @ LREC-COLING 2024\. [https://aclanthology.org/2024.eurali-1.1](https://aclanthology.org/2024.eurali-1.1)

Reimers, N. and Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. In Proceedings of EMNLP 2020, pp. 4512–4525. [https://arxiv.org/abs/2004.09813](https://arxiv.org/abs/2004.09813)

Shi, P., Zhang, R., Bai, H., and Lin, J. (2022). Cross-lingual retrieval augmented prompt for low-resource languages. [arXiv:2212.09651.](https://arxiv.org/abs/2212.09651)

### **Evaluation Methodology**

#### **Metric Choice for Low-Resource Caption Evaluation**

Evaluating generated text in indigenous languages presents a fundamental measurement problem: the languages we target are morphologically rich and word boundaries do not correspond to meaningful units in the way they do in English. Popović (2017) addressed precisely this limitation with ChrF++, a character n-gram F-score augmented with word unigrams and bigrams. By operating at the character level, ChrF++ awards partial credit for morphologically inflected forms that BLEU would penalize as complete mismatches, and its language-independent formulation removes the tokenization dependency that has repeatedly been shown to distort cross-system comparisons. This metric is the official scoring function of the CICIL shared task (Bui et al., 2026\) and consequently anchors every automatic evaluation in our pipeline.

However, sole reliance on ChrF++ is not without risk. Kumar et al. (2026) conduct a comparative study of ChrF++ and BLEU across extremely low-resource Indic languages under both LLM and neural MT systems, demonstrating that character-level metrics can obscure specific translation artifacts including hallucination, source-copying, and repetitive outputs — failure modes to which few-shot LLM translators such as our Gemini 2.5 Flash Stage 2 are particularly susceptible. They recommend jointly interpreting ChrF++ with BLEU because their divergence patterns diagnose linguistic issues that neither metric captures alone. We adopt this recommendation as a secondary safeguard: while ChrF++ remains our headline metric to preserve comparability with the shared task leaderboard, we additionally report BLEU alongside it to flag potential artifacts, and we design our human evaluation rubric to catch cases where character overlap inflates ChrF++ without genuine cultural fidelity.

#### **Per-Language and Per-Category Reporting**

Muhammad et al. (2025) present the findings of SemEval-2025 Task 11 on multilingual emotion detection across thirty-two languages spanning seven language families. Although their task and metrics differ from ours — they use macro F1 and Pearson correlation rather than ChrF++ — their reporting structure provides a directly applicable template. They present per-language results tables in which top-performing systems are shown alongside two baselines (a majority-class floor and a fine-tuned RoBERTa reference), enabling readers to interpret each system's contribution against both a naive and a reasonable-effort bar. Their finding that performance gaps between high-resource and low-resource languages persist even under strong systems provides empirical grounding for our RQ2, which asks whether cultural-aware vision encoding disproportionately benefits data-starved languages such as Wixárika and Bribri. Our per-language ChrF++ bar chart mirrors their reporting layout: for each target language we present our full pipeline, the Gators reference (Dhawan et al., 2026), a generic-VLM baseline, and two ablations, allowing per-language contributions to be read at a glance.

The AmericasNLP 2024 shared task findings (Ebrahimi et al., 2024\) provide the closest methodological precedent for our specific setting. That task used ChrF++ as its official metric across eleven indigenous American languages, including Bribri, one of our four target languages. Beyond confirming metric conventions, Ebrahimi et al. present a two-part human evaluation of Bribri outputs from the best-performing system: a quantitative rating of meaning and fluency, followed by a qualitative error analysis performed by native and heritage speakers. Their protocol demonstrates that meaningful human evaluation of caption quality is tractable even without full fluency across all evaluators, provided the rubric is well-scoped and error categories are pre-defined. We follow this design: our 3-point cultural accuracy rubric is scoped narrowly to a single dimension our team can reliably annotate given our cultural taxonomy, and we supplement quantitative scores with a qualitative error analysis on failure cases to identify systematic weaknesses that ChrF++ cannot surface.

#### **Human Evaluation Design**

Kasai et al. (2022) introduce THumB, a transparent rubric-based human evaluation protocol for image captioning that separates the caption's precision (are the elements it mentions actually in the image?) from its recall (does it cover the salient elements?), while additionally scoring fluency, conciseness, and inclusive language. Two features of their design directly inform ours. First, their rubrics were iteratively developed through pilot annotation on a small held-out sample before scaling to the full evaluation set, a protocol we adopt for our 30-caption human evaluation: we pilot the rubric on approximately five captions per language, refine the anchor definitions of the 3-point scale, and only then proceed with the full annotation. Second, they report Cohen's κ for inter-annotator agreement (0.86 for precision, 0.82 for recall), establishing a concrete reliability standard against which our own agreement statistics can be compared. Our rubric departs from THumB in two respects: we replace the precision–recall decomposition with a unified cultural accuracy dimension appropriate for our research question, and our evaluators consult the ethnographic documentation underlying our taxonomy rather than relying purely on visual inspection, since indigenous cultural elements are not always identifiable to non-community members from the image alone.

Finally, our evaluation is situated within the shared task defined by Bui et al. (2026), whose findings paper establishes the official CICIL dataset, per-language ChrF++ leaderboard, and the top-performing Gators system (Dhawan et al., 2026\) against which all our comparisons are made. Their diagnosis that the vision stage is the pipeline bottleneck — an observation left as future work in the shared task — is the specific gap our Stage 1 targets, and their per-language baseline scores (20.82 ChrF++ for Guaraní, 17.77 for Wixárika, 11.53 for Nahuatl, 7.57 for Bribri) constitute the reference points our results tables report against.

### **Evaluation Setup**

All automatic scoring is performed using sacrebleu with the ChrF++ configuration matching the official CICIL scorer (word\_order=2), ensuring comparability with the shared task leaderboard. Scores are computed at the corpus level over each target language's development set, and additionally over per-cultural-category subsets defined by our taxonomy to support the language-by-category heatmap addressing RQ3. Given the small size of per-category cells (typically 5–15 examples), we plan to report bootstrap 95% confidence intervals over 1,000 resamples alongside point estimates. For the retrieval-depth ablation, ChrF++ is computed at k ∈ {3, 5, 8} per language with all other pipeline components held fixed. Human evaluation is performed on a stratified sample of 30 captions, drawn proportional to per-language dev set size with a minimum of five captions per language, after final system outputs are frozen. Two annotators score each caption independently using the 3-point cultural accuracy rubric described above; disagreements are resolved through discussion, and inter-annotator agreement is reported as Cohen's κ following the protocol of Kasai et al. (2022).

**References for this section:**

Bui, T., et al. (2026). Findings of the AmericasNLP 2026 Shared Task on Cultural Image Captioning for Indigenous Languages. In *Proceedings of the AmericasNLP Workshop 2026*.

Ebrahimi, A., de Gibert, O., Vázquez, R., Coto-Solano, R., Denisov, P., Pugh, R., Mager, M., Oncevay, A., Chiruzzo, L., von der Wense, K., and Rijhwani, S. (2024). Findings of the AmericasNLP 2024 Shared Task on Machine Translation into Indigenous Languages. In *Proceedings of the 4th Workshop on Natural Language Processing for Indigenous Languages of the Americas (AmericasNLP 2024\)*, pp. 236–246.

Kasai, J., Sakaguchi, K., Dunagan, L., Morrison, J., Le Bras, R., Choi, Y., and Smith, N. A. (2022). Transparent Human Evaluation for Image Captioning. In *Proceedings of NAACL 2022*.

Kumar, S., et al. (2026). Evaluating Extremely Low-Resource Machine Translation: A Comparative Study of ChrF++ and BLEU Metrics. arXiv:2602.17425.

Muhammad, S. H., Ousidhoum, N., Abdulmumin, I., et al. (2025). SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. In *Proceedings of SemEval-2025*.

Popović, M. (2017). chrF++: words helping character n-grams. In *Proceedings of the Second Conference on Machine Translation*, pp. 612–618.

