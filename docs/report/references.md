# References

Numbering: [1]-[41] are the references of `docs/proposal_v5.pdf`, numbered as in the proposal so that citations in the proposal, `plan.md` and the code comments stay valid. [42]-[74] continue with the additions of `plan.md` section 18, in the order listed there. [75]-[77] are sources that the repo's own documents name (`docs/t1_week0.md`, `configs/compliance/red_flags.yaml`). Details are copied from the proposal text and `plan.md`; where they are missing or look incomplete the entry ends with a CHECK marker in square brackets, and every marker is listed in the "Open checks" section of `README.md`. Nothing in this file has been looked up outside the repository.

## Proposal references [1]-[41]

[1] E. Altman et al. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. NeurIPS 2023 Datasets and Benchmarks; arXiv:2306.16424.

[2] J. Blanuša et al. Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection. arXiv:2402.08593, 2024.

[3] R. Sharma. Toward Auditable Fraud Detection: Combining Graph Features, Model Explanations, and Agentic Case Investigation. arXiv:2607.19266, 2026.

[4] D. Palma, K. B. Kim, Z. Han, A. Dsouza, Z. Liu. Business Compromise Detection with Agentic AI and LLM-driven Knowledge Discovery. EMNLP 2026; arXiv:2609.32643.

[5] Shuster, Zaloof, Shabtai, Puzis. FAA Framework: A Large Language Model-Based Approach for Credit Card Fraud Investigations. arXiv:2506.11635, 2025. [CHECK: the proposal lists this entry as "FIA Framework" without authors; the title prefix and the author names are taken from docs/external_code_check.md, which says the arXiv title is "FAA Framework"; confirm the full title and the author initials on arXiv]

[6] D. Torres, W. Cheng, K. Hu. Explainable AML Triage with LLMs: Evidence Retrieval and Counterfactual Checks. arXiv:2604.19755, 2026.

[7] Bank for International Settlements. Project Aurora: the power of data, technology and collaboration to combat money laundering across institutions and borders. 2023.

[8] Z. Tian, Y. Ding, W. Qu et al. Towards Collaborative Anti-Money Laundering Among Financial Institutions. WWW 2025; arXiv:2502.19952.

[9] Ceydeli, Wang, Atasu. Federated Cross-Client Subgraph Pattern Detection. arXiv:2605.06433, 2026. [CHECK: docs/t1_week0.md records that the arXiv page is now titled "Federated Layer-Wise Exchange for Cross-Client Money Laundering Pattern Detection"; decide which title to print, and add the initials]

[10] D. Choudhury et al. BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design. arXiv:2508.21184, 2025.

[11] GraphDx: A Cost-Aware Knowledge-Enhanced Multi-Agent Framework for Sequential Diagnosis. ACL Findings 2026; arXiv:2607.15280. [CHECK: authors missing in the proposal]

[12] Perets et al. Zero-Shot Active Feature Acquisition via LLM-Elicitation. arXiv:2606.18933, 2026.

[13] T. LaRock et al. NOL*: reinforcement learning for exploring partially observed networks. Applied Network Science, 2020; arXiv:2001.07607.

[14] E. Debenedetti et al. Defeating Prompt Injections by Design (CaMeL). arXiv:2503.18813, 2025.

[15] Operational Evidence Gaps for LLMs in Fraud Detection and Trust-and-Safety Workflows (FORTE). arXiv:2607.13078, 2026. [CHECK: authors missing in the proposal]

[16] 31 CFR 1010.540: Voluntary information sharing among financial institutions. U.S. Code of Federal Regulations.

[17] FinCEN. Section 314(b) Fact Sheet. 12 June 2026.

[18] Monetary Authority of Singapore. Notice FSM-N02 and Guidelines on COSMIC. 2024. [CHECK: confirm the notice designation and the year; the proposal gives "Notice FSM-N02", while the red-flag corpus (configs/compliance/red_flags.yaml) uses the Guidelines to MAS Notice 626 (Banks), listed here as [77]]

[19] FATF. Money laundering red-flag indicator reports. [CHECK: no report titles, years or URLs in the proposal; list the reports actually used; configs/compliance/red_flags.yaml says the FATF PDFs could not be fetched (HTTP 403), so the v0 corpus does not contain FATF red flags yet]

[20] E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

[21] T. Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.

[22] lablab.ai hackathon submissions: Federated analytics for cross-bank AML; AEGIS autonomous financial-crime investigation. 2026. [CHECK: no authors or URLs in the proposal; docs/external_code_check.md gives the AEGIS code at github.com/Kunsh162007/AEGIS (a June 2026 submission, no licence); the federated-analytics submission has no further details anywhere in the repo]

[23] N. Hollmann, S. Müller, F. Hutter. Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering. NeurIPS 2023.

[24] Y. Cui, Z. Zhang, Q. Liu. LivingRAG: Augmenting Graph RAG with Experience. arXiv:2608.25960, 2026.

[25] M. Li, S. Miao, P. Li. Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation (SubgraphRAG). ICLR 2025; arXiv:2410.20724.

[26] X. He et al. G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering. NeurIPS 2024; arXiv:2402.07630.

[27] V. Gandhi et al. GRASP: GRanularity-Aware Search Policy for Agentic RAG. EMNLP 2026 Findings; arXiv:2607.10463.

[28] X. Tan, Y. Ma, X. Zhang. Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection (FinFRE-RAG). ACL 2026; arXiv:2512.13040.

[29] TypeSafe AI. Jev, a System One decision model (commercial; mentioned for context, not used). September 2026.

[30] SemIf: typed decisions from a frozen language model's answer probabilities (MIT). Open-source software, 2026. [CHECK: no authors or URL in the proposal]

[31] Laya: small non-autoregressive typed decision model with calibrated outputs (Apache-2.0). Open-source software, 2026. [CHECK: no authors or URL in the proposal]

[32] C. Dwork, A. Roth. The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science, 2014.

[33] K. Huang et al. Automated Hypothesis Validation with Agentic Sequential Falsifications (POPPER). ICML 2025; arXiv:2502.09858.

[34] S. Ouyang et al. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. ICLR 2026; arXiv:2509.25140.

[35] Y. Dou, S. Lian, S. Li. Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference. arXiv:2607.25018, 2026.

[36] C. Purcell et al. Split, Count, and Share: A Differentially Private Set Intersection Cardinality Estimation Protocol. PMLR vol. 216 (UAI 2023).

[37] R. Wang, A. Ramdas. False Discovery Rate Control with E-values. Journal of the Royal Statistical Society Series B, 2022.

[38] Y. Xu, W. Guo, Z. Wei. Selective Conformal Risk Control. arXiv:2512.12844, 2025.

[39] Non-exchangeable Conformal Prediction for Temporal Graph Neural Networks. arXiv:2507.02151, 2025. [CHECK: authors missing in the proposal]

[40] J. Deng. When Can Fraud Operations Authorize Automation? A Decision-Support Framework for Fresh Audit Evidence and Review Workload (FCAC). arXiv:2608.08577, 2026.

[41] S. Hu et al. Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward (InfoReasoner). ICML 2026; arXiv:2602.00845.

## Additions from plan.md section 18 [42]-[74]

[42] Egressy et al. Provably Powerful Graph Neural Networks for Directed Multigraphs (Multi-GNN). AAAI 2024. [CHECK: initials and arXiv identifier missing in plan.md]

[43] Lin et al. FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection. ICAIF 2024. [CHECK: initials and arXiv identifier missing in plan.md]

[44] Bilgi et al. MEGA-GNN: Multi-Edge Aggregation Graph Neural Networks. arXiv:2412.00241. [CHECK: plan.md gives this title; docs/t1_week0.md gives the arXiv title as "Multigraph Message Passing with Bi-Directional Multi-Edge Aggregations" (Bilgi, Chen, Atasu); confirm which to print, and the year]

[45] Co-Investigator AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives. arXiv:2509.08380. [CHECK: authors and year missing in plan.md; docs/external_code_check.md attributes the paper to PayPal]

[46] FedGraph-VASP. arXiv:2601.17935. [CHECK: authors, full title and year missing in plan.md]

[47] FC2L: Federated Continual Graph Contrastive Learning for Evolving Cross-Bank Money Laundering Detection. NeurIPS 2026. [CHECK: authors missing in plan.md; docs/t1_week0.md names Bashir and Mohan]

[48] Networked Markets, Fragmented Data. arXiv:2512.24487. [CHECK: authors and year missing in plan.md; the repo's documents do not describe this paper, so it is listed but not discussed in the text]

[49] Yao et al. ReAct. ICLR 2023. [CHECK: initials, full title and arXiv identifier missing in plan.md]

[50] Shinn et al. Reflexion. NeurIPS 2023. [CHECK: initials, full title and arXiv identifier missing in plan.md]

[51] Xu et al. ReWOO. 2023. [CHECK: initials, full title, venue and arXiv identifier missing in plan.md]

[52] Zhou et al. Language Agent Tree Search (LATS). ICML 2024. [CHECK: initials and arXiv identifier missing in plan.md]

[53] Kim et al. LLMCompiler. ICML 2024. [CHECK: initials, full title and arXiv identifier missing in plan.md]

[54] Wang et al. CodeAct. ICML 2024. [CHECK: initials, full title and arXiv identifier missing in plan.md]

[55] Du et al. Improving Factuality and Reasoning through Multiagent Debate. ICML 2024. [CHECK: initials and arXiv identifier missing in plan.md]

[56] Wu et al. AutoGen. 2023. [CHECK: initials, full title, venue and arXiv identifier missing in plan.md]

[57] Hu et al. Uncertainty of Thoughts. NeurIPS 2024. [CHECK: initials, full title and arXiv identifier missing in plan.md]

[58] Li et al. MediQ. NeurIPS 2024. [CHECK: initials, full title and arXiv identifier missing in plan.md]

[59] Zhao et al. ExpeL. AAAI 2024. [CHECK: initials, full title and arXiv identifier missing in plan.md]

[60] Wang et al. Agent Workflow Memory. 2024. [CHECK: initials, venue and arXiv identifier missing in plan.md]

[61] Khattab et al. DSPy. ICLR 2024. [CHECK: initials and full title missing in plan.md]

[62] Ong et al. RouteLLM. 2024. [CHECK: initials, full title, venue and arXiv identifier missing in plan.md]

[63] Chen et al. FrugalGPT. 2023. [CHECK: initials, full title, venue and arXiv identifier missing in plan.md]

[64] Ying et al. GNNExplainer. NeurIPS 2019. [CHECK: initials and full title missing in plan.md]

[65] Luo et al. PGExplainer. NeurIPS 2020. [CHECK: initials and full title missing in plan.md]

[66] Hines et al. Spotlighting. 2024. [CHECK: initials, full title, venue and arXiv identifier missing in plan.md]

[67] Chen et al. StruQ. USENIX Security 2025. [CHECK: initials and full title missing in plan.md]

[68] Xu and Ramdas. Online multiple testing with e-values (e-LOND). AISTATS 2024. [CHECK: initials missing in plan.md]

[69] Qwen Team. Qwen3 Technical Report. 2025.

[70] Qwen3.8-27B model card. Qwen, August 2026.

[71] Gemma Team. Gemma 4 model cards (12B, 26B A4B, 31B). Google, 2026.

[72] Abdin et al. Phi-4-reasoning Technical Report. Microsoft, 2025. [CHECK: initials missing in plan.md]

[73] OpenAI. gpt-oss-120b & gpt-oss-20b Model Card. 2025.

[74] Benchmark listing for the HI-Small SOTA figure: https://www.wizwand.com/dataset/ibm-aml. [CHECK: plan.md notes the source paper was still to be traced; docs/t1_week0.md says the page redirects to sota2.com and that the figure comes from [75]]

## Additions from the repo's own documents [75]-[77]

[75] H. Tariq, M. Hassani. Extracting Money Laundering Transactions from Quasi-Temporal Graph Representation (ExSTraQt). arXiv:2604.02899, 2026. (Source: docs/t1_week0.md.)

[76] FFIEC. BSA/AML Examination Manual, Appendix F: Money Laundering and Terrorist Financing Red Flags. Read from the Internet Archive snapshot of 2024-12-23. (Source: configs/compliance/red_flags.yaml.) [CHECK: year of the manual edition and the live URL; the repo cites the archive snapshot]

[77] Monetary Authority of Singapore. Guidelines to MAS Notice 626 (Banks), November 2015, Appendix B: Examples of Suspicious Transactions. (Source: configs/compliance/red_flags.yaml, which notes that newer red flags are on the STRO website and should be re-checked.) [CHECK: re-check against the current guidelines]
