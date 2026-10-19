# T1 week-0 report (Abhishek)

Written by the overnight run on 2026-10-19. Every claim has a URL. GitHub licence and date fields come from the GitHub REST API (`api.github.com/repos/<owner>/<repo>`), read the same day.

## 1. The "~80 F1 on HI-Small" figure

- The aggregator page https://www.wizwand.com/dataset/ibm-aml redirects (301) to https://sota2.com/research/dataset/ibm-aml. It shows **80.89 minority-class F1 on IBM AML HI-Small** (page last updated 2026-04-03).
- The leaderboard behind it, https://sota2.com/research/sota/money-laundering-detection-on-ibm-aml-hi-small, lists: ExSTraQt (FraudGT split) **80.89**, ExSTraQt (GFP split) 78.9, FraudGT 76.41, MultiGNN 68.16, GFP 64.77, PNA 56.77.
- **Paper:** "Extracting Money Laundering Transactions from Quasi-Temporal Graph Representation" (ExSTraQt), H. Tariq (ING, TU Eindhoven) and M. Hassani (TU Eindhoven), arXiv:2604.02899, 2026-04-03: https://arxiv.org/abs/2604.02899 (HTML: https://arxiv.org/html/2604.02899v1).
- **Model:** a supervised, non-GNN method on engineered quasi-temporal graph features (paper abstract: simpler and more scalable than existing approaches).
- **Numbers:** HI-Small minority-class F1 78.90 +- 0.23 with the GFP-style split and **80.89 +- 0.32 with the FraudGT split**, mean of 5 seeds. The GFP split is temporal 60/20/20 by timestamp; the paper does not state the FraudGT ratios in the text I could read, and I could not confirm whether the decision threshold was tuned on test. Check both before quoting the number.
- **Code:** yes, https://github.com/mhaseebtariq/exstraqt (notebooks plus `gfp/` and `ibm/` folders, last push 2026-06-10). **No licence file** (GitHub reports none), so by default all rights are reserved: we may read and reproduce it for study but should ask the authors before redistributing or building on it.
- **Verdict:** traced. The ~80 comes from a different (feature-based) method evaluated on a different split from ours; it is not comparable with the GFP 63.23 / Multi-GNN numbers without re-running on one split. The same table lists FraudGT at 76.41 and MultiGNN at 68.16.

## 2. Group A code and licences

| Model | Repo | Licence | Framework | Last update | HI-Small out of the box? |
|---|---|---|---|---|---|
| Multi-GNN (AAAI 2024) | https://github.com/IBM/Multi-GNN | Apache-2.0 | PyTorch + PyG (conda env: Python 3.9, CUDA 11.8) | last commit 2024-02-13; last push 2025-09-18 | Yes: `format_kaggle_files.py HI-Small_Trans.csv`, then `--data Small_HI` (README) |
| FraudGT (ICAIF 2024) | https://github.com/junhongmit/FraudGT (paper: https://dspace.mit.edu/handle/1721.1/157762) | **none declared** | PyTorch + PyG, GraphGym-style configs (Python 3.9, CUDA 11.8) | 2025-06-05 | Yes: README says put `HI-small.csv` in `./data`; processed on first run |
| MEGA-GNN (arXiv:2412.00241, "Multigraph Message Passing with Bi-Directional Multi-Edge Aggregations", Bilgi, Chen, Atasu) | https://github.com/hcagri/MEGA-GNN (linked from https://arxiv.org/html/2412.00241v2) | Apache-2.0 | PyTorch + PyG (Python 3.9 env; ships `.so` files built for cp39) | 2025-03-28 | Yes: `format_kaggle_files.py`, evaluates Small HI/LI and Medium HI/LI with the Multi-GNN temporal split. The C++ extensions (`negative_sampling.cpp`, `ports_cpp.cpp`) must be rebuilt for Python 3.12 |

Reported HI-Small minority F1 (MEGA paper Table 2, https://arxiv.org/html/2412.00241v2): MEGA-GenAgg 74.88, MEGA(PNA)-GIN 74.69, Multi-PNA baseline 68.16.
FraudGT needs a licence question to the authors before we copy code into `ftm/baselines/`; use it as a separate checkout. Multi-GNN and MEGA-GNN can be vendored under Apache-2.0 with the notice kept.

## 3. Group B

- **Flower** (DP-FedAvg framework): Apache-2.0, https://github.com/flwrlabs/flower, pushed 2026-10-18.
- **Opacus** (per-sample DP-SGD): Apache-2.0, https://github.com/meta-pytorch/opacus, pushed 2026-09-24.
- AML-specific federated candidates (plan.md section 16, decision 3):

| Method | Code? | Licence | Data | Note |
|---|---|---|---|---|
| Tian et al., "Towards Collaborative Anti-Money Laundering Among Financial Institutions", WWW 2025, arXiv:2502.19952 (https://arxiv.org/abs/2502.19952) | **Yes**, https://github.com/zhihuat/Collaborative-AML (pushed 2025-06-24) | **none declared** (no LICENSE file on `main`) | AMLSim and **AMLworld HI-Small** (its `data_process/ml_extraction.py` takes `HI-Small_Trans.csv` and `HI-Small_Patterns.txt`) | Pandas/sklearn-style code; institutions as clients, which matches our bank split |
| Ceydeli, Wang, Atasu, arXiv:2605.06433 (https://arxiv.org/abs/2605.06433), now titled "Federated Layer-Wise Exchange for Cross-Client Money Laundering Pattern Detection" | No link found on the arXiv page | n/a | multigraphs with laundering patterns; dataset not named on the abstract page | Idea fits (layer-wise exchange, matches centralised training); no code to run |
| FedGraph-VASP, arXiv:2601.17935 (https://arxiv.org/abs/2601.17935) | No: "source code will be made publicly available upon acceptance" | n/a | Elliptic Bitcoin and an Ethereum set, not AMLworld | Out |
| FC2L, NeurIPS 2026 (https://neurips.cc/virtual/2026/163758), Bashir and Mohan | No code or results found | n/a | not checked | Stretch only |

**Recommendation: Tian et al. [8].** It is the only candidate with public code that already reads AMLworld HI-Small and its Patterns file. Risk: no licence, so email the authors for permission or reimplement the core from the paper; ask this in week 1. Fallback: Ceydeli et al. [9] reimplemented from the paper, since it shares MEGA-GNN's group and framework. FedGraph-VASP and FC2L are not options (no code).

## 4. snapml licence verdict (G0)

- snapml 1.17.2 installs from PyPI on Python 3.12 (a cp312 manylinux wheel exists), https://pypi.org/project/snapml/ . `from snapml import GraphFeaturePreprocessor` works and produces features on a 5-edge toy graph (63 columns with fan, degree, scatter-gather, temporal-cycle and length-constrained-cycle enabled).
- Licence: metadata says "Proprietary"; the bundled `LICENSE.pdf` is the **IBM International License Agreement for Non-Warranted Programs (ILAN)**. It grants a non-exclusive licence to use and copy for the authorised use, with no redistribution, sub-licensing or reverse engineering; no warranty or support. It has no academic or non-commercial carve-out and no clause on publishing benchmark results (I searched the text for "benchmark", "academic", "non-commercial", "publish": only the warranty clause matched). It is not an open-source licence.
- **G0 decision: use snapml for development and for the see-everything / per-bank GFP+GBT runs** (a university project using it as a library for research, without redistributing it or its weights, is within the grant). Do not vendor it into the repo, do not put it in the base dependencies (it stays an optional install, as `pyproject.toml` already has it), and keep a code path that does not need it. **Open point for the TA:** confirm the project may use it, since the ILAN refers to an "Authorized Use" set by IBM's invoice, which is empty for a free PyPI download. Reimplementing GFP (30-50 h) is the fallback only if the TA says no.

## 5. Week-0 items and status

- Qwen3-8B 4-bit on the 3070 Ti: **ran as an ollama GGUF Q4_K_M proxy, not vLLM** (section 6). A vLLM check is still open.
- HI-Medium download and conversion: see `ftm/data/README.md` (added after the HI-Small hand-off; status there).
- Conversion benchmark (laptop hours to reference hours): **not run**, T3's benchmark script is not published yet (`docs/hardware.md` section 4).
- Not downloaded by design: LI-* datasets, HI-Large.

## 6. Qwen3-8B test on the 3070 Ti

**Caveat: this is a proxy.** vLLM is not installed (it would replace the user-site torch), so I used the local **ollama 0.15.0 server with `qwen3:8b` (GGUF Q4_K_M, 5.2 GB)**. It is not the vLLM 4-bit serving stack that plan.md section 12 specifies, so the second-server decision still needs a vLLM check on the same checkpoint. The numbers below show only that the weights, the KV cache for the 6,000-token cap and a 8,192-token context fit on the 8 GB card and how fast they run in llama.cpp-style serving.

Setup: `/api/chat`, `num_ctx` 8192, `temperature` 0, `num_predict` 256, `think: false`. Prompt = 120 aliased HI-Small rows of bank 70 (val split, from `ftm.data.io` + `alias.apply_aliases`) plus a one-line question, about 5,900 tokens (the harness cap is 6,000). Each of the 3 runs used a different slice of rows so the prompt cache could not help (a first attempt that reused the prefix gave meaningless prompt speeds of over 100,000 tokens/s; discarded). The model was unloaded before run 0, and no other GPU process was running.

| run | prompt tokens | prompt eval tok/s | generated tokens | generation tok/s | load s | total s |
|---|---|---|---|---|---|---|
| 0 (cold load) | 5,927 | 1,982 | 26 | 62.0 | 2.4 | 5.84 |
| 1 | 5,919 | 2,099 | 27 | 61.9 | 0.07 | 3.34 |
| 2 | 5,830 | 2,102 | 29 | 62.4 | 0.07 | 3.32 |

- Loaded **100% on GPU** (`ollama ps`: 6.6 GB, 8,192 context). Peak VRAM from `nvidia-smi` was **6.39 GB of 8.19 GB** (about 0.23 GB of that is the desktop).
- It answered the toy question sensibly each time (it only produced 26-29 tokens, so the 62 tok/s generation rate is measured over short outputs; expect similar for longer ones).
- Result: **Qwen3-8B Q4_K_M runs on the 3070 Ti at the 6,000-token cap with about 1.8 GB VRAM to spare** (ollama proxy). `ollama stop qwen3:8b` was run afterwards; the model stays downloaded.
- The tokens/s here are ollama numbers, not the T3 benchmark (fixed prompt, vLLM), so they must not be used for the reference-hour factor.
