# Schedule to the 10 November deadline

Written on 25 October 2026, with 16 days left. This replaces the week-by-week tables in `tracks.md` §3.9, §4.10, §5.8 and §6.11 from today on. **Nothing is dropped.** Every remaining item, including secondary and stretch ones, has a slot. The plan's own rule (`plan.md` §15, "A larger backbone" row) applies: at G4, if hours run over, lower-priority ablations run on the 1k-alert subset.

The work is front-loaded so that **7–10 November is free for changes, new analyses, reruns and polishing**. The tables, figures and the report text that do not depend on test numbers are written during the runs, not after them.

## Compute

- **Remaining LLM work:** about 160–320 reference GPU-hours (5070 Ti-hours):
  - the E18 backbone sweep;
  - teacher runs;
  - QLoRA 8B and 4B;
  - validation runs of our system and the frameworks;
  - external agents;
  - the final run.
- **Available:**
  - the 5070 Ti (S3), about 24 h a day;
  - from 28 October, the 3070 Ti (S1) at about half that speed, once the T1 GPU jobs finish.
- **The final run** is about 60 h at the measured 7.5 s per alert with Qwen3-8B (`docs/t3_first_runs.md`). That is about 2.5 days on S3 and S1 together.

The schedule only holds if S3 never sits idle. It runs a queue around the clock, and CPU work runs beside it, never waiting for it.

## Schedule

| Phase | T1 (Abhishek) | T2 (Bhavya) | T3 (Akshat) | T4 (Shreyas) |
|---|---|---|---|---|
| **A. 25–27 Oct** | Weeks 10–13: MEGA-GNN, the GNNExplainer, PGExplainer and PPR comparisons, federated ε reporting, final configs (running). Write the T1 methods section: data, splits, intake, one-bank and see-everything, GNNs, retriever. | All non-LLM planners and ablation configs on val (G4). Check that `cross_currency_out(..., window=after(72))` cannot see the future. Agree the run-log schema with T4. Write the methods for the DSL, banks, DP mechanisms and back-testing. | **E18 backbone sweep, starting now; decision by 26 Oct.** Teacher runs from 26 Oct. HI-Medium GFP see-everything on S3's CPU, alongside. | Send the TA addendum if it has not gone (G0). Fusion v1 (per-family) on the real back-test table. Lock the external list. **Report skeleton, introduction and related work.** |
| **B. 28–30 Oct** | S1: FraudGT full run with a prediction dump (about 14 h), layer-wise exchange on HI-Medium (neighbour-sampled), then LLM validation runs. Write the T1 validation-results section with figures. | ModernBERT/DeBERTa and RouteLLM-style baselines (S2). Async runs. External agents if time. Write the validation-results section. | QLoRA planner SFT (8B) and fast model (4B), library build, DSPy-tuned ReAct, LATS, LLMCompiler, Reflexion and ReWOO on val, conformal cascade, SemIf/Laya. HI-Medium Multi-GNN see-everything on S3 overnight (about 8 h). Write the methods section. | Injection tests on our system and the external agents. CaMeL comparison. UoT, BED-LLM, debate/AutoGen and FCAC ports. Eval dashboard on val. **Dry-run every analysis script on val, so the test tables and figures are pre-built.** Presentation skeleton. |
| **30 Oct** | **G4 review:** pick the top 2–3 external agents for the full test, compute check. **Pre-registration frozen and its hash sent to the TA. All final configs merged.** | | | |
| **C. 31 Oct–3 Nov: final run (G5)** | T1 test jobs first (GFP, GNNs, federated, retriever; about 6 h), then half of the LLM investigations on S1. Fill the T1 test numbers into the pre-built tables. | All non-LLM arms, and the banks for every LLM arm (CPU). Fill the T2 tables. | Host the run on S3, keep the GPU calendar. Pick case-study candidates on val, so only the test ones need swapping in. | Run the stats on each arm as it finishes, not all at the end. |
| **D. 4–6 Nov: write-up** | T1 results and discussion; review the other sections. | T2 results and discussion. | 10 case studies; T3 results. | Holm and bootstrap over everything. **Full report draft by 6 Nov.** Presentation draft. |
| **E. 7–10 Nov: slack** | Changes, new analyses, reruns of anything broken, polishing the report and presentation. **Submit on 10 Nov (G6).** | | | |

## Rules

1. **26 Oct is the backbone decision point.** If E18 picks a model above about 8B, S1 cannot serve it. The final run then needs up to about 155 h on S3 alone, so it starts on 30 Oct and ends about 5 Nov. Most of the slack is used, but the deadline holds.
2. **S3 never waits.** Keep one queue per night, with E18, teacher runs, QLoRA, validation runs and the final run back to back. GFP see-everything, back-testing shards and stats run on CPUs beside it.
3. **S1 takes every GPU job that fits in 8 GB:** FraudGT, the HI-Medium layer-wise exchange with neighbour sampling, and LLM runs if the backbone fits. From 31 Oct to 3 Nov nothing else runs on S1.
4. **Test tables and figures are built on val first** (30 Oct). After the run, results arrive as numbers in place, not as new code.
5. **No new features after 30 Oct**, only fixes. New ideas go into the 7–10 Nov slack.
6. **G2 needs a team decision this week.** On real data the expert questions do not reach the 0.01 Brier gain (HI-Small −0.0053; HI-Medium naive −0.0008, per-family +0.0129 with CI [−0.034, 0.064]), and the planner asks no accepted question, so its Brier equals p0. Decide by 27 Oct whether to report this as it stands or to change something in phase B, not later.

## Risks

| Risk | What happens |
|---|---|
| S3 or its owner unavailable for 1–2 days | No slack on that machine before 30 Oct; the 7–10 Nov slack absorbs it, and the write-up moves earlier |
| E18 picks a model above about 8B | Rule 1: the final run starts 30 Oct on S3 alone |
| TA turnaround on the pre-registration hash (30 Oct) or the addendum | Send both as early as possible; the final run waits for the hash |
| Another project uses S1's GPU | It stays off S1 from 31 Oct to 3 Nov |
