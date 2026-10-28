# T4 hand-off prompt for the next working session (written 28 Oct 2026, ~13:00 IST)

Paste the block below at the start of a new session. It is also the quickest way for anyone taking over T4 to see where it stands.

```
I am Shreyas Deb, owner of track T4 (rules/gate, fusion, decision, evaluation/statistics, safety, Group C ports, pre-registration, report) in the LMA group project "Follow the Money Without Moving the Data" (repo /home/kinomorph/Documents/sem7/LMA/Transaction-Thieves, GitHub BhavyaV05/Transaction-Thieves, `gh` logged in as shrey715). Teammates: Abhishek = T1 (data, detection models), Bhavya = T2 (DSL, banks, DP, back-testing, harness), Akshat = T3 (LLM planner, serving, 5070 Ti machine). Deadline 10 Nov 2026; schedule in docs/schedule_nov10.md (phase B 28-30 Oct, freeze + G4 on 30 Oct, final run 31 Oct-3 Nov, write-up 4-6 Nov, slack 7-10 Nov, rule 5: no new features after 30 Oct).

First: `git fetch --all --prune && git pull`, read STATUS.md (top T4 row), tasks/T4_shreyas.md, docs/t4_prereg_values.md, docs/t4_final_run.md, docs/t4_fusion_findings.md (sections 6-7), and `gh issue list` / `gh pr list`. Then tell me what changed since 28 Oct and what is blocked on whom.

Standing rules (follow them exactly):
- Commit messages lowercase, short, crisp, NO Claude attribution/co-author/session-link lines in commits or PR bodies. Review the diff before committing.
- Do not run LLMs, GPU or heavy jobs on my laptop (CPU-only tests and lightweight scripts only; ask teammates for anything heavy). Python is .venv (3.12); `uv pip install --python .venv/bin/python <pkg>` (no pip in the venv).
- Team rule: check and merge, no pair review: CI green + local tests pass, then `gh pr merge --merge --subject "..." --body ""`. Branches `t4/<topic>`.
- Subagents: Sonnet only, worktree-isolated, they log to .claude/AGENT_LOG.md (excluded from git). Opus does not do the work itself. Close finished agents and remove their worktrees.
- Data (3.4 GB, from Abhishek) is unzipped in data/ (gitignored). Real-data scripts are fine if they are light (fusion_diagnostics, g2_check, replication_check, final_non_llm --split val, final_analysis --split val).
- Never touch the test split or set FTM_FINAL_RUN outside scripts/run_final.py. The user said to forget the TA addendum.

State on 28 Oct (all merged to main unless noted):
- Built: run_final.py orchestrator + freeze (prereg/FREEZE.json is NOT written yet), final_analysis pipeline (+ power, eps=inf row, review capacity S1, p0-adjusted S2), E15 injection runner + CaMeL-style arm (mock only), replication check tool, report sections 1-3, 5.4, 6, 8-11 + abstract skeleton, slide outline.
- PR #97 (freeze also hashes configs/final/non_llm.yaml and configs/experiments/ablations_nonllm.yaml, as Bhavya asked; preflight lists tables.realistic until it is named) and the status PR with this file: check whether they merged.
- Real HI-Medium val findings: G2 not met (naive -0.0053 HI-Small / -0.0008 HI-Medium; fitted +0.0007 / +0.0053, CIs span 0). History-based accepted questions are time-confounded (dormant log LR -0.92 pooled, +0.02/-0.15 in train halves); 12 of 23 accepted questions replicate across time blocks; val direction agrees 10/12 vs 4/11; G2 with only replicating questions: naive +0.0073 [-0.003,+0.018] eval, +0.0086 [-0.001,+0.018] cal (descriptive; issue #84). No-noise gain -0.0287 vs eps 2 -0.0058: noise is not the limiter. Equivalence margin 2.4e-05 (random excluded from "best simple planner" since PR #83); hub cap not triggered (0.0149). E18 decided: Gemma 4 12B, thinking off; H6 not met (E8 and T1). Final-run plan 168.5 h on the 5070 Ti, ends ~6 Nov if started 30 Oct (E15, QLoRA, full-test external agents not included); fix at G4: move the 15 full-test ablation arms to the 1,000-alert subset (saves ~62 h).

My decisions still open (ask me, do not decide for me): (1) G2/fusion default: proposal per_family + prior="zero" + fit_cv, naive reported as a pre-registered negative result; (2) equivalence margin: keep the definition and state the limit (option 1 of docs/t4_prereg_values.md) vs a fixed margin; (3) whether to ask T2 for time-aware acceptance before the freeze (issue #84); (4) confirm S1 (10% review capacity) and S2 wording for the prereg.

To do next (T4, in order): send/refresh the team message (G2 proposal, #84 for Bhavya, compute warning, tables.realistic, asks per person, fix_on_val numbers by Thu 30 Oct morning); merge anything green; after Akshat's val runs exist: run final_analysis --split val on real logs (fixes the best ported external for H9 and the selectors in configs/analysis/systems.yaml marked ASSUMED), fill fix_on_val in prereg/hypotheses.yaml, write outputs/g4/top_external.json after the G4 review, then freeze (docs/t4_final_run.md checklist), send the combined hash to the TA, `--record-ta-sent`; then stats per arm during the final run; report sections 4/5.1-5.3/7 come from their owners, 7.4 and the abstract from me; presentation draft from docs/presentation/outline.md. FCAC (stretch) is not started. Ideas are in issues #84-#92.
```
