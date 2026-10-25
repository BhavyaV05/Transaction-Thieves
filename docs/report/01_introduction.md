# 1 Introduction

*Owner: T4 (Shreyas). Draft 1, written before the test run: it states the questions, the predictions and the design, and it contains no test result. Numbers that come from a script are written as `{{TBD:<id>}}` and are filled in after the final run (G5); see `docs/report/README.md`.*

## 1.1 Problem and motivation

Banks catch money laundering by watching transactions, but each bank sees only the transactions that touch its own customers. Launderers know this and route money through accounts at many different banks. The authors of the AMLworld dataset showed that a model that sees one bank's slice of the data catches far less than a model that sees everything [1]. The patterns that matter most (money passing straight through an account, going in a circle, or being split up and recombined) cross banks on purpose, so the part of the picture a single bank is missing is the part where the laundering is.

Banks cannot simply pool their data because of privacy law. They are, however, allowed to ask each other about specific accounts, under "314(b)" in the United States (31 CFR 1010.540 [16], FinCEN's fact sheet [17]) and under COSMIC in Singapore [18]. In practice, answering such a request usually means sending over actual transaction records, which is slow and exposes a lot of customer data. A more private option is to ask a question that the other bank answers with one small, slightly noisy value, for example "did this account pass on more than 80% of what it received within a day?". Nobody has measured how much such questions recover, which ones are worth asking, or whether an LLM is any good at inventing them.

The last point matters for a practical reason. Banks are adopting LLM agents for investigation now, and a recent LLM investigation agent made worse calls than the simple model it was meant to help, because it trusted its own convincing-sounding reasoning [3]. More generally, fraud-LLM papers rarely report what an LLM adds, whether its evidence can be trusted, or what each decision costs [15]. A system that lets an LLM propose questions to other banks therefore needs three things beyond the idea itself: a way to test every question before it is allowed to influence a decision, a final decision rule that the LLM's text cannot move, and an evaluation that is computed by code against known labels instead of by judgement.

Reviewing alerts is the most expensive part of AML work. If a few private questions recover most of what full data sharing would, banks get a cheaper and more private way to investigate, and the privacy cost becomes a number instead of an argument. If they do not, or if the LLM turns out no better than simple question pickers, that is also a useful, measured result, and we report it with its costs.

## 1.2 The idea

We study one narrow workflow. A bank's own model scores each transaction; for the ones it genuinely cannot decide (the alerts), an agent at the home bank asks other banks a few small, private, typed questions. The system has four ingredients, each of which exists in some form elsewhere, and the contribution is in how they are combined and measured.

1. **Small typed questions.** A question is a program in a small language that the other bank can execute exactly on its own ledger. Its slots are relative to the alert ("the alert account's payers", "the day the transfer arrived"), never fixed accounts or dates. The other bank returns one value (a yes/no, a count, a choice among a short list of banks, or a private set-intersection count [36]) after adding differential-privacy noise [32]. The noise is deliberately visible in the analysis: an answer bought with budget ε_q can shift the odds of laundering by at most e^{ε_q} for a yes/no answer, and the ceiling is drawn on every privacy plot.
2. **Back-testing.** Before a question can affect any decision, it is run on past alerts with known labels, and its likelihood ratio is measured. Acceptance uses e-values and e-value false-discovery-rate control [37], following POPPER [33], so that an LLM that invents thousands of questions cannot slip useless ones in by chance. All main results use *realistic* back-testing, where labels come only from the home bank's own past alerts and other banks answer historical questions with the same noise as live ones, charged to a separate calibration budget.
3. **A formula makes the decision.** The final probability starts from the bank's own score and is updated by each answer according to its back-tested likelihood ratio, with weights learned on validation data so overlapping questions are not counted twice. The LLM's text never enters this calculation, which is how we target the failure reported in [3].
4. **Rules and safety around the planner.** Every question and action passes an allowlist-based compliance gate and a privacy accountant, replies are treated as data and not as instructions [14], and a fast model screens the one short text note a reply may carry.

An illustrative example, taken from the design (these are design numbers, not results): account A at Bank 7 receives 9 payments from 6 different banks within 20 hours, then sends one large payment to account X at Bank 214, and Bank 7's model scores it 0.46. A proven question to Bank 214 ("did X pass on at least 80% within 24 hours, and if so, which of these 4 banks got most of it?"), asked at ε = 1.0 with a back-tested likelihood ratio of 2.6 (just under the ceiling e^1 ≈ 2.7), moves the probability to 0.69. A newly invented private set count to Bank 88, asked at ε = 0.8 with a back-tested ratio of 4.4 (ceiling e^{0.8×2} ≈ 5.0), moves it to 0.91. No transaction record leaves any bank, and the same alert with a fixed question list has no question that asks whether money came back to A's own payers.

Because the main claim is about private questions, we also compare against the alternatives a reviewer would expect: strong published detection models, privacy-preserving federated training, existing AML and fraud LLM agents, and general agent and information-seeking frameworks, all run on the same alerts, the same simulated banks and the same privacy rules (section 6).

## 1.3 Research questions

The first two questions are the core of the project and our main claims rest only on them. RQ6 was added after the instructor's review (`plan.md` section 1) and is also in the core tier.

- **RQ1 (core).** Of the gap between "our bank only" and "see everything", how much do a few private questions close? How does that compare with asking for raw records?
- **RQ2 (core).** Does an LLM that writes its own questions beat simpler planners? Does making every new question prove itself on past data (back-testing) stop the LLM from fooling itself?
- **RQ6 (core).** How does the full system compare with state-of-the-art detection models, privacy-preserving cross-bank methods, and other LLM agents, on the same benchmark and under the same conditions?
- **RQ3 (secondary).** Can a small, fast model make the routine decisions (dozens per alert) as well as the large LLM, for much less?
- **RQ4 (secondary).** Does showing the planner the right 40 transactions (it can only read about 40) lead to better questions?
- **RQ5 (stretch).** Other banks take hours or days to reply. Is it better to ask one question at a time and adapt, or several at once?

## 1.4 Hypotheses

Each hypothesis is a prediction that can fail. The thresholds below are the ones in `plan.md` section 2; they are fixed on validation data and committed to `prereg/` before the test set is touched, and any change to a primary hypothesis after gate G3 goes back to the TA. Primary hypotheses are tested with Holm correction (reported in the order H1, H2, H3, H4, H9); secondary hypotheses are corrected separately; exploratory ones are reported without significance claims. Confidence intervals are 95% cluster-bootstrap intervals (10,000 resamples, clusters of laundering attempts plus connected accounts).

**Primary.**

- **H1.** With a privacy budget of ε = 2 per alert, the questions close at least 30% of the gap, measured in Brier score. We are wrong if the 95% interval of the gain includes zero, or if the share is below 30%.
- **H2.** The questions get at least 70% of the benefit of sharing raw records (three banks, no noise), while sharing none. We are wrong if the ratio is below 70%.
- **H3.** The LLM writing its own questions is at least as good as the best simple planner and as the no-LLM enumerator of all expressible questions, and better than itself limited to a fixed list. We are wrong if it is clearly worse than either the best simple planner or the enumerator, or no better than the fixed list. "As good as" means a difference smaller than half the gap between the best simple planner and random choice.
- **H4.** Back-testing matters: without it results get worse, and with it fewer than 10% of accepted questions turn out useless on later data. We are wrong if there is no difference, or if more than 10% are useless.
- **H9.** At ε = 2, our system closes at least as much of the gap as the best ported external agent on the same backbone model. We are wrong if the best external agent is clearly better (95% interval). H9 is last in the reporting order, so it leaves H1 to H4 unaffected.

**Secondary.**

- **H6.** The evidence retriever shows the planner at least 15 points more of the real laundering pattern than the baseline view, and improves results. We are wrong if the gain is under 15 points or there is no effect.
- **H7.** A calibrated formula making the final call beats the LLM deciding (repeating Palma [4] in our setting). We are wrong if there is no measurable difference.
- **H10.** Private questions match or beat differentially private federated training on Brier score at a comparable privacy level. We are wrong if federated training is clearly better. The privacy units differ (training-time ε against per-question ε), so this comparison carries an explicit caveat and is secondary for that reason.

**Exploratory.**

- **H5.** The fast model handles most routine decisions within its stated accuracy bound, at least 10 times faster, with no loss in detection. We are wrong if it falls below the bound, is under 10 times faster, or detection gets worse.
- **H8.** Asking two questions at once and then one more keeps 80% of the benefit in 60% of the time. We are wrong if either number is missed.

The headline metric is the Brier score, and *gap closed* turns it into a percentage: 0% is no better than the bank's own model and 100% is as good as seeing everything. Gap closed is reported against two upper limits (the dataset authors' graph-feature model and the best state-of-the-art detection model), and at two levels side by side: on the alerts the agent actually sees, and end to end over all rows including those the bank's own model clears, with the intake miss rate printed next to every number.

## 1.5 Contributions

Each building block exists somewhere already (Table 1 of the proposal, section 2 of this report); what is new is how they are put together and what we measure. We give:

1. **A privacy-versus-detection curve for cross-bank questions**, including private set counts for following money, compared against sharing raw records, with the theoretical privacy ceiling drawn on the same plot (RQ1; H1, H2).
2. **LLM-invented questions that must pass error-controlled back-testing before they count**, which directly targets the failure reported in [3] (RQ2; H3, H4).
3. **A strategy library filed by the shape of the situation rather than by account**, with a test showing why filing by account cheats: the same laundering rings reappear later in AMLworld, so a library keyed by account would memorise known criminals.
4. **The first test of open "System One" fast decision models inside a multi-agent system**, with a hand-off to the larger model that has a stated accuracy bound (RQ3; H5).
5. **Evidence retrieval for one bank's transaction graph that keeps the trail connected and in time order** (RQ4; H6).
6. **Head-to-head comparisons on one benchmark under one fairness protocol** (RQ6; H9, H10): state-of-the-art detection models used as both starting point and upper limit; differentially private federated training and an AML-specific federated method; existing AML and fraud LLM agents (Torres et al. [6], Sharma [3] and others), each run as published (native) and given our question tool with the same budget (ported); and general agent and information-seeking frameworks (ReAct, Reflexion, ReWOO and others). Every LLM system gets two scores, end to end and questions-only, which separates asking better questions from making better final calls.

Beyond these, we release the code, the simulator, the frozen alert sets and the harness so that other agents can be plugged into the same benchmark. We do not release trained weights, because a system that learns which questions clear an account could help criminals (section 8).

## 1.6 Scope and honesty about what this is

The study is synthetic and simulated, and the claims are limited accordingly. The data is IBM's AMLworld [1], where every transaction is labelled, so every number is computed by code against the true answer and no LLM grades anything; the only human judgement is two teammates checking whether 100 sampled questions match the planner's stated hypothesis. Other banks are simulated nodes, not real institutions, and we simulate the output of the private set-intersection protocol [36] rather than build the cryptography. Differential privacy here protects one transaction, which is weaker than protecting a whole customer, so we also cap the total ε spent on each partner account across all alerts and all asking banks for the whole test period. Conformal and false-discovery-rate bounds hold only approximately under day-to-day drift, so we always report the rates we actually get next to the targets. Where the system does not work, the report says where and at which step (section 9).

## 1.7 Preview of the findings

*(To be completed after the final run; every number is a placeholder until then.)* At ε = 2 per alert, private questions close {{TBD:gap_closed_gfp}} of the gap to the graph-feature upper limit and {{TBD:gap_closed_sota}} of the gap to the best state-of-the-art upper limit, and reach {{TBD:h2_ratio}} of the benefit of sharing raw records. The LLM planner is compared with the best simple planner and the enumerator in {{TBD:planner_comparison}}. Against the best ported external agent the difference is {{TBD:h9_difference}}. The prespecified verdicts for H1 to H10 are in {{TBD:hypothesis_verdicts}}.

## 1.8 Roadmap

Section 2 reviews related work on AML data and detection, cross-bank and privacy-preserving methods, LLM agents for fraud, general agent frameworks, error-controlled use of LLM ideas, retrieval and memory, prompt-injection defences and regulation. Section 3 defines the problem and walks one alert through the system. Section 4 describes the data, the splits and the evaluation resources. Section 5 describes each component and how it is tested. Section 6 lists the baselines, external systems and ablations and the fairness protocol, and the metrics and statistics. Section 7 reports the results against the hypotheses. Section 8 covers safety: privacy, compliance and prompt injection. Section 9 analyses failures by pipeline step and presents ten case studies. Section 10 states limitations and future work, and section 11 gives the compute budget and how to reproduce every number.
