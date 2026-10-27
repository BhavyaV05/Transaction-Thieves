# 3 Problem and approach

*Owner: T4 (Shreyas), with a read by every track. Draft 1 (27 Oct 2026), written before the final run: it defines the problem, states the invariants, redraws Figure 1 of the proposal and follows one illustrative alert through the system, and it contains no result. Numbers from a script are `{{TBD:<id>}}` (convention in `docs/report/README.md`). Remove this line before the final build.*

This section states the problem formally (3.1), the five rules that hold everywhere (3.2), the system as a whole (3.3), one alert followed through it (3.4), and why each of the five capabilities is needed and how it is measured (3.5). Section 4 describes the data and section 5 each component.

## 3.1 The setting

There are N banks. The **home bank** b owns the alert; the other N − 1 are **partner banks**. Each bank sees only the transactions that touch its own customers. In the experiments the views are those of the 30 largest banks of AMLworld [1] (section 4), and every partner is a simulated, independent node with a private ledger L_j.

An **alert** is one transaction that the home bank's own model scores and cannot settle. The model gives a score p0, and conformal intake [38, 39] lets the transaction through to the agent only when the model cannot confidently rule out either answer (section 5.1). Its label is the transaction's laundering flag in the dataset. The task is to output a probability p̂ that the alert is laundering, and then one decision: escalate if p̂ ≥ t, otherwise clear (section 5.4.5).

To improve on p0 the agent may ask partner banks **typed questions**. A question q is a program in a small language that the partner executes exactly on its own ledger (section 5.2). Its slots are relative to the alert, such as `ALERT.receiver` (the account that received the alerted transfer) and `ALERT.payers`, and never an account number or a date. The partner does not return rows. It returns one value, a yes/no, a count, a choice among a short list, or a private set-intersection count [36], after adding differential-privacy noise [32] with a privacy parameter ε_q chosen by the asker from the levels {0.25, 0.5, 1, 2}. The protected unit is one transaction: adding or removing one transaction changes the chance of any answer by at most a factor e^{ε_q}. The same bound limits what an answer can tell the agent, since a yes/no answer bought with ε_q can shift the odds of laundering by at most e^{ε_q}.

Three limits apply, all counted as sums of ε (basic composition). The **per-alert budget** is ε = 2.0 across all questions about one alert, whichever banks are asked. The **per-account cap** is ε = 8.0 on each partner account across all alerts and all asking banks for the whole test period, because protecting one transaction is weaker than protecting a customer. Back-testing (section 5.2) spends from a separate **calibration budget** ε_cal, which is reported on its own and never mixed with the other two. The cap of 8.0 is provisional (section 5.4.2).

Other banks act on their own time: a reply usually comes within a day, sometimes takes up to four, may be a decline, and is a timeout after five days. A reply may also carry one short text note, as real 314(b) replies do. The note is untrusted text that the planner may read, which makes it the place where hidden instructions would be planted (section 8).

The quality of p̂ is its Brier score, and the headline number is the gap closed, (Brier(one bank) − Brier(our system)) / (Brier(one bank) − Brier(see everything)), where "see everything" is the same one-bank model trained on every bank's data. It is an upper limit that needs every bank's data, and it is reported next to the system in every table (section 6.4).

## 3.2 Invariants

Five rules hold in every run, each enforced in code and covered by a test, so the privacy and safety claims can be checked and are not just stated.

1. **No transaction record ever leaves a partner bank; only single noisy values do.** A partner's handler returns only a reply, a decline or a timeout, and a reply carries a `bool` or an `int`. The tests check the returned types and the public surface, and scan the package for calls that return rows.
2. **LLM text never enters the decision.** The final probability is computed from the home bank's score and from numbers: typed answers turned into back-tested likelihood ratios. The fusion function raises an error on non-numeric input (section 5.4.4), so neither the planner's reasoning nor a reply note can reach it. This is the design answer to the failure reported for an LLM investigator that trusted its own reasoning [3].
3. **Time guard.** Every tool takes the time `as_of` of the alert and sees only data up to it: the home-bank graph raises an error if asked for a later time, and a partner's ledger is cut at the time it answers, so a reply received after 12 hours covers only 12 hours of a 24-hour window. The test split is also locked until the final run (`FTM_FINAL_RUN`, section 6.6).
4. **Aliases.** Account identifiers are replaced by random aliases in everything an LLM reads. The mapping to real node ids exists only inside the simulator, and the gate rejects any question that contains an alias, an account number or a date.
5. **One gate for every system.** Every question and action, from our planner or from any external agent, passes the same allowlist gate and the same privacy accountant, with the same settings file. The harness toolbox is the only road to a partner bank and keeps its own log of what was asked, so a system cannot misreport its questions or bypass its limits (sections 5.2 and 5.4.1).

## 3.3 The system

Figure 1 redraws the system overview of the proposal.

```
+- HOME BANK b: sees only its own customers' transactions --------------------------------------+
|                                                                                               |
| +---------------------+             +----------------------+           +--------------------+ |
| | Alert intake        | local graph | * Evidence retriever |           | * Strategy library | |
| | one-bank score p0   |------------>| edge scorer + PCST,  |           | successes+failures | |
| | with conformal      |             | time-respecting,     |           | filed by shape,    | |
| | filter: only truly  |             | K = 40               |           | not by account     | |
| | unsure alerts pass  |             +----------------------+           +--------------------+ |
| +---------------------+                         | evidence                       |            |
|                                                 | subgraph                       | proven     |
|                                                 v                                | questions  |
|                                     +----------------------+                     |            |
| +---------------------+             | * System 2 planner   |                     |            |
| | + System 1 (fast)   | typed calls | states hypotheses;   |<--------------------+            |
| | small typed model:  |             | composes or selects  |                                  |
| | triage, compliance  |<----------->| typed questions      |           +--------------------+ |
| | injection, typology | / hand-offs |                      | citations | Regulatory RAG     | |
| | stop; if unsure,    |             |                      |<----------| 314(b), COSMIC,    | |
| | hands over          |             |                      |           | red flags          | |
| +---------------------+             |                      |           +--------------------+ |
|            |                        +----------------------+                                  |
|            | stop / escalate               ^          | typed question                        |
|            |                    posterior, |          +--------------------------+            |
|            v                   budget left |                                     v            |
| +---------------------+             +----------------------+           +--------------------+ |
| | Three-way decision  |             | * Evidence fusion    |           | Compliance gate    | |
| | auto-clear / human  |<------------| learned weights on   |           | checker + privacy  | |
| | / auto-escalate     |             | back-tested LRs,     |           | accountant; eps    | |
| +---------------------+             | no LLM text          |           | per alert, per     | |
|                                     +----------------------+           | account cap        | |
|                                                 ^                      +--------------------+ |
|                                                 |                                |            |
|                                                 |                                |            |
+-------------------------------------------------|--------------------------------|------------+
                                                  |             question (network) |
                                                  |                                |
                                                  |                                v
                                                  |                  +-------------------------+
                                                  |                  | OTHER BANKS             |
                                                  |                  | Partner banks:          |
                                                  |                  | independent nodes;      |
                             noisy typed answer + |                  | private ledger up to    |
                     back-tested likelihood ratio |                  | the alert time; exact   |
                                                  |                  | execution of each       |
                                                  +------------------| question; DP noise on   |
                                                                     | answers; per-account    |
                                                                     | eps cap; private set    |
                                                                     | intersection counts;    |
                                                                     | latency, declines;      |
                                                                     | untrusted text note     |
                                                                     +-------------------------+

Offline: back-testing on historical alerts (training period, the home bank's own labels,
noisy answers charged to the calibration budget eps_cal) supplies the back-tested likelihood
ratios (LRs) that evidence fusion uses.
```

**Figure 1.** System overview, redrawn from Figure 1 of `docs/proposal_v5.pdf` as text. Boxes marked * are our main contributions (coloured orange in the proposal); the box marked + is the fast decision model (green). The LLM never makes the final call: only the other banks' answers, weighted by how reliable they proved on past data, reach the decision. No transaction record leaves another bank.

Reading Figure 1 from the top: intake keeps only alerts the one-bank model cannot settle, and the evidence retriever picks about 40 connected, time-ordered transactions of the home bank's graph, because the planner can read only about that many (section 5.1). The planner (section 5.3) states a hypothesis and picks or writes a typed question, with the strategy library and regulatory retrieval as inputs, and the fast model takes the routine decisions. Each question passes the gate and accountant to a partner bank. Fusion turns the noisy answer into a log-likelihood ratio from the back-test table and updates p0 into a posterior; the stop rule decides whether to ask again, and the decision rule escalates, sends to a human or clears.

## 3.4 One alert, followed through

*This example is illustrative. Its numbers are chosen to show how the pieces connect and are not results from any run. It extends the example of the proposal and of section 1.2, with the privacy levels moved onto the gate's grid (the proposal's ε = 0.8 is not one of the four allowed levels).*

Account A at Bank 7 receives 9 payments from 6 different banks within 20 hours, then sends one large payment to account X at Bank 214. Bank 7's model scores that transfer p0 = 0.46, and intake lets it through because the model cannot tell. The planner sees the retrieved evidence with every account shown as an alias, and states a hypothesis: the money is passing straight through X. It looks up the shape of the situation in the library and finds a proven question.

**First question.** To Bank 214 it asks `passed_on(ALERT.receiver, window=after(24), min_frac=0.8)` at ε = 1.0: did X send out at least 80% of what it received within 24 hours of the alert? It cites US-314B-SHARE. The gate checks that Bank 214 is a known partner, that 1.0 is an allowed level, that the program parses, that it holds no alias, number or date (only the slot `ALERT.receiver`), that the cited rule exists, applies and permits a yes/no question, and that the accountant can pay (section 5.4.1). The accountant then records 1.0 of the 2.0 alert budget and 1.0 of the cap of 8.0 on X's account. Say the reply comes back after 9 hours: Bank 214 answers on 9 hours of the 24-hour window (invariant 3), adds randomised-response noise and returns "yes". The back-test table, built earlier from past alerts under the same noise, says a "yes" to this question was about 2.6 times as likely for launderers as for clean accounts; the ceiling at ε = 1.0 is e ≈ 2.7, so the ratio is allowed. The reply carries a short note, which the planner reads and the fast model screens for hidden instructions; fusion never sees it (invariant 2).

**Second question.** The planner now asks the question that checks whether the money circled back: `overlap(ALERT.receiver, dir=out, window=after(72), among=ALERT.payers)` to Bank 214 at ε = 1.0, a private set count of how many of X's recipients in the next 72 hours are among A's own payers. It passes the same checks and the accountant records 2.0 of 2.0 on the alert. The noisy answer is 2. Launderers typically give 2 and clean accounts 0, so the ceiling is e^{1.0 × 2} ≈ 7.4; the back-tested ratio for this answer, in this example, is 4.4.

**Fusion and decision.** With naive weights of 1, the log-odds of the alert go from logit(0.46) ≈ −0.16 to −0.16 + ln 2.6 ≈ 0.80, which is a probability of 0.69, and then to 0.80 + ln 4.4 ≈ 2.28, which is 0.91. (The real system uses weights fitted on validation data, section 5.4.4, so its numbers would differ.) The privacy budget is now spent, so the stop rule ends the run on budget before anything else. With an illustrative threshold t = 0.5 the alert is escalated to human review, and the investigator receives the question log, the answers, the ratios and a draft report that stays inside Bank 7 (rule INTERNAL-SAR-DRAFT). No transaction record left Bank 214, and every step was logged by the toolbox.

## 3.5 Why each capability is needed and how it is measured

The course brief asks that every capability used be justified and measured. Table 3 follows the proposal's table of the same number. The metrics are defined in section 6.3, the ablations are in Table 5b (section 6.1) and the results are in section 7.

| Capability | Why the problem needs it, and what it does here | How it is measured |
|---|---|---|
| Agentic | Investigating means asking, reading the answer and deciding what to ask next under a budget, while other banks act on their own. A slow planner, a fast decision model and the bank agents work together and can hand over to a human | Gap closed per unit of privacy; time to decision; cost per alert |
| Tool calling | The only way to get evidence from another bank is a question that bank runs, and every question spends real budget, so it goes through the rules and privacy check first | Questions that parse and run; recovery from timeouts and refusals; unsafe actions proposed |
| Reasoning | Picking a question means guessing what is going on and testing it. The planner states hypotheses; back-testing checks its new questions | New questions that pass; passed questions that later fail; stated against real confidence; gain over the enumerator of every expressible question |
| RAG | The planner needs what worked before and the right part of the graph, and the gate must find which legal basis applies, since 314(b) and COSMIC differ | Real pattern edges among the 40 shown; correct citations against the whole rules corpus in the prompt |
| Fine-tuning | Banks need this on their own hardware, cheaply, with honest confidence. The 8B planner learns from the teacher only from questions that passed back-testing and moved toward the right answer; the 4B fast model learns calibrated confidence | Planner against teacher; the fast model's accuracy, calibration, speed and share of decisions kept |

**Table 3.** The five capabilities, as in the proposal. Tool calling is where invariants 1, 4 and 5 are enforced, because a question is the only tool that touches another bank. Reasoning is where invariant 2 matters: the planner reasons freely, but its new questions are tested on past labelled data before they count, and its text never touches the final probability.

Each capability is also switched off or replaced in an ablation on the same alerts, banks and privacy rules, for example a fixed question list, no back-testing, the whole rules corpus in the prompt, other evidence views and other fast-model options. If a capability does not help on these measures, the report says so.
