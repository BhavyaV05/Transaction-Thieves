"""The small-decision dataset (hand-off #13, T2 to T3), rebuilt in wave 4 against the tuned pipeline: labelled one-token
decisions for the System 1 fast model (C13).

Owner: T2 (Bhavya); wave-4 rebuild by T3 (Akshat).  Spec: plan.md 6.13/6.14 and 5.x, tracks.md 4.9, docs/t3_system1.md.
Format: `ftm/fastmodel/decisions.py` (`decision, context, label, alert_id, split`).

The labels now come from the components the pipeline runs with, not from the alert label or a hand rule:

  worth_asking      the state the planner is in (alert, p0, the fused posterior, the threshold, eps left, the answers so far, the
                    routed back-tested menu) -> yes iff the best AFFORDABLE, RIGHT-BANK-ROUTED, not-yet-asked question has an
                    expected Brier reduction >= GAIN_MIN under the fitted ConditionalFusion (headroom_v1): with the back-test table's
                    p_cell = (P(cell | clean), P(cell | launder)) and the fused posterior p' after each cell, scored against
                    the outcomes the table says occur in that cell,
                    gain = p(1-p) - sum_cell [p P(cell|launder) (1-p')^2 + (1-p) P(cell|clean) p'^2]
                    (a fusion that overshoots a weak answer gets a NEGATIVE gain).  The old label ("the alert is a real attempt") leaked the answer
                    and drifted 57% -> 32% yes across val parts.
  stop              NOT in the set: it is an exact function of the state (`ftm.fastmodel.decisions.exact_stop`), computed by the
                    pipeline.  `stop_label` below is that function, kept for the checks and the manifest.
  typology          the pattern-file typology with rare classes merged (`merge_typology`); clean alerts are `none`; the train
                    side keeps only a share of the `none` rows (class balance), val keeps the natural distribution
  rule_applies      yes iff the cited rule permits ask_bank in the home bank's jurisdiction for this answer kind (the Gate's
                    rules); val shows the rule in held-out wording (paraphrases), train shows the Gate's text
  injection_present yes iff the partner-bank note carries a planted instruction; train and val use DISJOINT template families
                    (`ftm.safety.injection.generate_diverse`: paraphrases, other languages, encodings, benign look-alikes in val)

Contexts are built by `ftm.fastmodel.context`, the same functions the pipeline's System 1 node calls, from aliased PlannerView
ingredients only: no account id, name or date, nothing past the alert time.  Test-split alerts are never used.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from ftm.backtest.engine import BtAlert
from ftm.banks.partner import Question
from ftm.compliance.gate import Gate
from ftm.dsl.expert import expert_programs
from ftm.dsl.parser import canonical, parse
from ftm.fastmodel import context as ctxt
from ftm.fastmodel.decisions import DECISIONS, exact_stop, merge_typology
from ftm.fusion.conditional import ConditionalFusion
from ftm.fusion.fuse import sigmoid
from ftm.fusion.gain import expected_gain, fused_logit
from ftm.graph.policy import MenuItem
from ftm.llm.routing import asked_bank
from ftm.planners.pool import Candidate, QuestionPool
from ftm.safety import injection as inj
from ftm.types import AlertContext, Reply

DECISION_LABELS = {d: dec.labels for d, dec in DECISIONS.items()}
DEFAULT_MIX = {"worth_asking": 8, "typology": 1, "rule_applies": 8, "injection_present": 6}
EPS_PER_ALERT = 2.0  # Limits.eps_per_alert
GAIN_MIN = 1e-3  # worth_asking = yes iff the best question's expected Brier reduction is at least this (absolute Brier)
TYPOLOGY_NONE_KEEP = 0.2  # train: share of the clean-alert typology rows kept (class balance)

RULE_PARAPHRASE = {  # val wording of the Gate's rules (held out: train shows the Gate's own text)
    "US-314B-SHARE": ["Once an institution has told FinCEN it takes part, it can exchange information with other participants about "
                      "people, companies or payments to spot and report suspected money laundering or terrorist financing; that "
                      "information must not be used for anything else.",
                      "Registered 314(b) participants are allowed to swap data on customers and transactions with each other in order "
                      "to detect and report possible laundering or terror financing, and only for that."],
    "US-314B-NOTICE": ["To take part, an institution files a yearly notice with FinCEN and has to maintain safeguards for the security "
                       "and secrecy of the information it gets.",
                       "Each year the participant must renew its FinCEN notification, and it must have procedures that keep received "
                       "information secure and confidential."],
    "SG-COSMIC-REQUEST": ["A COSMIC participant can ask another participant for information about a customer whose behaviour passes "
                          "risk thresholds, when that information bears on laundering, terrorism financing or proliferation risk.",
                          "Where a customer's activity crosses the risk thresholds, one participant may put a request to another to "
                          "obtain information relevant to money laundering, terror financing or proliferation financing."],
    "INTERNAL-SAR-DRAFT": ["An investigator is permitted to prepare a draft suspicious activity report for the home bank's internal "
                           "compliance queue; such drafts stay inside the bank.",
                           "Drafting a SAR for the bank's own compliance review is allowed, and the draft is never passed to a "
                           "different bank."],
}


# --- labels -----------------------------------------------------------------------------------------------------
def stop_label(posterior: float, threshold: float, eps_left: float, *, max_delta: float, max_weight: float) -> bool:
    """`stop` is exact (see ftm.fastmodel.decisions.exact_stop); this is the dataset-side name, used for the checks only."""
    return exact_stop(posterior, threshold, eps_left, max_delta=max_delta, max_weight=max_weight)


def worth_asking_label(model: ConditionalFusion, p0: float, answered, options: Sequence[Candidate], asked: set[str],
                       eps_left: float, gain_min: float = GAIN_MIN) -> tuple[bool, float]:
    """(label, best gain). Candidates: affordable, not asked, with cell probabilities in the table."""
    z = fused_logit(model, p0, answered)
    best = 0.0
    for c in options:
        if c.program in asked or c.eps > eps_left + 1e-9:
            continue
        g = expected_gain(model, z, c)
        if g is not None and g > best:
            best = g
    return best >= gain_min, best


def rule_label(gate: Gate, rule_id: str, jurisdiction: str, kind: str) -> bool:
    r = gate.rules[rule_id]
    return r.applies_to(jurisdiction) and "ask_bank" in r.permits and kind in r.kinds


def menu_items(options: Iterable[Candidate], bank) -> list[MenuItem]:
    """The pipeline's MenuItem for each routed candidate (what PlannerView.menu holds at run time)."""
    return [MenuItem(f"Q{i}", c.program, c.eps, str(bank), c.family, sorted(c.lr.items())) for i, c in enumerate(options)]


# --- the builder ------------------------------------------------------------------------------------------------
class DecisionSetBuilder:
    """`pool`: accepted rows of the back-test table the pipeline runs with (configs/fusion/headroom_v1_table.jsonl);
    `fusion`: the fitted ConditionalFusion (configs/fusion/headroom_v1.json); `contexts`: alert_id -> AlertContext as the planner
    receives it (summary with the shape and the bank fields: `scripts/run_llm_agent.data_alerts`)."""

    def __init__(self, ledgers: dict[int, object], banks: dict, pool: QuestionPool, fusion: ConditionalFusion,
                 contexts: dict[str, AlertContext], threshold: float = 0.3227, gate: Gate | None = None, mix: dict | None = None,
                 seed: int = 0, gain_min: float = GAIN_MIN, none_keep: float = TYPOLOGY_NONE_KEEP,
                 injections: dict[str, Sequence] | None = None):  # fmt: skip
        self.ledgers, self.banks, self.pool, self.fusion, self.contexts = ledgers, banks, pool, fusion, contexts
        self.threshold, self.gain_min, self.none_keep = threshold, gain_min, none_keep
        self.gate = gate or Gate.from_config()
        self.mix = {**DEFAULT_MIX, **(mix or {})}
        self.seed = seed
        self.injections = injections or {sp: inj.generate_diverse(1500, 1500, seed, sp) for sp in ("train", "val")}
        self.gains: dict[str, list[float]] = {}  # split -> best expected Brier reduction of each worth_asking row (manifest stats)
        self.rules = list(self.gate.rules)
        self.programs = [p["program"] for p in expert_programs("receiver")]

    def per_alert(self) -> int:
        return sum(self.mix.values())

    def routed_options(self, a: BtAlert, ctx: AlertContext) -> list[Candidate]:
        """The pool's questions this alert can be asked: applicable to the direction, and whose subject the PARTNER bank holds
        (right-bank routing, `ftm.llm.routing`)."""
        direction = "out" if a.binding.sender_bank == a.binding.home_bank else "in"
        out = []
        for c in self.pool.applicable(direction):
            b = asked_bank(c.program, ctx.summary, ctx.home_bank)
            if b is not None and str(b) == str(a.binding.partner_bank):
                out.append(c)
        return out

    def rows_for(self, a: BtAlert) -> Iterator[dict]:
        rng = random.Random(f"{self.seed}|{a.alert_id}")
        ctx = self.contexts[a.alert_id]
        base = {"alert_id": a.alert_id, "split": a.split}
        if a.label is False or a.typology:
            if a.split != "train" or a.label or rng.random() < self.none_keep:
                for _ in range(self.mix["typology"]):
                    yield {**base, "decision": "typology", "context": ctxt.alert_text(ctx.summary, a.p0),
                           "label": merge_typology(a.typology if a.label else None)}  # fmt: skip
        yield from self._worth_rows(a, ctx, rng, base)
        yield from self._rule_rows(a, rng, base)
        yield from self._injection_rows(a, rng, base)

    def _worth_rows(self, a, ctx, rng, base):
        options = self.routed_options(a, ctx)
        bank = a.binding.partner_bank
        if not options or bank not in self.banks:
            return
        n = self.mix["worth_asking"]
        for i in range(n):
            k = i % 4  # answers so far: 0, 1, 2, 3
            asked: set[str] = set()
            families: set[str] = set()
            answers: list[dict] = []
            eps_left = EPS_PER_ALERT
            for _ in range(k):
                avail = [c for c in options if c.program not in asked and c.family not in families and c.eps <= eps_left + 1e-9]
                if not avail:
                    break
                c = rng.choice(avail)
                asked.add(c.program)
                r = self.banks[bank].handle(Question(c.program, c.eps, a.binding), a.binding.t0, purpose=f"s1-{i}")
                if isinstance(r, Reply):
                    eps_left -= r.eps_spent
                    families.add(c.family)
                    lr = c.lr.get(c.cell(r.value))
                    answers.append({"program": c.program, "family": c.family, "value": r.value, "lr": lr, "eps": c.eps})
            answered = [(x["family"], x["lr"]) for x in answers if x["lr"]]
            posterior = float(sigmoid(fused_logit(self.fusion, a.p0, answered)))
            label, gain = worth_asking_label(self.fusion, a.p0, answered, options, asked, eps_left, self.gain_min)
            self.gains.setdefault(a.split, []).append(gain)
            text = ctxt.step_text(ctx.summary, a.p0, posterior, self.threshold, eps_left, answers, menu_items(options, bank))
            yield {**base, "decision": "worth_asking", "context": text, "label": "yes" if label else "no"}

    def _rule_rows(self, a, rng, base):
        for i in range(self.mix["rule_applies"]):
            prog = rng.choice(self.programs)
            kind = parse(prog).prim.kind
            jur = rng.choice(["US", "US", "SG"])
            permitting = [r for r in self.rules if rule_label(self.gate, r, jur, kind)]
            other = [r for r in self.rules if not rule_label(self.gate, r, jur, kind)]
            rid = rng.choice(permitting if (i % 2 == 0 and permitting) or not other else other)
            rule = self.gate.rules[rid]
            text = rule.text if a.split == "train" else rng.choice(RULE_PARAPHRASE.get(rid, [rule.text]))
            yield {**base, "decision": "rule_applies",
                   "context": ctxt.rule_text(jur, kind, canonical(parse(prog)), rid, rule.jurisdiction, text),
                   "label": "yes" if rule_label(self.gate, rid, jur, kind) else "no"}  # fmt: skip

    def _injection_rows(self, a, rng, base):
        items = self.injections["train" if a.split == "train" else "val"]
        for _ in range(self.mix["injection_present"]):
            item = rng.choice(items)
            yield {**base, "decision": "injection_present", "context": ctxt.note_text(a.binding.partner_bank, item.note),
                   "label": "yes" if item.injection_present else "no"}  # fmt: skip

    def build(self, alerts: Iterable[BtAlert], max_total: int | None = None) -> Iterator[dict]:
        n = 0
        for a in alerts:
            if a.split == "test":
                raise ValueError("the test split is never used for the decision set")
            if a.alert_id not in self.contexts:
                continue
            for row in self.rows_for(a):
                yield row
                n += 1
                if max_total is not None and n >= max_total:
                    return


# --- writing ----------------------------------------------------------------------------------------------------
def write_set(rows: Iterable[dict], out_dir: str | Path, name: str = "decisions") -> dict:
    """One JSONL file per split plus manifest.json (counts, label balance, sha256). Not committed (README rule 7)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files, counts, labels = {}, Counter(), Counter()
    handles = {}
    try:
        for r in rows:
            f = handles.get(r["split"]) or handles.setdefault(r["split"], (out / f"{name}_{r['split']}.jsonl").open("w", encoding="utf-8"))
            f.write(json.dumps(r) + "\n")
            counts[(r["split"], r["decision"])] += 1
            labels[(r["decision"], r["label"])] += 1
    finally:
        for f in handles.values():
            f.close()
    for split in handles:
        p = out / f"{name}_{split}.jsonl"
        files[split] = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest = {"files": files, "n": sum(counts.values()),
                "by_split_decision": {f"{s}/{d}": n for (s, d), n in sorted(counts.items())},
                "labels": {f"{d}/{lab}": n for (d, lab), n in sorted(labels.items())}}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
