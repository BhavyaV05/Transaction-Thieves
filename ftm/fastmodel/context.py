"""The text each System 1 decision is asked about, built the same way for the dataset and for the live pipeline (C13).

Owner: T3 (Akshat).  Spec: wave 4 (docs/t3_system1.md).

A fast model is only as good as the match between the text it was trained on and the text it sees in the pipeline. The first
dataset described the alert from the home ledger, which the planner never sees, so its numbers said little about the live
system. These builders take the PlannerView ingredients only (alert summary, p0, the fused posterior, the threshold, eps left,
the answers so far, the routed back-tested menu) and are called by BOTH `ftm.orchestrator.decision_set` (training / calibration
rows) and `ftm.graph.system1` (the pipeline), so a row means the same thing in both places. Aliased text only: no account id,
no name, no date.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

SHAPE_SKIP = {"typology", "receiver"}  # `typology` is the planner's own guess slot; `receiver` is an alias used by one ablation


def _money(x) -> str:
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return str(x)


def alert_text(summary: dict, p0: float) -> str:
    """The alert as the planner sees it, minus the long evidence trail: the transfer, its compact shape, the one-bank score."""
    sh = summary.get("shape") or {}
    shape = ", ".join(f"{k} {v}" for k, v in sh.items() if k not in SHAPE_SKIP)
    t = str(summary.get("time", ""))
    when = f" (hour {t[11:13]})" if len(t) >= 13 else ""
    return (f"Alert: a payment of {_money(summary.get('amount'))} {summary.get('currency', '?')} ({summary.get('format', '?')}), "
            f"{summary.get('direction', '?')} of the home bank{when}. Shape of the counterparty's activity: {shape or 'n/a'}. "
            f"One-bank model score {p0:.2f}.")  # fmt: skip


def _affordable(menu: Iterable, asked: set[str], eps_left: float) -> list:
    return [m for m in menu if m.eps <= eps_left + 1e-9 and m.canonical not in asked]


def menu_stats(menu: Sequence, asked: set[str], eps_left: float) -> tuple[int, int, float, float]:
    """(affordable questions, their families, the largest |log LR| of any answer, the largest per unit of eps)."""
    aff = _affordable(menu, asked, eps_left)
    best, per_eps = 0.0, 0.0
    for m in aff:
        shift = max((abs(math.log(v)) for _, v in m.lr_bins if v and v > 0), default=0.0)
        best, per_eps = max(best, shift), max(per_eps, shift / m.eps if m.eps > 0 else 0.0)
    return len(aff), len({m.family for m in aff}), best, per_eps


def step_text(summary: dict, p0: float, posterior: float, threshold: float, eps_left: float, answers: Sequence[dict],
              menu: Sequence) -> str:
    """The state the planner is about to act on, for `worth_asking`. `answers`: dicts with family, value and lr (or None);
    `menu`: objects with canonical, eps, family, lr_bins [(cell, LR)] (ftm.graph.policy.MenuItem)."""
    lines = [f"- {a.get('family') or '?'}: answer {a.get('value')!r}, LR {a['lr']:.2f}" if a.get("lr") else
             f"- {a.get('family') or '?'}: answer {a.get('value')!r}, not used by the fusion" for a in answers]
    asked = {a["program"] for a in answers if a.get("program")}
    n, fams, best, per_eps = menu_stats(menu, asked, eps_left)
    side = "above" if posterior >= threshold else "below"
    return (alert_text(summary, p0)
            + f"\nPosterior {posterior:.3f} ({side} the escalation threshold {threshold:.3f}); eps left {eps_left:g}."
            + f"\nAnswers so far ({len(answers)}):\n" + ("\n".join(lines) if lines else "- none")
            + f"\nAffordable back-tested questions: {n} in {fams} families; largest single-answer log-LR {best:.2f}, "
              f"per unit eps {per_eps:.2f}.")  # fmt: skip


def note_text(bank, note: str) -> str:
    return f"Partner bank {bank} replied to a question with an answer and this note: \"{note}\""


def rule_text(jurisdiction: str, kind: str, program: str, rule_id: str, rule_jurisdiction: str, text: str) -> str:
    return (f"Home bank jurisdiction: {jurisdiction}. Candidate question to a partner bank (answer kind {kind}): {program}. "
            f"Cited rule {rule_id} ({rule_jurisdiction}): {' '.join(text.split())}")
