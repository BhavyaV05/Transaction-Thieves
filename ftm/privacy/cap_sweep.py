"""How often does the per-partner-account cap bite? (E14 "per-account cap on/off", and the choice of its value.)

Owner: T2 (Bhavya).  Spec: tracks.md 4.5 (per-partner-account cap), 4.11 (E14).

`simulate_cap` replays a stream of alerts, each charging `questions_per_alert` questions of `eps` to the partner accounts
its questions can see, against the real `PrivacyAccountant`, for several caps. `repeat_stats` says how often one partner
account is the subject of several alerts, which is what decides the cap. Input is only (alert id, account keys), so it
runs on the alert sets without any answers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from ftm.privacy.accountant import AccountKey, BudgetExceeded, PrivacyAccountant

Stream = Sequence[tuple[str, list[AccountKey]]]


def subject_stream(alerts) -> list[tuple[str, list[AccountKey]]]:
    """(alert id, [(partner bank, partner-side account)]) for back-test alerts (`BtAlert`), in alert-time order."""
    out = []
    for a in sorted(alerts, key=lambda a: (a.binding.t0, a.alert_id)):
        b = a.binding
        if b.partner_bank is None:
            continue
        subject = b.receiver if b.sender_bank == b.home_bank else b.sender
        out.append((a.alert_id, [(b.partner_bank, subject)]))
    return out


def repeat_stats(stream: Stream) -> dict:
    """How many alerts each partner account is the subject of."""
    per = Counter(k for _, keys in stream for k in set(keys))
    return {
        "alerts": len(stream), "accounts": len(per), "max_alerts_per_account": max(per.values(), default=0),
        "accounts_in_2plus_alerts": sum(v >= 2 for v in per.values()),
        "accounts_in_3plus_alerts": sum(v >= 3 for v in per.values()),
        "accounts_in_5plus_alerts": sum(v >= 5 for v in per.values()),
        "alerts_touching_a_repeat_account": sum(any(per[k] >= 2 for k in keys) for _, keys in stream),
    }  # fmt: skip


def simulate_cap(stream: Stream, caps: Iterable[float | None], questions_per_alert: int = 2, eps: float = 1.0,
                 eps_per_alert: float = 2.0) -> list[dict]:  # fmt: skip
    """For each cap (None = off): how many questions the accountant refused, and how hard the busiest account was hit."""
    rows = []
    for cap in caps:
        acc = PrivacyAccountant(eps_per_alert, cap)
        asked = refused = alerts_hit = 0
        for aid, keys in stream:
            hit = False
            for _ in range(questions_per_alert):
                asked += 1
                try:
                    acc.spend(aid, keys[0][0] if keys else None, eps, keys)
                except BudgetExceeded:
                    refused += 1
                    hit = True
            alerts_hit += hit
        rep = acc.report()
        rows.append({"cap": cap, "questions": asked, "refused": refused, "refused_share": refused / max(asked, 1),
                     "alerts_with_a_refusal": alerts_hit, "max_account_eps": rep["max_account_eps"],
                     "refusal_reasons": rep["refusals"]})  # fmt: skip
    return rows
