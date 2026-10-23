"""The expert question list in the DSL (tracks.md 4.3: "write the ~60 expert questions").

Owner: T2 (Bhavya) writes the programs; T4 (Shreyas) owns the specs they come from
(configs/questions/expert_specs.yaml, one row per family and variant, each citing red flags).

`expert_programs(subject)` returns one program text per spec variant. SUBJ in the specs is the party the ASKED bank
holds: ALERT.receiver for an outgoing alert, ALERT.sender for an incoming one, so the caller picks `subject`.
`entity_accounts` is left out on purpose (spec status `needs_privacy_review`).
"""

from __future__ import annotations

from pathlib import Path

import yaml

SPECS_PATH = Path(__file__).resolve().parents[2] / "configs" / "questions" / "expert_specs.yaml"


def _w(mode: str, h: int) -> str:
    return f"{mode}({h})"


# family -> function(subject, **variant) -> program text
_BUILDERS = {
    "pass_through": lambda s, frac, window_h: f"passed_on({s}, window=after({window_h}), min_frac={frac:g})",
    "onward_bank": lambda s, banks, window_h: f"top_outflow_bank({s}, banks=[70, 1588, 220], window=after({window_h}))",
    "time_to_forward": lambda s: f"time_to_forward({s})",
    "fan_in_count": lambda s, window_h: f"n_counterparties({s}, dir=in, window=before({window_h}))",
    "fan_in_many": lambda s, k, window_h: f"at_least(n_counterparties({s}, dir=in, window=before({window_h})), k={k})",
    "fan_out_count": lambda s, window_h: f"n_counterparties({s}, dir=out, window=after({window_h}))",
    "fan_out_many": lambda s, k, window_h: f"at_least(n_counterparties({s}, dir=out, window=after({window_h})), k={k})",
    "gather_across_banks": lambda s, window_h: f"n_banks({s}, dir=in, window=before({window_h}))",
    "distinct_banks_out": lambda s, window_h: f"n_banks({s}, dir=out, window=after({window_h}))",
    "small_inflows": lambda s, amount, window_h: f"n_tx({s}, dir=in, window=before({window_h}), max_amount={amount})",
    "just_below_threshold": lambda s, window_h: f"n_tx({s}, dir=any, window=around({window_h}), min_amount=9000, max_amount=10000)",
    "round_amounts": lambda s, unit, window_h: f"n_round({s}, window=around({window_h}), unit={unit})",
    "cash_format": lambda s, window_h: f"n_format({s}, window=around({window_h}), format=cash)",
    "return_to_home_bank": lambda s, window_h: f"returns_home({s}, window=after({window_h}))",
    "payees_hit_home_payers": lambda s, window_h: f"overlap({s}, dir=out, window=after({window_h}), among=ALERT.payers)",
    "payers_hit_home_payees": lambda s, window_h: f"overlap({s}, dir=in, window=before({window_h}), among=ALERT.payees)",
    "matching_amount_out": lambda s, tol: f"matching_amount_out({s}, tol={tol:g})",
    "repeated_amounts": lambda s, window_h: f"n_repeated({s}, window=around({window_h}))",
    "velocity_balanced": lambda s, window_h: f"velocity_balanced({s}, window=around({window_h}))",
    "cross_currency_out": lambda s, window_h: f"cross_currency_out({s}, window=after({window_h}))",
    "dormant_then_active": lambda s, days: f"dormant({s}, days={days})",
    "new_account": lambda s, days: f"new_account({s}, days={days})",
    "volume_spike": lambda s, mult: f"volume_spike({s}, mult={mult})",
    "same_entity_transfers": lambda s, window_h: f"n_same_entity({s}, window=around({window_h}))",
    "internal_chain": lambda s, window_h: f"chain_length({s}, window=after({window_h}))",
    "new_counterparties_out": lambda s, window_h: f"n_new_counterparties({s}, window=after({window_h}))",
    "payees_pay_each_other": lambda s, window_h: f"payees_pay_each_other({s}, window=after({window_h}))",
    "inflow_vs_alert": lambda s: f"inflow_bucket({s})",
}
SKIPPED = {"entity_accounts"}  # needs a privacy decision (spec status needs_privacy_review)


def load_specs(path: Path = SPECS_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["families"]


def expert_programs(subject: str = "receiver", specs: list[dict] | None = None) -> list[dict]:
    """One dict per spec variant: family, variant, program (text), kind and sensitivity from the spec."""
    out = []
    for fam in specs if specs is not None else load_specs():
        name = fam["family"]
        if name in SKIPPED or fam.get("status") in ("needs_privacy_review", "restricted"):
            continue
        build = _BUILDERS[name]
        for variant in fam["variants"]:
            out.append(
                {
                    "family": name,
                    "variant": dict(variant),
                    "program": build(f"ALERT.{subject}", **variant),
                    "kind": fam["kind"],
                    "sensitivity": fam["sensitivity"],
                }
            )
    return out


def expert_ranking(subject: str = "receiver") -> list[str]:
    """DSL family keys (`ftm.dsl.parser.family`) in the order of the expert list: the 'expert-ranked order' of the
    fixed-order planner. T4 owns the order of configs/questions/expert_specs.yaml; reorder the yaml to re-rank."""
    from ftm.dsl.parser import family, parse

    return list(dict.fromkeys(family(parse(p["program"])) for p in expert_programs(subject)))
