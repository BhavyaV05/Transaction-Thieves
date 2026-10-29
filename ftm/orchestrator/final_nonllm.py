"""The G5 runner for T2's non-LLM arms: the arms of G4 (week 9) on the frozen test alerts, blind.

Owner: T2 (Bhavya).  Spec: docs/schedule_nov10.md (phase C: "All non-LLM arms, and the banks for every LLM arm (CPU)"),
docs/t4_final_run.md (an `external_cmd` job writes `{out}/runlog.parquet`, run-log v1, no labels), docs/t2_final_run.md.

Nothing is configured twice. An arm is built by `ftm.orchestrator.ablations.prepare_arm`, the function `run_ablations` (G4)
builds it with, from the same `configs/experiments/ablations_nonllm.yaml`; what differs is only where it runs:

                 G4 (validation)                      G5 (this module)
    scored on    the `eval` half of val               the frozen alert ids of the final run (test)
    fitted on    the `fit` half of val                the `fit` half of val (the same alerts, the same cut)
    tables       the back-test tables                 the same tables, not rebuilt, not re-accepted
    labels       in the rows, Brier in the summary    in no row and no summary (`blind`): the analysis joins them in

An alert with no modelled partner bank (`load_world` cannot bind it) still gets a verdict row, with no probability, so it is
scored as p0 like every other system's no-verdict (`stop_reason` = no_partner_bank), and every arm covers the same ids.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ftm.backtest.engine import BtAlert
from ftm.banks.binding import alert_context
from ftm.harness.runlog_rows import to_frame, verdict_rows
from ftm.orchestrator.ablations import AblationSpec, Arm, _refused_share, prepare_arm
from ftm.orchestrator.raw_share import RawShareArm, RawShareBank
from ftm.orchestrator.runner import run_alerts
from ftm.planners.learned import LearnedPolicy, split_val
from ftm.planners.pool import QuestionPool
from ftm.types import Limits

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "final" / "non_llm.yaml"
RAW_SHARE = "raw_share"  # the H2 arm: not a planner, so not in ablations_nonllm.yaml
NO_PARTNER = "no_partner_bank"
POOL_NAMES = ("realistic", "enumerator", "exact_labels")


@dataclass
class FinalConfig:
    data: str
    tables: dict[str, str | None]  # pool name -> lr_table.jsonl (built by scripts/backtest.py --assemble on TRAIN alerts)
    policy: str | None = None  # a fitted LearnedPolicy (scripts/train_policy.py); None = fit it on the fit half of val
    path: str = ""
    sha256: str = ""  # of the file, recorded in the run's summary

    def table_path(self, pool: str, root: Path = ROOT) -> Path | None:
        p = self.tables.get(pool)
        return None if not p else (root / p.format(data=self.data) if not Path(p).is_absolute() else Path(p.format(data=self.data)))


def load_config(path: str | Path = CONFIG) -> FinalConfig:
    text = Path(path).read_text(encoding="utf-8")
    c = yaml.safe_load(text)
    unknown = set(c.get("tables", {})) - set(POOL_NAMES)
    if unknown:
        raise ValueError(f"unknown question tables {sorted(unknown)}; known: {POOL_NAMES}")
    return FinalConfig(data=c["data"], tables={n: c.get("tables", {}).get(n) for n in POOL_NAMES}, policy=c.get("policy"),
                       path=str(path), sha256=hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest())  # fmt: skip


def arm_names(spec: AblationSpec) -> list[str]:
    return [a.name for a in spec.arms] + [RAW_SHARE]


def find_arm(spec: AblationSpec, name: str) -> Arm:
    for a in spec.arms:
        if a.name == name:
            return a
    raise ValueError(f"unknown arm {name!r}; the arms are {arm_names(spec)}")


def needs_fit(arm: Arm, cfg: FinalConfig) -> bool:
    """Does this arm fit something on the validation alerts (so the val world has to be loaded)?"""
    s = arm.settings
    return bool(s["fusion"]) or (arm.system == "learned" and not cfg.policy)


def check_config(cfg: FinalConfig, spec: AblationSpec, root: Path = ROOT) -> list[str]:
    """Problems that would stop the final run before it starts (nothing is loaded): a table path that is not set or does not
    exist for a pool some arm uses, a policy path that does not exist."""
    problems = []
    used = {a.settings["pool"] for a in spec.arms}
    for pool in sorted(used):
        p = cfg.table_path(pool, root)
        if p is None:
            problems.append(f"configs/final/non_llm.yaml: tables.{pool} is not set (arms using it: "
                            f"{[a.name for a in spec.arms if a.settings['pool'] == pool]})")  # fmt: skip
        elif not p.exists():
            problems.append(f"tables.{pool}: {p} does not exist (scripts/backtest.py --assemble writes it)")
    if cfg.policy and not (root / cfg.policy).exists():
        problems.append(f"policy: {cfg.policy} does not exist (scripts/train_policy.py writes it)")
    return problems


def load_pools(cfg: FinalConfig, arm: Arm, root: Path = ROOT) -> dict[str, QuestionPool]:
    """Only the table the arm uses."""
    pool = arm.settings["pool"]
    p = cfg.table_path(pool, root)
    if p is None or not p.exists():
        raise FileNotFoundError(f"arm {arm.name!r} needs the {pool!r} question table: set tables.{pool} in "
                                f"{cfg.path or 'configs/final/non_llm.yaml'} (now {p})")  # fmt: skip
    return {pool: QuestionPool.from_jsonl(p)}


@dataclass
class ArmResult:
    rows: list[dict]
    counts: dict = field(default_factory=dict)  # counts and means only: nothing that depends on a label


def unbound_rows(unbound: Sequence[tuple[str, str, float]], base: dict, split: str) -> list[dict]:
    """Verdict rows with no probability (scored as p0) for alerts the world could not bind to a partner bank."""
    return [{**base, "split": split, "alert_id": alert_id, "event": "verdict", "probability": None, "label": None,
             "home_bank": str(home_bank), "p0": float(p0), "eps_spent": 0.0, "sim_days": 0.0, "stop_reason": NO_PARTNER,
             "trials": 1} for alert_id, home_bank, p0 in unbound]  # fmt: skip


def _counts(res_summary: dict, n_bound: int, n_unbound: int, refused_share: float) -> dict:
    return {"alerts": n_bound + n_unbound, "with_partner_bank": n_bound, "no_partner_bank": n_unbound,
            "mean_questions": res_summary["mean_questions"], "mean_eps": res_summary["mean_eps"],
            "refused_share": refused_share, "mean_sim_days": res_summary["mean_sim_days"],
            "p95_sim_days": res_summary["p95_sim_days"]}  # fmt: skip


def run_arm(arm: Arm, spec: AblationSpec, *, banks: dict, alerts: Sequence[BtAlert], fit: Sequence[BtAlert],
            fit_banks: dict | None, pools: dict[str, QuestionPool], policy: LearnedPolicy | None, seed: int, base: dict,
            split: str, unbound: Sequence[tuple[str, str, float]] = (), cache: dict | None = None) -> ArmResult:  # fmt: skip
    """One arm on `alerts` (in time order), blind. `base`: run_id, commit, machine; system and variant are set here."""
    base = {**base, "system": arm.system, "variant": arm.name, "seed": seed}
    prep = prepare_arm(arm, spec, banks=banks, bindings={a.alert_id: a.binding for a in alerts}, fit=fit, fit_banks=fit_banks,
                       pools=pools, policy=policy, seed=seed, cache=cache if cache is not None else {})  # fmt: skip
    if isinstance(prep, str):
        raise RuntimeError(f"arm {arm.name!r} cannot run: {prep}")
    res = run_alerts(prep.planner, list(alerts), prep.harness, prep.limits, base, split, blind=True)
    rows = res.rows + unbound_rows(unbound, base, split)
    return ArmResult(rows, _counts(res.summary, len(alerts), len(unbound), _refused_share(res.verdicts)))


def run_raw_share(*, train_banks: dict, train: Sequence[BtAlert], banks: dict, alerts: Sequence[BtAlert], seed: int, base: dict,
                  split: str, unbound: Sequence[tuple[str, str, float]] = (), ridges: tuple | None = None) -> ArmResult:  # fmt: skip
    """The H2 arm: fitted on `train` through the `train_banks`, applied to `alerts` through `banks`. Three banks share raw
    records, no noise, no budget; it is the one arm that asks the gate for `request_records` (allow_raw_share, this arm only).
    `train` is the fit half of val (what the fused arms fit on, so the base rate is the one the arm is scored at); `ridges`, if
    given, are chosen from by time-ordered CV on it. A train-split fit is the shifted variant (train prevalence 0.19, val 0.45)."""
    from ftm.compliance.gate import Gate

    def raw(world_banks: dict) -> dict:
        return {b: RawShareBank(b, bk._ledger, bk.cfg, bk.seed, bk._entity_of) for b, bk in world_banks.items()}

    gate = Gate.from_config(allow_raw_share=True)
    fitted = RawShareArm(raw(train_banks), gate, seed=seed, ridges=ridges).fit(list(train))
    arm = RawShareArm(raw(banks), gate, seed=seed, model=fitted.model, bindings={a.alert_id: a.binding for a in alerts})
    base = {**base, "system": RAW_SHARE, "variant": RAW_SHARE, "seed": seed}
    rows = []
    for a in alerts:
        ctx = alert_context(a.binding, a.p0, split)
        v = arm.investigate(ctx, None, Limits())
        rows += verdict_rows(v, {**base, "split": split, "alert_id": a.alert_id}, None, ctx=ctx,
                             facts={"eps_spent": 0.0, "sim_days": 0.0, "stop_reason": "raw_share"})  # fmt: skip
    rows += unbound_rows(unbound, base, split)
    counts = {"alerts": len(alerts) + len(unbound), "with_partner_bank": len(alerts), "no_partner_bank": len(unbound)}
    return ArmResult(rows, counts)


def write_run(out: Path, result: ArmResult, meta: dict) -> Path:
    """`runlog.parquet` (run-log v1) and `summary.json` (counts and the settings; no label, no Brier)."""
    out.mkdir(parents=True, exist_ok=True)
    frame = to_frame(result.rows)
    if frame["label"].null_count() < frame.height:  # cannot happen: a blind run writes no label
        raise RuntimeError("a final-run log must not carry labels")
    path = out / "runlog.parquet"
    frame.write_parquet(path)
    (out / "summary.json").write_text(json.dumps({**meta, "counts": result.counts}, indent=1, default=str), encoding="utf-8")
    return path


def split_fit(val: Sequence[BtAlert], spec: AblationSpec) -> list[BtAlert]:
    """The fit half of val: the same cut G4 makes (`split_val`), so a fusion or policy fitted here is the one G4 validated."""
    fit, _ = split_val(list(val), spec.fit_frac)
    return fit
