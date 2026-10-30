"""T2's non-LLM arms on the frozen final-run alerts: the command behind `nonllm.<arm>.<subset|full_test>` of the G5 job plan.

Owner: T2 (Bhavya).  Specs: docs/t2_final_run.md, ftm/orchestrator/final_nonllm.py, docs/t4_final_run.md (the job contract).
Called by scripts/run_final.py (configs/final/jobs.yaml, `external_cmd.templates.non_llm_arm`), never by hand on the test split:

    python scripts/final_non_llm.py --arm eig --alert-ids <ids.txt> --out <job dir> --data hi_medium --seed 0 ...

Writes `<out>/runlog.parquet` (run-log v1: one verdict row per id in the file, plus one step row per question; the `label` column
stays empty) and `<out>/summary.json` (counts and settings, no Brier score). It never sets FTM_FINAL_RUN: `--split test` only loads
when scripts/run_final.py set it; without that the world loader refuses.

    --check                 config and arm list only: tables exist, policy exists (no data is read)
    --list                  the arm names
    --split val             rehearsal on validation alerts (the ids file then holds val ids); same code path
    --data synthetic        the dry run of scripts/run_final.py: the ids are laid over a synthetic world, never a result
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from ftm.orchestrator import final_nonllm as fn  # noqa: E402
from ftm.orchestrator.ablations import CONFIG as ABLATIONS  # noqa: E402
from ftm.orchestrator.ablations import load_spec  # noqa: E402
from ftm.orchestrator.raw_share import RIDGES  # noqa: E402
from ftm.planners.learned import LearnedPolicy  # noqa: E402


def read_ids(path: Path) -> list[str]:
    ids = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: need a non-empty list of distinct alert ids")
    return ids


def _unbound(data: str, split: str, missing: list[str]) -> list[tuple[str, str, float]]:
    """(alert_id, home bank, p0) of ids the world could not bind to a partner bank, from the alert table."""
    from ftm.data import io

    cfg = io.load_config(data)
    t = pl.read_parquet(ROOT / cfg["alerts_dir"] / f"{split}.parquet", columns=["alert_id", "bank", "p0"])  # time-guard: not transactions
    t = t.filter(pl.col("alert_id").is_in(missing))
    if t.height != len(missing):
        gone = sorted(set(missing) - set(t["alert_id"].to_list()))
        raise SystemExit(f"{len(gone)} ids are not in the {split} alert table (first: {gone[0]})")
    return [(r["alert_id"], str(r["bank"]), float(r["p0"])) for r in t.iter_rows(named=True)]


def real_run(a, cfg, spec, arm_name: str, ids: list[str], base: dict) -> fn.ArmResult:
    from ftm.backtest.world import load_world

    data = a.data or cfg.data
    want = set(ids)
    target = load_world(data, a.split, a.seed)
    alerts = [x for x in target.alerts if x.alert_id in want]
    missing = sorted(want - {x.alert_id for x in alerts})
    unbound = _unbound(data, a.split, missing) if missing else []
    if arm_name == fn.RAW_SHARE:
        if a.raw_fit == "train":  # the shifted variant: fitted on train, one ridge
            train = load_world(data, "train", a.seed)
            return fn.run_raw_share(train_banks=train.banks, train=train.alerts, banks=target.banks, alerts=alerts, seed=a.seed,
                                    base=base, split=a.split, unbound=unbound)  # fmt: skip
        val = target if a.split == "val" else load_world(data, "val", a.seed)  # the arm: the fit half of val, ridge by time-ordered CV
        fit = fn.split_fit(val.alerts, spec)
        if a.split == "val" and {x.alert_id for x in fit} & {x.alert_id for x in alerts}:
            raise SystemExit("raw_share is fitted on the fit half of val: score it on the eval half (--alert-ids), not on its own fit alerts")
        return fn.run_raw_share(train_banks=val.banks, train=fit, banks=target.banks, alerts=alerts,
                                seed=a.seed, base=base, split=a.split, unbound=unbound, ridges=RIDGES)  # fmt: skip
    arm = fn.find_arm(spec, arm_name)
    pools = fn.load_pools(cfg, arm)
    fit, fit_banks = [], None
    if fn.needs_fit(arm, cfg):
        val = target if a.split == "val" else load_world(data, "val", a.seed)  # one copy of the val world when scoring on val
        fit, fit_banks = fn.split_fit(val.alerts, spec), val.banks
    policy = LearnedPolicy.load(ROOT / cfg.policy) if cfg.policy and arm.system == "learned" else None
    return fn.run_arm(arm, spec, banks=target.banks, alerts=alerts, fit=fit, fit_banks=fit_banks, pools=pools, policy=policy,
                      seed=a.seed, base=base, split=a.split, unbound=unbound)  # fmt: skip


def synthetic_run(a, spec, arm_name: str, ids: list[str] | None, base: dict) -> fn.ArmResult:
    """The ids laid over the synthetic world's eval alerts in time order (the rest: no partner bank). A rehearsal, never a result."""
    from ftm.backtest.synth import make_world
    from ftm.orchestrator.ablations import synthetic_env
    from ftm.planners.learned import split_val

    n = a.synthetic_n
    if arm_name == fn.RAW_SHARE:
        w = make_world(n, a.seed, 1.0)
        alerts = sorted(w.alerts, key=lambda x: x.binding.t0)
        cut = int(len(alerts) * 0.6)
        rate = sum(x.label for x in alerts[:cut]) / cut
        train, ev = [replace(x, p0=rate) for x in alerts[:cut]], [replace(x, p0=rate) for x in alerts[cut:]]
        banks, fit, fit_banks, pools = w.banks, [], None, {}
    else:
        pool_name = fn.find_arm(spec, arm_name).settings["pool"]
        # the thinned enumeration accepts nothing in a small world; the exact-label table is the realistic one here (rehearsal)
        env = synthetic_env(max(n, 600) if pool_name == "enumerator" else n, a.seed, spec.eps_levels[1:],
                            enumerator=pool_name == "enumerator")  # fmt: skip
        fit, ev = split_val(env.val, spec.fit_frac)
        banks, fit_banks, train = env.banks, env.banks, []
        pools = {**env.pools, "exact_labels": env.pools["realistic"]}
    ids = ids or [x.alert_id for x in ev]
    use = ev[: len(ids)]
    alerts = [replace(x, alert_id=new, binding=replace(x.binding, alert_id=new)) for new, x in zip(ids, use, strict=False)]
    p0 = sum(x.p0 for x in ev) / len(ev)
    unbound = [(i, "0", p0) for i in ids[len(use):]]
    if arm_name == fn.RAW_SHARE:
        return fn.run_raw_share(train_banks=banks, train=train, banks=banks, alerts=alerts, seed=a.seed, base=base,
                                split=a.split, unbound=unbound)  # fmt: skip
    return fn.run_arm(fn.find_arm(spec, arm_name), spec, banks=banks, alerts=alerts, fit=fit, fit_banks=fit_banks, pools=pools,
                      policy=None, seed=a.seed, base=base, split=a.split, unbound=unbound)  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=None, help="an arm of configs/experiments/ablations_nonllm.yaml, or raw_share")
    ap.add_argument("--alert-ids", type=Path, default=None, help="the frozen id file (one alert id per line)")
    ap.add_argument("--out", type=Path, default=None, help="job directory; runlog.parquet and summary.json go here")
    ap.add_argument("--data", default=None, help="hi_small | hi_medium | synthetic (default: the config's)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default="", help="run id, commit and machine are copied into the run-log rows")
    ap.add_argument("--commit", default="")
    ap.add_argument("--machine", default="")
    ap.add_argument("--config", default=str(fn.CONFIG))
    ap.add_argument("--ablations", default=str(ABLATIONS))
    ap.add_argument("--raw-fit", default="val", choices=["val", "train"],
                    help="raw_share only: fit on the fit half of val (the arm) or on the train split (the shifted variant)")
    ap.add_argument("--synthetic-n", type=int, default=240, help="size of the synthetic world for --data synthetic")
    ap.add_argument("--check", action="store_true", help="validate the config and the arm list; read no data")
    ap.add_argument("--list", action="store_true", help="print the arm names")
    a = ap.parse_args(argv)

    spec = load_spec(a.ablations)
    cfg = fn.load_config(a.config)
    if a.list:
        print("\n".join(fn.arm_names(spec)))
        return 0
    if a.check:
        problems = fn.check_config(cfg, spec)
        for p in problems:
            print("PROBLEM:", p)
        verdict = "ok" if not problems else f"{len(problems)} problems"
        print(f"{len(spec.arms)} arms + {fn.RAW_SHARE}; config {cfg.path} ({cfg.sha256[:12]}): {verdict}")
        return 1 if problems else 0
    if not (a.arm and a.out):
        ap.error("--arm and --out are required (or --check / --list)")
    if a.arm not in fn.arm_names(spec):
        ap.error(f"unknown arm {a.arm!r}; the arms are {fn.arm_names(spec)}")
    synthetic = (a.data or cfg.data) == "synthetic"
    if a.alert_ids is None and not synthetic:
        ap.error("--alert-ids is required (the frozen set)")
    ids = read_ids(a.alert_ids) if a.alert_ids else None
    base = {"run_id": a.run_id, "commit": a.commit, "machine": a.machine}
    if synthetic:
        a.split = "val"
    res = synthetic_run(a, spec, a.arm, ids, base) if synthetic else real_run(a, cfg, spec, a.arm, ids, base)
    meta = {"arm": a.arm, "data": "synthetic" if synthetic else (a.data or cfg.data), "split": a.split, "seed": a.seed,
            "n_ids": len(ids) if ids else None, "config": cfg.path, "config_sha256": cfg.sha256, "run_id": a.run_id,
            "commit": a.commit, "machine": a.machine, "blind": True}  # fmt: skip
    if ids:
        from ftm.final.runner import ids_sha256

        meta["ids_sha256"] = ids_sha256(ids)
    path = fn.write_run(a.out, res, meta)
    print(f"{a.arm}: {res.counts['alerts']} alerts ({res.counts['no_partner_bank']} without a partner bank) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
