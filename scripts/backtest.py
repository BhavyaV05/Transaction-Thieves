"""Back-test a batch of DSL questions (shardable); writes the shared cache, then assembles the LR table.

Owner: T2 (Bhavya).  Spec: plan.md 6.8, tracks.md 4.6.  Run from the repo root.

    # shard i of n on any machine; all shards write the same cache (data/backtest/<ds>/<namespace>/)
    python scripts/backtest.py --data hi_small --eps 0.5 1 2 --shard 0/4
    python scripts/backtest.py --data hi_small --eps 0.5 1 2 --shard 1/4 ...
    # after all shards: e-BH over everything in the cache, write the LR table and a report
    python scripts/backtest.py --data hi_small --eps 0.5 1 2 --assemble

Questions: the expert list (`--questions expert`, both subject slots), every DSL program up to a size (`--questions
enumerate --max-size 4`, the Enumerator planner's search space) and/or a JSONL file of DSL programs
(`--questions-file`, T3's hand-off #8: one {"program": "..."} or plain program per line). `--exact` runs the no-noise
variant (eps = inf) on the same alerts. Realistic mode is the default and the one the main results use.
The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.backtest.engine import BacktestCache, BacktestConfig, Backtester, assemble, lr_table_rows  # noqa: E402
from ftm.backtest.world import load_world  # noqa: E402
from ftm.dsl.expert import expert_programs  # noqa: E402
from ftm.dsl.parser import canonical, check, parse  # noqa: E402
from ftm.privacy.accountant import PrivacyAccountant  # noqa: E402


def read_programs(args) -> list[str]:
    progs: list[str] = []
    if "enumerate" in args.questions:
        from ftm.dsl.enumerate import enumerate_programs

        progs += [canonical(n) for n in enumerate_programs(args.max_size, ("receiver", "sender"))]
    if "expert" in args.questions:
        for slot in ("receiver", "sender"):
            progs += [p["program"] for p in expert_programs(slot)]
    if args.questions_file:
        for line in Path(args.questions_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            progs.append(json.loads(line)["program"] if line.startswith("{") else line)
    ok, bad = [], 0
    for p in progs:
        if check(p).ok:
            ok.append(p)
        else:
            bad += 1
    if bad:
        print(f"skipped {bad} programs that do not parse", file=sys.stderr)
    return list(dict.fromkeys(canonical(parse(p)) for p in ok))  # dedup by canonical form


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--questions", nargs="+", default=["expert"], choices=["expert", "enumerate", "none"])
    ap.add_argument("--max-size", type=int, default=4, help="program size bound for --questions enumerate")
    ap.add_argument("--questions-file", default=None)
    ap.add_argument("--shard", default="0/1", help="i/n")
    ap.add_argument("--time-blocks", type=int, default=1,
                    help="> 1: e-values permute labels within this many time blocks (robust to drift; changes the namespace)")
    ap.add_argument("--exact", action="store_true", help="no-noise variant (eps = inf) on the same alerts")
    ap.add_argument("--exact-labels", action="store_true",
                    help="exact mode proper: every bank's labels (a label-stratified sample of cross-bank transactions), no noise")
    ap.add_argument("--heldout-cycles", action="store_true",
                    help="E12: back-test on a train split without the cycle attempts (rows and alerts dropped)")
    ap.add_argument("--n-pos", type=int, default=2000, help="laundering transactions sampled for --exact-labels")
    ap.add_argument("--neg-ratio", type=int, default=4, help="clean per laundering transaction for --exact-labels")
    ap.add_argument("--compare", nargs=2, metavar=("REALISTIC_DIR", "EXACT_DIR"),
                    help="compare two assembled cache directories (realistic vs exact) and print the summary")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-alerts", type=int, default=None)
    ap.add_argument("--only-keep", action="store_true", help="drop alerts flagged keep=False (sensitivity analysis)")
    ap.add_argument("--out", default="data/backtest")
    a = ap.parse_args(argv)

    if a.compare:
        from ftm.backtest.exact import compare_modes

        loaded = [assemble(BacktestCache(Path(d).parent, Path(d).name), a.fdr) for d in a.compare]
        out = compare_modes(*loaded)
        print(json.dumps(out, indent=1))
        return out
    exact = a.exact or a.exact_labels
    if a.exact_labels:
        from ftm.backtest.exact import load_exact_world

        world = load_exact_world(a.data, a.split, a.n_pos, a.neg_ratio, a.seed)
    else:
        world = load_world(a.data, a.split, a.seed, max_alerts=a.max_alerts, only_keep=a.only_keep,
                           heldout_cycles=a.heldout_cycles)
    cfg = BacktestConfig(mode="exact" if exact else "realistic", time_blocks=a.time_blocks)
    ns = cfg.namespace(world.alert_set_id, a.seed)
    cache = BacktestCache(Path(a.out) / Path(a.data).stem, ns)
    eps_levels = [float("inf")] if exact else a.eps
    if a.assemble:
        entries = assemble(cache, a.fdr)
        outdir = cache.dir
        with (outdir / "lr_table.jsonl").open("w", encoding="utf-8") as f:
            for r in lr_table_rows(entries):
                f.write(json.dumps(r) + "\n")
        fam = {}
        for e in entries:
            d = fam.setdefault(e["family"], {"tested": 0, "accepted": 0})
            d["tested"] += 1
            d["accepted"] += bool(e["accepted"])
        report = {"namespace": ns, "alert_set": world.alert_set_id, "n_hypotheses": len(entries),
                  "n_eligible": sum(e["eligible"] for e in entries), "n_accepted": sum(e["accepted"] for e in entries),
                  "fdr": a.fdr, "by_family": fam}  # fmt: skip
        (outdir / "report.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return
    i, n = (int(x) for x in a.shard.split("/"))
    acc = PrivacyAccountant(eps_cal=None)
    bt = Backtester(world.banks, world.alerts, cfg, accountant=acc, cache=cache)
    progs = read_programs(a)
    print(f"{len(world.alerts)} alerts ({world.n_skipped} without a partner bank), {len(progs)} questions, "
          f"eps {eps_levels}, shard {i}/{n}", file=sys.stderr)  # fmt: skip
    entries = bt.run_batch(progs, eps_levels, shard=(i, n))
    (cache.dir / f"eps_cal_shard{i}of{n}.json.txt").write_text(json.dumps(acc.to_dict()))
    print(f"ran {len(entries)} hypotheses; eps_cal spent {acc.cal_spent:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
