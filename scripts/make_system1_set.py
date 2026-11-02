"""Build the System 1 decision set against the tuned pipeline (wave 4): labels from the fitted fusion and the routed menu.

Owner: T3 (Akshat).  Spec: docs/t3_system1.md.  CPU only; run from the repo root with FTM_P0_VARIANT unset (v2 is set here).

    python scripts/make_system1_set.py --out data/decisions/hi_medium_s1v1            # train + val, all askable alerts
    python scripts/make_system1_set.py --out /tmp/s1 --max-alerts 40 --splits val     # smoke

Writes <out>/decisions_{train,val}.jsonl and manifest.json (counts, label balance, worth_asking gain quantiles, sha256).
Train rows are for SFT; val rows carry the alert's `val_part` (fit / cal / eval) through the alert table for calibration, the
conformal sets and the report.  Never reads the test split.  The output is data: do not commit it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402
import yaml  # noqa: E402

from ftm.backtest.world import load_world  # noqa: E402
from ftm.fusion.conditional import ConditionalFusion  # noqa: E402
from ftm.orchestrator.decision_set import DecisionSetBuilder, write_set  # noqa: E402
from ftm.planners.pool import QuestionPool  # noqa: E402


def run_llm_agent():
    spec = importlib.util.spec_from_file_location("run_llm_agent", ROOT / "scripts" / "run_llm_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="hi_medium")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--table", type=Path, default=ROOT / "configs/fusion/headroom_v1_table.jsonl")
    ap.add_argument("--fusion", type=Path, default=ROOT / "configs/fusion/headroom_v1.json")
    ap.add_argument("--decision", type=Path, default=ROOT / "configs/decision/headroom_v1.yaml")
    ap.add_argument("--gain-min", type=float, default=None, help="worth_asking threshold (absolute Brier); default decision_set.GAIN_MIN")
    ap.add_argument("--max-alerts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    os.environ["FTM_P0_VARIANT"] = "v2"
    threshold = float(yaml.safe_load(a.decision.read_text())["threshold"])
    pool = QuestionPool.from_jsonl(a.table)
    fusion = ConditionalFusion.from_json(a.fusion)
    rla = run_llm_agent()
    ledgers, banks, alerts, contexts = {}, {}, [], {}
    parts: dict[str, str] = {}
    from dataclasses import replace

    from ftm.data import io

    cfg = io.load_config(a.data)
    for split in a.splits:
        w = load_world(a.data, split, a.seed, max_alerts=a.max_alerts)
        ledgers.update(w.ledgers)
        banks.update(w.banks)
        v2 = pl.read_parquet(ROOT / io.alerts_dir(cfg, "v2") / f"{split}.parquet")  # time-guard: not transactions
        p0 = dict(zip(v2["alert_id"].to_list(), v2["p0"].to_list(), strict=True))
        if split == "val":
            parts = dict(zip(v2["alert_id"].to_list(), v2["val_part"].to_list(), strict=True))
        alerts += [replace(x, p0=float(p0[x.alert_id])) for x in w.alerts]
        ids = {x.alert_id for x in w.alerts}
        got = rla.data_alerts(a.data, split, 10**9, a.seed, 1, "all" if split == "val" else None, ids, 0, None, p0_variant="v2")
        contexts.update({c.alert_id: c for c, _ in got})
        print(f"{split}: {len(w.alerts)} askable alerts, {len(got)} contexts", flush=True)
    kw = {"gain_min": a.gain_min} if a.gain_min is not None else {}
    builder = DecisionSetBuilder(ledgers, banks, pool, fusion, contexts, threshold, seed=a.seed, **kw)
    rows = []
    for r in builder.build(alerts):
        if r["split"] == "val":
            r = {**r, "val_part": parts[r["alert_id"]]}
        rows.append(r)
    manifest = write_set(rows, a.out)
    manifest["config"] = {"table": str(a.table), "fusion": str(a.fusion), "threshold": threshold, "gain_min": builder.gain_min,
                          "typology_none_keep": builder.none_keep, "mix": builder.mix, "seed": a.seed, "p0_variant": "v2"}
    import numpy as np

    manifest["worth_asking_best_gain_quantiles"] = {sp: {str(q): float(np.quantile(g, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
                                                     for sp, g in builder.gains.items() if g}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("n", "by_split_decision", "labels")}, indent=1))
    return manifest


if __name__ == "__main__":
    main()
