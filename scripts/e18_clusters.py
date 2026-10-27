"""alert_id -> cluster map for the E18 bootstrap (prereg/e18_decision.yaml `clusters: attempt+accounts`).

Clusters are built over every askable val alert of the dataset at once (all parts), so stage 1 (eval) and stage 2 (all)
use the same labels. Same rule as G2 (`ftm.eval.g2.clusters_for`): alerts that share a laundering attempt or a sender /
receiver account (the harness binding's real ids, server-side) are one cluster. No hub cap for alerts
(`docs/t1_handoffs.md`: largest alert cluster 1.5 % on HI-Medium eval, 2.5 % on all val).

    python scripts/e18_clusters.py --data hi_medium --out outputs/e18/clusters.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_llm_agent import askable_ids, make_toolbox  # noqa: E402

from ftm.eval import bootstrap as bs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="hi_medium")
    ap.add_argument("--hub-cap", type=int, default=None, help="sensitivity check only; the E18 rule uses none")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "e18" / "clusters.json")
    a = ap.parse_args(argv)

    tools, _ = make_toolbox("harness", 0, a.data, "val")
    ids = askable_ids(tools)
    bindings = tools._h.bindings  # noqa: SLF001 - real ids stay server-side; only cluster labels leave this script
    alerts = pl.read_parquet(ROOT / "data" / "alerts" / a.data / "val.parquet").filter(pl.col("alert_id").is_in(sorted(ids)))
    alerts = alerts.sort("alert_id")
    aid = alerts["alert_id"].to_list()
    labels = bs.cluster_ids(alerts["attempt_id"].to_list(), [[bindings[i].sender for i in aid], [bindings[i].receiver for i in aid]],
                            a.hub_cap)  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({i: int(c) for i, c in zip(aid, labels, strict=True)}, indent=0))
    rep = bs.cluster_report(labels)
    print(f"e18 clusters [{a.data}]: {len(aid)} askable val alerts, {rep} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
