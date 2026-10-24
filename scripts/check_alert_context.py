"""Check that the alert sets' `direction` / `counterparty_bank` columns, `views.alert_summary` and T2's
`ftm.banks.binding.alert_context` agree on a seeded sample of real alerts (train and val).

  python scripts/check_alert_context.py --data hi_small --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.banks.binding import alert_context, bind_alert  # noqa: E402
from ftm.data import alias, io  # noqa: E402
from ftm.retrieval import views  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = io.load_config(a.data)
    aliases = alias.read_aliases(cfg["parquet_dir"], seed=0)
    bad = checked = no_cp = with_tf = 0
    for split in ("train", "val"):
        al = pl.read_parquet(Path(cfg["alerts_dir"]) / f"{split}.parquet")  # time-guard: not transactions
        tf = pl.read_parquet(Path(cfg["alerts_dir"]) / f"top_features_{split}.parquet") if (Path(cfg["alerts_dir"]) / f"top_features_{split}.parquet").exists() else None  # fmt: skip
        for r in al.sample(min(a.n, al.height), seed=a.seed).iter_rows(named=True):
            view = io.scan_view(cfg["views_dir"], r["bank"], as_of=r["timestamp"]).collect()  # cut at the alert time
            tx = view.filter(pl.col("tx_id") == r["tx_id"]).row(0, named=True)
            ctx = alert_context(bind_alert(r["alert_id"], tx, r["bank"], view), r["p0"], split)
            _, _ = views.top_k_by_amount_view(cfg["views_dir"], r["bank"], tx, aliases, k=5)  # stamps home_bank
            s = views.alert_summary(tx, aliases)
            want = (ctx.summary["direction"], ctx.summary["counterparty_bank"])
            got = [(r["direction"], r["counterparty_bank"]), (s["direction"], s["counterparty_bank"])]
            checked += 1
            no_cp += want[1] is None
            if any(g != want for g in got):
                bad += 1
                print("MISMATCH", r["alert_id"], want, got)
            if tf is not None:
                t = tf.filter(pl.col("alert_id") == r["alert_id"])["top_features"][0]
                with_tf += (t is not None) and ("top_features" in s) and len(s["top_features"]) == len(t) == 5
    print(f"{a.data}: checked {checked} alerts, mismatches {bad}, alerts without a counterparty bank {no_cp}, "
          f"summaries with 5 top_features {with_tf}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
