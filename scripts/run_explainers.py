"""Week 10 (C2): GNNExplainer and PGExplainer masks of the home bank's Multi-GNN one-bank model for every val `cal`+`eval` alert.

Owner: T1 (Abhishek).  Writes the cache that `evidence_views.view(alert, "gnnexplainer" | "pgexplainer")` and `evaluate(explainer_views=...)` read:
  data/retrieval/<ds>/explainers/<method>/bank<b>.parquet   (bank, alert_tx, tx_id, score, seconds, sub_seconds, p_hist, p0_raw)
  models/pgexplainer/<ds>_bank<b>.pt                        (PGExplainer weights per bank, trained on TRAIN alerts only)
  data/runs/explainers_<ds>.json                            (seconds per alert, PGExplainer training, p0 reproduction check)

  python scripts/run_explainers.py --data hi_small                      # both methods, val cal+eval, CPU
  python scripts/run_explainers.py --data hi_medium --banks 3 4 --methods gnnexplainer

`--final` (week 14, needs FTM_FINAL_RUN=1): the same for the TEST alerts (`data/alerts/<ds>/test.parquet`, written by `make_alerts.py --final`) with the
model checkpoints and the per-bank PGExplainer weights of the val run (nothing is trained or tuned; PGExplainer is trained only if a bank has no
weights, on train alerts), the history up to each alert's time taken from train, val and test edges; the cache is `explainers_final/` and the
reference raw score is the test score of `scripts/train_gnn.py --final`.

Resumable: a bank whose shard exists is skipped. Val/train alerts only (`data/alerts/<ds>/{train,val}.parquet`), history up to the alert time only.
`p_hist` is the model's P(laundering) for the alerted edge on its history subgraph and `p0_raw` the stored raw score of the p0 run
(`data/scores/<ds>/p0_multi_gnn__one_bank.parquet`): the check that the explained prediction is the p0 prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.data import io  # noqa: E402
from ftm.retrieval import evidence_views as ev  # noqa: E402
from ftm.retrieval import explainers as ex  # noqa: E402


def log(msg) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--methods", nargs="+", choices=ex.EXPLAINERS, default=list(ex.EXPLAINERS))
    ap.add_argument("--parts", nargs="+", default=["cal", "eval"])
    ap.add_argument("--banks", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pg-alerts", type=int, default=ex.PG_TRAIN_ALERTS, help="train alerts per bank for PGExplainer")
    ap.add_argument("--pg-epochs", type=int, default=ex.PG_EPOCHS)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="alerts per bank (smoke test)")
    ap.add_argument("--final", action="store_true", help="week 14: explain the TEST alerts (needs FTM_FINAL_RUN=1)")
    a = ap.parse_args()
    if a.final:
        from ftm import guards

        guards.require_final_run("the explainers on test alerts")
    import torch

    torch.set_num_threads(a.threads)
    cfg = io.load_config(a.data)
    if a.final:
        alerts = pl.read_parquet(Path(cfg["alerts_dir"]) / "test.parquet")  # time-guard: not transactions
    else:
        alerts = pl.read_parquet(Path(cfg["alerts_dir"]) / "val.parquet").filter(pl.col("val_part").is_in(a.parts))  # time-guard: not transactions
    train = pl.read_parquet(Path(cfg["alerts_dir"]) / "train.parquet")  # time-guard: not transactions
    banks = io.read_banks(cfg["views_dir"]).sort("rank", descending=True)["bank"].to_list()  # the smallest views first, the hub last
    banks = [b for b in banks if b in set(alerts["bank"].to_list()) and (a.banks is None or b in a.banks)]
    out_dir = Path("data/retrieval") / a.data / ("explainers_final" if a.final else "explainers")
    stats_f = Path(f"data/runs/explainers_{a.data}{'_final' if a.final else ''}.json")
    p0_file = Path("data/scores") / a.data / ("final/p0_multi_gnn__one_bank.parquet" if a.final else "p0_multi_gnn__one_bank.parquet")
    stats = json.loads(stats_f.read_text()) if stats_f.exists() else {"dataset": a.data, "banks": {}}
    log(f"{a.data}: {alerts.height} alerts of {len(banks)} banks, methods {a.methods}")
    for bank in banks:
        todo = [m for m in a.methods if not (out_dir / m / f"bank{bank}.parquet").exists()]
        if not todo:
            continue
        t_bank = time.time()
        bm = ex.BankModel(a.data, bank, a.device, final=a.final)
        info = stats["banks"].setdefault(str(bank), {})
        if "pgexplainer" in todo:
            pg_f = ex.pg_path(a.data, bank)
            if not pg_f.exists():
                tr = train.filter(pl.col("bank") == bank).sample(fraction=1.0, shuffle=True, seed=0).head(a.pg_alerts).sort("timestamp")
                subs = [bm.history_subgraph(r["tx_id"], r["timestamp"]) for r in tr.iter_rows(named=True)]
                t0 = time.time()
                hist = bm.pg_train(subs, a.pg_epochs, seed=0) if subs else []
                if subs:
                    bm.pg_save(pg_f, a.pg_epochs)
                info["pg_train"] = {"alerts": len(subs), "epochs": a.pg_epochs, "seconds": time.time() - t0, "loss_first_last": [hist[0], hist[-1]] if hist else None}
                log(f"bank {bank}: PGExplainer trained on {len(subs)} train alerts, {time.time() - t0:.0f}s")
            if pg_f.exists():
                bm.pg_load(pg_f)
            else:  # no train alert at this bank: nothing to train on, the method is skipped there
                todo.remove("pgexplainer")
        al = alerts.filter(pl.col("bank") == bank)
        al = al.head(a.limit) if a.limit else al
        raw = (pl.scan_parquet(p0_file).filter((pl.col("bank") == bank) & pl.col("tx_id").is_in(al["tx_id"].to_list()))  # time-guard: not transactions
               .select("tx_id", "score_raw").collect())  # fmt: skip
        raw0 = dict(zip(raw["tx_id"].to_list(), raw["score_raw"].to_list(), strict=True))
        rows: dict[str, list] = {m: [] for m in todo}
        for r in al.iter_rows(named=True):
            cand = ev.candidates(cfg, bank, r["timestamp"], final=a.final)
            ctx = cand["tx_id"].to_numpy()
            t0 = time.time()
            sub = bm.history_subgraph(r["tx_id"], r["timestamp"])
            sub_s = time.time() - t0
            p_hist = bm.p_launder(sub)
            for m in todo:
                sc, sec = bm.scores(m, ctx, r["tx_id"], r["timestamp"], seed=r["tx_id"] % 2**31, sub=sub)
                keep = (sc > 0) | (ctx == r["tx_id"])  # the alerted edge always has a row: a row means "explained"
                rows[m].append(pl.DataFrame({"bank": bank, "alert_tx": r["tx_id"], "tx_id": ctx[keep], "score": sc[keep], "seconds": sec,
                                             "sub_seconds": sub_s, "p_hist": p_hist, "p0_raw": raw0.get(r["tx_id"])}))  # fmt: skip
        for m in todo:
            (out_dir / m).mkdir(parents=True, exist_ok=True)
            pl.concat(rows[m]).write_parquet(out_dir / m / f"bank{bank}.parquet")
        info["seconds"] = time.time() - t_bank
        info["alerts"] = al.height
        stats_f.write_text(json.dumps(stats, indent=1))
        log(f"bank {bank}: {al.height} alerts, {time.time() - t_bank:.0f}s")
        del bm
    # summary: seconds per alert and the p0 reproduction check
    summ = {}
    for m in a.methods:
        fs = sorted((out_dir / m).glob("bank*.parquet"))
        if not fs:
            continue
        d = pl.concat([pl.read_parquet(f) for f in fs]).group_by(["bank", "alert_tx"]).agg(pl.col("seconds").first(), pl.col("sub_seconds").first(), pl.col("p_hist").first(), pl.col("p0_raw").first(), pl.len().alias("edges"))  # time-guard: not transactions
        summ[m] = {"alerts": d.height, "explainer_seconds_mean": float(d["seconds"].mean()), "explainer_seconds_median": float(d["seconds"].median()),
                   "subgraph_seconds_mean": float(d["sub_seconds"].mean()), "mean_scored_edges": float(d["edges"].mean())}  # fmt: skip
        ok = d.drop_nulls(["p_hist", "p0_raw"])
        diff = (ok["p_hist"] - ok["p0_raw"]).abs().to_numpy()
        from scipy.stats import spearmanr

        summ[m]["p0_check"] = {"alerts": ok.height, "share_abs_diff_below_1e-4": float((diff < 1e-4).mean()), "median_abs_diff": float(np.median(diff)),
                               "spearman": float(spearmanr(ok["p_hist"].to_numpy(), ok["p0_raw"].to_numpy())[0])}  # fmt: skip
    stats["summary"] = summ
    stats_f.write_text(json.dumps(stats, indent=1))
    log(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
