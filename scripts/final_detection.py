"""Final-run run-log writer for T1's detection systems (Groups A and B): `external_cmd.templates.detection` in `configs/final/jobs.yaml`.

Owner: T1 (Abhishek).  Run by `scripts/run_final.py` (FTM_FINAL_RUN=1 set there), once per detection job `{system}.{variant}.{alerts}`:

  python scripts/final_detection.py --system multi_gnn --variant one_bank --data hi_medium --seed 0 --alert-ids <ids file> --out <job dir>

Reads T1's TEST scores (`data/scores/<ds>/final/`, written by the `--final` modes of the t1 steps) for the alerts listed in `--alert-ids` (one alert id per
line; the alert set is `data/alerts/<ds>/test.parquet`) and writes `<out>/runlog.parquet`, run-log v1 (`ftm/eval/runlog.py`): one `verdict` row per
alert with `probability` = the system's calibrated p0 for that alert (null where the model has no score for it: scored as the alert's p0 downstream),
`p0` = the alert's one-bank score. **No label column**: the analysis joins labels in. `require_final_run` first.

Score files (one per system and variant; an unknown or unavailable one exits non-zero with the reason, the job fails):
  multi_gnn, mega_gnn  one_bank | see_everything   final/p0_<system>__<variant>.parquet   (one_bank: matched on tx_id and home bank)
  fraudgt              see_everything (HI-Small)   final/p0_fraudgt.parquet
  dp_fedavg_gnn        30_clients                  final/dpfedavg_client_eps<--eps>_seed<seed>.parquet   (client-level DP-FedAvg, default eps 8)
  tian_2025            as_published                final/sync_sgd_bwd_seed<seed>.parquet   (layer-wise exchange, Sync-SGD fwd/bwd: the fallback for Tian et al., tracks.md 3.7)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm import guards  # noqa: E402
from ftm.eval import runlog  # noqa: E402

# (system, variant) -> (file name template, kind: "view" = (tx_id, bank) rows, "tx" = one row per transaction, model, datasets)
SOURCES: dict[tuple[str, str], tuple[str, str, str, tuple[str, ...]]] = {
    ("multi_gnn", "one_bank"): ("p0_multi_gnn__one_bank.parquet", "view", "multi_gnn", ("hi_small", "hi_medium")),
    ("multi_gnn", "see_everything"): ("p0_multi_gnn__see_everything.parquet", "tx", "multi_gnn", ("hi_small", "hi_medium")),
    ("mega_gnn", "one_bank"): ("p0_mega_gnn__one_bank.parquet", "view", "mega_gnn", ("hi_small", "hi_medium")),
    ("mega_gnn", "see_everything"): ("p0_mega_gnn__see_everything.parquet", "tx", "mega_gnn", ("hi_small", "hi_medium")),
    ("fraudgt", "see_everything"): ("p0_fraudgt.parquet", "tx", "fraudgt", ("hi_small",)),
    ("dp_fedavg_gnn", "30_clients"): ("dpfedavg_client_eps{eps}_seed{seed}.parquet", "view", None, ("hi_small", "hi_medium")),
    ("tian_2025", "as_published"): ("sync_sgd_bwd_seed{seed}.parquet", "view", None, ("hi_small", "hi_medium")),
}
NOT_AVAILABLE = {
    ("fraudgt", "one_bank"): "FraudGT is a full-graph (see-everything) model, HI-Small only: there is no one-bank FraudGT",
    ("exstraqt", "see_everything"): "ExSTraQt is a stretch system that was not implemented",
}


def source(system: str, variant: str, data: str, seed: int, eps: str = "8") -> tuple[str, str, str | None]:
    """(file name, kind, model filter) of the score file of a system and variant, or SystemExit with the reason."""
    key = (system, variant)
    if key in NOT_AVAILABLE:
        raise SystemExit(f"no T1 scores for {system}.{variant}: {NOT_AVAILABLE[key]}")
    if key not in SOURCES:
        raise SystemExit(f"unknown detection system/variant {system}.{variant}; known: {sorted('.'.join(k) for k in SOURCES)}")
    name, kind, model, datasets = SOURCES[key]
    if data not in datasets:
        raise SystemExit(f"no T1 scores for {system}.{variant} on {data}: only on {list(datasets)}")
    return name.format(seed=seed, eps=eps), kind, model


def alert_rows(path: Path, ids: list[str]) -> pl.DataFrame:
    """alert_id, tx_id, bank, p0 of the listed alerts. The label column of the file is never read; an id that is not in the file is an error."""
    a = pl.read_parquet(path, columns=["alert_id", "tx_id", "bank", "p0"]).with_columns(pl.col("alert_id").cast(pl.Utf8))  # time-guard: not transactions; no label column
    a = a.filter(pl.col("alert_id").is_in(ids))
    missing = set(ids) - set(a["alert_id"].to_list())
    if missing:
        raise SystemExit(f"{len(missing)} alert ids are not in {path} (first: {sorted(missing)[0]})")
    return a


def build_runlog(alerts: pl.DataFrame, scores: pl.DataFrame, kind: str, *, system: str, variant: str, seed: int, run_id: str, commit: str, machine: str) -> pl.DataFrame:
    """The run log: one verdict row per alert, probability from `scores` (tx_id [, bank], p0), no label column."""
    keys = ["tx_id", "bank"] if kind == "view" else ["tx_id"]
    s = scores.select(*keys, pl.col("p0").alias("probability")).with_columns(pl.col("tx_id").cast(pl.Int64), *([pl.col("bank").cast(pl.Int64)] if kind == "view" else []))
    if s.select(pl.struct(keys).is_duplicated().any()).item():
        raise SystemExit("the score file has more than one row per transaction" + (" and bank" if kind == "view" else ""))
    a = alerts.with_columns(pl.col("tx_id").cast(pl.Int64), pl.col("bank").cast(pl.Int64))
    j = a.join(s, on=keys, how="left").sort("alert_id")
    if j["probability"].null_count() == j.height:
        raise SystemExit("none of the alerts has a score in the score file: wrong dataset, split or file")
    n = j.height
    return pl.DataFrame({
        "run_id": [run_id] * n, "commit": [commit] * n, "machine": [machine] * n, "system": [system] * n, "variant": [variant] * n, "seed": [int(seed)] * n,
        "split": ["test"] * n, "alert_id": j["alert_id"], "event": ["verdict"] * n, "probability": j["probability"].cast(pl.Float64), "p0": j["p0"].cast(pl.Float64),
        "home_bank": j["bank"].cast(pl.Utf8), "log_version": [runlog.RUN_LOG_VERSION] * n, "trials": [1] * n,
    })  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--data", default="hi_medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alert-ids", required=True, help="file with one alert id per line (run_final's {alert_ids})")
    ap.add_argument("--out", required=True, help="job directory; runlog.parquet is written into it")
    ap.add_argument("--eps", default="8", help="dp_fedavg_gnn: epsilon of the client-level arm that stands for the system")
    ap.add_argument("--run-id", default="final")
    ap.add_argument("--commit", default="")
    ap.add_argument("--machine", default="t1-3070ti")
    ap.add_argument("--root", default="data", help="the data directory (scores/<ds>/final, alerts/<ds>/test.parquet)")
    a = ap.parse_args()
    guards.require_final_run("the T1 test scores")
    name, kind, model = source(a.system, a.variant, a.data, a.seed, a.eps)
    root = Path(a.root)
    f = root / "scores" / a.data / "final" / name
    if not f.exists():
        raise SystemExit(f"{f} does not exist: the t1 step that writes it has not run (docs/t1_final_run.md)")
    scores = pl.read_parquet(f)  # time-guard: not transactions
    scores = scores.filter(pl.col("split") == "test") if "split" in scores.columns else scores
    if model:
        scores = scores.filter(pl.col("model") == model)
    ids = [x.strip() for x in Path(a.alert_ids).read_text().splitlines() if x.strip()]
    log = build_runlog(alert_rows(root / "alerts" / a.data / "test.parquet", ids), scores, kind, system=a.system, variant=a.variant, seed=a.seed, run_id=a.run_id, commit=a.commit, machine=a.machine)
    assert "label" not in log.columns and not runlog.validate(log.columns, log["event"].unique().to_list())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log.write_parquet(out / "runlog.parquet")
    print(f"{a.system}.{a.variant}: {log.height} verdict rows, {int(log['probability'].null_count())} without a score")


if __name__ == "__main__":
    main()
