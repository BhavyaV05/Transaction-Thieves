"""build: run the whole data pipeline for one data config and write the manifest.

Owner: T1 (Abhishek).  Spec: plan.md 5.2, tracks.md 3.3.  CLI: scripts/prepare_data.py.

Idempotent: if manifest.json already holds the same build key (raw sha256, split rule, n_banks,
PIPELINE_VERSION) and the outputs exist, nothing is rebuilt unless force=True.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from ftm.data import alias, convert, patterns, views
from ftm.data.splits import validate_split_days

PIPELINE_VERSION = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def raw_files(cfg: dict) -> dict[str, Path]:
    raw, name = Path(cfg["raw_dir"]), cfg["name"]
    return {k: raw / f"{name}_{k}" for k in ("Trans.csv", "accounts.csv", "Patterns.txt")}


def _build_key(cfg: dict, shas: dict) -> str:
    blob = json.dumps(
        [PIPELINE_VERSION, shas, cfg["split_days"], cfg["n_banks"], cfg.get("alias_seeds", [0])],
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def build_dataset(cfg: dict, *, root: str | Path = ".", force: bool = False, log=print) -> dict:
    root = Path(root)
    cfg = {**cfg, **{k: str(root / cfg[k]) for k in ("raw_dir", "parquet_dir", "views_dir")}}
    split_days = validate_split_days(cfg["split_days"])
    out, files = Path(cfg["parquet_dir"]), raw_files(cfg)
    log("hashing raw files")
    shas = {k: sha256(p) for k, p in files.items()}
    key = _build_key(cfg, shas)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        if old.get("build_key") == key:
            log("up to date, nothing to do (use force=True / --force to rebuild)")
            return old

    log("accounts + transactions")
    raw = convert.scan_raw_trans(files["Trans.csv"])
    accounts = convert.build_accounts(files["accounts.csv"], raw)
    out.mkdir(parents=True, exist_ok=True)
    accounts.write_parquet(out / "accounts.parquet")
    tx_path = out / "transactions.parquet"
    convert.build_transactions(files["Trans.csv"], accounts, split_days, tx_path)
    trans = pl.scan_parquet(tx_path)

    check = trans.select(
        pl.len().alias("n_tx"),
        pl.col("is_laundering").sum().alias("n_laundering"),
        pl.col("split").null_count().alias("n_null_split"),
        pl.col("src_id").null_count().add(pl.col("dst_id").null_count()).alias("n_null_node"),
        pl.col("timestamp").min().alias("t_min"),
        pl.col("timestamp").max().alias("t_max"),
        pl.col("day").max().alias("n_days"),
    ).collect(engine="streaming").row(0, named=True)
    if check["n_null_split"]:
        raise ValueError(f"{check['n_null_split']} rows fall outside split_days {split_days}")
    if check["n_null_node"]:
        raise ValueError("transactions with unknown account node")

    log("patterns")
    pat = patterns.parse_patterns(files["Patterns.txt"])
    matched, pstats = patterns.match_patterns(pat, trans)
    split_of = trans.select("tx_id", "split").join(matched.lazy().select("tx_id"), on="tx_id").collect()
    (
        matched.join(split_of, on="tx_id").select("tx_id", "attempt_id", "typology", "split")
        .sort("tx_id").write_parquet(out / "patterns.parquet")
    )  # fmt: skip
    attempts = patterns.build_attempts(matched, pat, trans)
    attempts.write_parquet(out / "attempts.parquet")
    n_not_in_attempt = int(
        trans.filter(pl.col("is_laundering")).join(matched.lazy(), on="tx_id", how="anti")
        .select(pl.len()).collect().item()
    )  # fmt: skip

    log("bank views")
    banks = views.bank_table(trans, cfg["n_banks"])
    views.write_views(trans, banks, cfg["views_dir"])

    log("aliases")
    for seed in cfg.get("alias_seeds", [0]):
        alias.write_aliases(out, seed)

    per_split = (
        trans.group_by("split").agg(pl.len().alias("n_tx"), pl.col("is_laundering").sum().alias("n_laundering"))
        .sort("split").collect()
    )  # fmt: skip
    per_day = (
        trans.group_by("day").agg(pl.len().alias("n_tx"), pl.col("is_laundering").sum().alias("n_laundering"))
        .sort("day").collect()
    )  # fmt: skip
    totals = attempts.group_by("typology").agg(
        pl.len().alias("n_attempts"), pl.col("n_lines").sum(), pl.col("n_tx").sum(),
        pl.col("spans_split").sum().alias("n_spanning_splits"),
    ).sort("typology")  # fmt: skip
    top = banks["bank"].to_list()
    cov = trans.filter(pl.col("from_bank").is_in(top) | pl.col("to_bank").is_in(top)).select(
        pl.len().alias("n"), pl.col("is_laundering").sum().alias("l")
    ).collect(engine="streaming").row(0, named=True)
    manifest = {
        "build_key": key,
        "pipeline_version": PIPELINE_VERSION,
        "dataset": cfg["name"],
        "raw_sha256": {p.name: s for (p, s) in zip(files.values(), shas.values(), strict=True)},
        "n_tx": check["n_tx"],
        "n_laundering": check["n_laundering"],
        "n_accounts": accounts.height,
        "n_accounts_not_in_accounts_csv": int(accounts["entity_id"].null_count()),
        "timestamp_min": str(check["t_min"]),
        "timestamp_max": str(check["t_max"]),
        "n_days": check["n_days"],
        "split_days": split_days,
        "per_split": {r["split"]: {"n_tx": r["n_tx"], "n_laundering": r["n_laundering"]} for r in per_split.to_dicts()},
        "per_day": {str(r["day"]): {"n_tx": r["n_tx"], "n_laundering": r["n_laundering"]} for r in per_day.to_dicts()},
        "patterns": {
            **pstats,
            "n_laundering_tx_not_in_any_attempt": n_not_in_attempt,
            "n_attempts_spanning_splits": int(attempts["spans_split"].sum()),
            "by_typology": {r["typology"]: {k: v for k, v in r.items() if k != "typology"} for r in totals.to_dicts()},
        },
        "banks": {
            "n_views": banks.height,
            "n_tx_in_any_view": cov["n"],
            "tx_coverage": cov["n"] / check["n_tx"],
            "n_laundering_in_any_view": cov["l"],
            "laundering_coverage": cov["l"] / max(check["n_laundering"], 1),
            "top": banks.to_dicts(),
        },
        "alias_seeds": cfg.get("alias_seeds", [0]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    log(f"done: {check['n_tx']} tx, {check['n_laundering']} laundering")
    return manifest
