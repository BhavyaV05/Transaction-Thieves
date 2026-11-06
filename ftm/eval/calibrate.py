"""Frozen calibration maps for the LLM systems' probabilities (audit finding 12).

Owner: T4.  A system's raw end-to-end probability is fitted on the validation `fit` part, checked on `cal`, frozen as a JSON with
its own hash, and applied to eval / test run logs without any label. Methods:

  platt    p' = sigmoid(a * logit(p) + b), fitted by Newton steps with a ridge toward a=1, b=0 (few alerts)
  levels   systems that emit a handful of probabilities (FinFRE-RAG's 5 score levels, Sharma's 0.9 / 0.1 verdicts): the launder
           rate per level, shrunk toward the level itself (`fit_score_map` rule); `sharma_verdict` calls `VerdictMap.fit`
  identity the fit did not help on `cal` (the map is written but marked rejected and left as the identity)

A row without a verdict (probability null) is scored as p0 and p0 is already calibrated, so it is never mapped.
Fit uses only run-log rows that carry a label and a `val_part` (validation runs); an eval or test log is only ever *applied* to.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

MAP_VERSION = 1
EPS = 1e-6
LEVEL_DECIMALS = 4


def _logit(p):
    p = np.clip(np.asarray(p, dtype="f8"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="f8")))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, dtype="f8") - np.asarray(y, dtype="f8")) ** 2))


def fit_platt(p, y, ridge: float = 1.0, iters: int = 50) -> dict:
    """Logistic regression of y on logit(p) with a ridge pulling (a, b) toward (1, 0): few alerts must not flip the map."""
    x, y = _logit(p), np.asarray(y, dtype="f8")
    w = np.array([1.0, 0.0])
    X = np.column_stack([x, np.ones_like(x)])
    for _ in range(iters):
        q = _sigmoid(X @ w)
        g = X.T @ (q - y) + ridge * (w - np.array([1.0, 0.0]))
        H = (X * (q * (1 - q))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-9:
            break
    return {"a": float(w[0]), "b": float(w[1])}


def fit_levels(p, y, strength: float = 4.0) -> dict:
    """{level (str): rate}: (sum y + strength * level) / (n + strength) per distinct rounded probability."""
    p, y = np.round(np.asarray(p, dtype="f8"), LEVEL_DECIMALS), np.asarray(y, dtype="f8")
    out = {}
    for lv in sorted(set(p.tolist())):
        m = p == lv
        out[f"{lv:.{LEVEL_DECIMALS}f}"] = float((y[m].sum() + strength * lv) / (m.sum() + strength))
    return {"levels": out, "strength": strength}


def fit_sharma_verdict(p, y, fraud: float = 0.9, not_fraud: float = 0.1) -> dict:
    """Sharma: a probability of 0.9 / 0.1 is a fraud / not_fraud verdict; `VerdictMap.fit` gives the per-class launder rate.
    Other probabilities (escalate = p0) are left alone."""
    from ftm.external.sharma import VerdictMap

    p = np.asarray(p, dtype="f8")
    verdicts = np.where(np.isclose(p, fraud), "fraud", np.where(np.isclose(p, not_fraud), "not_fraud", "escalate"))
    m = VerdictMap.fit(list(verdicts), [int(v) for v in y])
    return {"fraud_in": fraud, "not_fraud_in": not_fraud, "fraud": m.fraud, "not_fraud": m.not_fraud}


def apply_params(method: str, params: dict, p):
    p = np.asarray(p, dtype="f8")
    if method == "identity":
        return p
    if method == "platt":
        return _sigmoid(params["a"] * _logit(p) + params["b"])
    if method == "levels":
        table = {float(k): v for k, v in params["levels"].items()}
        keys = np.array(sorted(table))
        out = p.copy()
        for i, v in enumerate(p):
            j = int(np.argmin(np.abs(keys - v)))
            if abs(keys[j] - v) < 10 ** -LEVEL_DECIMALS:
                out[i] = table[float(keys[j])]  # a probability off every level (not a score level) stays as it is
        return out
    if method == "sharma_verdict":
        out = p.copy()
        out[np.isclose(p, params["fraud_in"])] = params["fraud"]
        out[np.isclose(p, params["not_fraud_in"])] = params["not_fraud"]
        return out
    raise ValueError(f"unknown calibration method {method!r}")


def _canonical(d: dict) -> bytes:
    return json.dumps({k: v for k, v in d.items() if k != "sha256"}, sort_keys=True, default=str).encode()


def map_hash(d: dict) -> str:
    return hashlib.sha256(_canonical(d)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- reading run logs ------------------------------------------------------------------------------------------
def labelled_verdicts(run_dirs: list[Path]) -> pl.DataFrame:
    """Verdict rows (probability, p0, label, val_part, system, variant, seed, alert_id) of validation run logs. Refuses a log
    without labels or without val_part: calibration is fitted on validation runs only."""
    frames = []
    for d in run_dirs:
        f = Path(d) / "runlog.parquet" if Path(d).is_dir() else Path(d)
        df = pl.read_parquet(f)  # time-guard: not transactions
        v = df.filter(pl.col("event") == "verdict")
        if "label" not in v.columns or v["label"].null_count() == v.height:
            raise ValueError(f"{f}: no labels in the run log (a blind or test run cannot be fitted on)")
        if "val_part" not in v.columns or v["val_part"].null_count() == v.height:
            raise ValueError(f"{f}: no val_part, only validation runs are fitted on")
        if (v["split"] != "val").any():
            raise ValueError(f"{f}: rows of split {sorted(set(v['split']) - {'val'})}; calibration is fitted on validation runs only")
        frames.append(v.select("system", "variant", "seed", "alert_id", "probability", "p0", "label", "val_part").with_columns(
            pl.lit(str(f)).alias("_src")))
    return pl.concat(frames, how="vertical_relaxed")


def _arrays(df: pl.DataFrame):
    d = df.filter(pl.col("probability").is_not_null())
    return d["probability"].to_numpy(), d["label"].cast(pl.Float64).to_numpy(), d


def end_to_end_brier(df: pl.DataFrame, method: str | None = None, params: dict | None = None) -> float:
    """Brier on all rows of `df` (no verdict = p0), the probability mapped by (method, params) if given."""
    p = df["probability"].to_numpy().astype("f8")
    miss = np.isnan(p)
    q = p.copy()
    if method:
        q[~miss] = apply_params(method, params or {}, p[~miss])
    q = np.where(miss, df["p0"].to_numpy(), q)
    return brier(q, df["label"].cast(pl.Float64).to_numpy())


def fit_map(df: pl.DataFrame, system: str, method: str = "auto", fit_parts=("fit",), check_parts=("cal",), strength: float = 4.0,
            min_rows: int = 30) -> dict:  # fmt: skip
    """Fit on the `fit_parts` rows of one system (all its variants and seeds in `df`), check on `check_parts`. Returns the map dict
    (not yet hashed). `auto`: levels for <= 8 distinct probabilities (sharma: sharma_verdict), else platt."""
    d = df.filter(pl.col("system") == system)
    fit, chk = d.filter(pl.col("val_part").is_in(list(fit_parts))), d.filter(pl.col("val_part").is_in(list(check_parts)))
    pf, yf, _ = _arrays(fit)
    if len(pf) < min_rows or len(set(yf.tolist())) < 2:
        raise ValueError(f"{system}: {len(pf)} fit rows with a verdict and {len(set(yf.tolist()))} class(es): too few to calibrate")
    if method == "auto":
        method = "sharma_verdict" if system.startswith("sharma") else ("levels" if len(np.unique(np.round(pf, LEVEL_DECIMALS))) <= 8 else "platt")
    params = {"platt": lambda: fit_platt(pf, yf), "levels": lambda: fit_levels(pf, yf, strength),
              "sharma_verdict": lambda: fit_sharma_verdict(pf, yf)}[method]()
    out = {"version": MAP_VERSION, "system": system, "method": method, "params": params, "fit_parts": list(fit_parts),
           "check_parts": list(check_parts), "n_fit": int(len(pf)), "n_fit_alerts_total": int(fit.height),
           "fit_label_rate": float(yf.mean()), "fit_brier_raw": end_to_end_brier(fit), "fit_brier_mapped": end_to_end_brier(fit, method, params),
           "variants": sorted(set(d["variant"].to_list())), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}  # fmt: skip
    if chk.height:
        raw, mapped = end_to_end_brier(chk), end_to_end_brier(chk, method, params)
        out["check"] = {"n": chk.height, "brier_raw": raw, "brier_mapped": mapped, "improves": bool(mapped < raw)}
        if not mapped < raw:  # the check set is the only guard against an over-fitted map
            out["rejected"] = f"brier on {list(check_parts)} {mapped:.5f} is not below raw {raw:.5f}: the map is the identity"
            out["method_fitted"], out["params_fitted"] = method, params
            out["method"], out["params"] = "identity", {}
    else:
        out["check"] = None
        out["rejected"] = None
    return out


def freeze_map(m: dict, path: Path) -> dict:
    m = {**m, "sha256": map_hash(m)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=2, default=str) + "\n", encoding="utf-8")
    return m


def load_map(path: Path, expect_sha: str | None = None) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("sha256") != map_hash(m):
        raise ValueError(f"{path}: the calibration map does not match its own hash (edited after freezing)")
    if expect_sha and m["sha256"] != expect_sha:
        raise ValueError(f"{path}: hash {m['sha256'][:12]} is not the expected {expect_sha[:12]}")
    if m.get("version") != MAP_VERSION:
        raise ValueError(f"{path}: map version {m.get('version')}")
    return m


def apply_map_to_frame(df: pl.DataFrame, m: dict) -> pl.DataFrame:
    """Map the `probability` of the verdict rows of system `m['system']` (no label is read). The raw value is kept in
    `probability_raw`; rows without a verdict stay null (they are scored as p0)."""
    if "probability_raw" in df.columns:
        raise ValueError("the run log is already calibrated (has probability_raw)")
    is_v = (df["event"] == "verdict").to_numpy() & (df["system"] == m["system"]).to_numpy() & df["probability"].is_not_null().to_numpy()
    raw = df["probability"].to_numpy().astype("f8")
    new = raw.copy()
    if is_v.any():
        new[is_v] = apply_params(m["method"], m["params"], raw[is_v])
    return df.with_columns(pl.Series("probability_raw", raw), pl.Series("probability", new)).with_columns(
        pl.when(pl.col("probability_raw").is_nan()).then(None).otherwise(pl.col("probability")).alias("probability"),
        pl.when(pl.col("probability_raw").is_nan()).then(None).otherwise(pl.col("probability_raw")).alias("probability_raw"))


def describe(m: dict) -> str:
    c = m.get("check")
    s = f"{m['system']}: {m['method']} {json.dumps(m['params'])[:120]} fit n={m['n_fit']}"
    if c:
        s += f"; cal brier {c['brier_raw']:.5f} -> {c['brier_mapped']:.5f}" + (" (REJECTED, identity)" if m.get("rejected") else "")
    return s
