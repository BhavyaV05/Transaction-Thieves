"""Score run logs (ftm.eval.runlog) per system: the numbers of Table 5a/5b and the eval dashboard.

Owner: T4 (Shreyas).  Spec: plan.md 9, tracks.md 6.7.

For every (system, variant) in a run log:
  end_to_end      the system's own probability; a missing verdict is p0 (`metrics.end_to_end_probability`)
  questions_only  p0 plus the run's answers through fusion: summed log_lr per family -> `Fusion.predict`
                  (only answers with an accepted back-test LR count; the system's own words never do)
  baseline        p0 alone
and, with cluster-bootstrap CIs, Brier, gap closed against each upper limit, the no-verdict rate and the cost
(LLM calls, tokens, GPU-seconds, eps spent). Labels come from the caller (eval code), never from the log.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ftm.eval import bootstrap as bs
from ftm.eval import metrics as mt
from ftm.fusion import fuse

GROUP = ("system", "variant")


def verdict_table(log: pl.DataFrame) -> pl.DataFrame:
    """One row per (system, variant, seed, alert): probability, p0 and the cost fields of the verdict row."""
    v = log.filter(pl.col("event") == "verdict")
    keep = [c for c in ("system", "variant", "seed", "alert_id", "probability", "p0", "llm_calls", "tokens_in",
                        "tokens_out", "gpu_seconds", "eps_spent", "stop_reason") if c in v.columns]  # fmt: skip
    v = v.select(keep)
    if v.select(pl.struct(["system", "variant", "seed", "alert_id"]).is_duplicated().any()).item():
        raise ValueError("more than one verdict for the same (system, variant, seed, alert)")
    return v


def answer_features(log: pl.DataFrame, alert_ids: list[str], families: list[str] | None = None):
    """Summed log-LR per (alert, family) from the step rows of ONE system/variant/seed (null log_lr = unused)."""
    s = log.filter((pl.col("event") == "step") & pl.col("log_lr").is_not_null() & pl.col("family").is_not_null()
                   & pl.col("alert_id").is_in(alert_ids))
    return fuse.answer_features(alert_ids, s["alert_id"].to_list(), s["family"].to_list(), s["log_lr"].to_numpy(),
                                families=families)  # fmt: skip


def score(log: pl.DataFrame, labels: dict, clusters: dict | None = None, upper: dict | None = None,
          fusion: fuse.Fusion | None = None, n_boot: int = bs.N_BOOT, seed: int = bs.SEED) -> list[dict]:  # fmt: skip
    """Per (system, variant, seed) summary. `labels`/`clusters`/`upper[name]`: alert_id -> value.

    `upper` maps an upper-limit name (e.g. "gfp", "multi_gnn") to {alert_id: probability}. Alerts without a
    label are dropped and counted.
    """
    fusion = fusion or fuse.Fusion("naive")
    out = []
    vt = verdict_table(log)
    for key, grp in vt.group_by(["system", "variant", "seed"], maintain_order=True):
        grp = grp.filter(pl.col("alert_id").is_in(list(labels)))
        ids = grp["alert_id"].to_list()
        if not ids:
            continue
        y = np.array([labels[a] for a in ids], dtype="f8")
        p0 = grp["p0"].to_numpy().astype("f8") if "p0" in grp.columns else None
        if p0 is None or np.isnan(p0).any():
            raise ValueError(f"{key}: every verdict row needs p0 to score end-to-end and questions-only")
        e2e, no_verdict = mt.end_to_end_probability(grp["probability"].to_list(), p0)
        sub = log.filter((pl.col("system") == key[0]) & (pl.col("variant") == key[1]) & (pl.col("seed") == key[2]))
        X, _ = answer_features(sub, ids, families=fusion.families or None)
        qo = fusion.predict(p0, X) if X.shape[1] else p0.copy()
        cl = np.array([clusters[a] for a in ids]) if clusters else np.arange(len(ids))
        row = {"system": key[0], "variant": key[1], "seed": key[2], "n": len(ids), "n_pos": int(y.sum()),
               "dropped_unlabelled": int(vt.filter((pl.col("system") == key[0]) & (pl.col("variant") == key[1])
                                                   & (pl.col("seed") == key[2])).height - len(ids)),
               "no_verdict_rate": no_verdict, "brier_p0": mt.brier(y, p0), "brier_end_to_end": mt.brier(y, e2e),
               "brier_questions_only": mt.brier(y, qo), "pr_auc_end_to_end": mt.pr_auc(y, e2e),
               "clusters": bs.cluster_report(cl)}  # fmt: skip
        d = bs.brier_diff(y, e2e, p0, cl, n_boot=n_boot, seed=seed)
        row["brier_e2e_minus_p0"] = {"diff": d["diff"], "ci95": d["ci"]}
        for name, up in (upper or {}).items():
            pu = np.array([up[a] for a in ids], dtype="f8")
            for which, p in (("end_to_end", e2e), ("questions_only", qo)):
                g = bs.gap_closed_ci(y, p0, p, pu, cl, n_boot=n_boot, seed=seed)
                row[f"gap_closed_{which}_vs_{name}"] = {"point": g["gap_closed"], "ci95": g["ci"],
                                                        "nan_share": g["nan_share"]}  # fmt: skip
        for c in ("llm_calls", "tokens_in", "tokens_out", "gpu_seconds", "eps_spent"):
            if c in grp.columns:
                vals = grp[c].drop_nulls()
                row[f"mean_{c}"] = float(vals.mean()) if len(vals) else None
        if "stop_reason" in grp.columns:
            row["stop_reasons"] = {k: int(n) for k, n in grp["stop_reason"].value_counts().iter_rows()}
        out.append(row)
    return out
