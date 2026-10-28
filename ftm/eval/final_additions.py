"""Three additions to the final analysis (issues 85, 87, 88): the no-noise row, review-capacity metrics, p0-adjusted differences.

Owner: T4 (Shreyas).  Companion of ftm/eval/final.py and final_tables.py; called from `final_tables.run_all`.  The power addition (issue 86)
is `final.power_details`.

- eps = infinity: reads the JSON of scripts/eps_sweep.py (a validation artefact, e.g. data/runs/eps_sweep_hi_medium.json), lays out the Brier
  gain against the budget including the no-noise row, and writes a one-line neutral reading.  No run log and no label is read.
- review capacity (S1, secondary / sensitivity, not in a Holm family): precision and recall among the alerts a fixed review capacity reaches
  (the top `fraction` of alerts by score), per system, with paired cluster-bootstrap CIs of the difference to the one-bank score.  The
  capacity comes from fixed_on_val.json (`review_capacity`); a `--split val` run writes the placeholder default (top 10%) when it is absent.
- p0-adjusted estimate (S2, sensitivity, the unadjusted difference stays primary): the per-alert Brier difference of H1 and H3 regressed on
  p0 and the number of answered questions (OLS, cluster bootstrap refits the regression on every resample of whole clusters).

Nothing here uses a label other than through the Brier differences the primary analysis already uses (and, for S1, the same alert labels).
"""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl

from ftm.eval import bootstrap as bs
from ftm.eval import final as fn
from ftm.eval.final import Missing, Panel, Scores, _fci

NO_NOISE_MARGIN = 0.005  # half the G2 gain target of 0.01: the no-noise gain must beat the eps = 2 gain by this much to call noise a limiter
DEFAULT_CAPACITY = {"kind": "top_fraction", "fraction": 0.10, "status": "placeholder, to be pre-registered"}
SENSITIVITY_LABEL = "sensitivity, unadjusted is primary"


# --------------------------------------------------------------------------------------------------------------------
# eps = infinity (issue 85)
# --------------------------------------------------------------------------------------------------------------------
def _budget(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = {"inf": math.inf, "infinity": math.inf, "none": math.inf, "no_noise": math.inf}.get(v.strip().lower(), v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def eps_reading(rows: list[dict]) -> str:
    """One neutral line: compare the no-noise gain with the eps = 2 gain (Brier gain, positive = better than one bank)."""
    gains = {r["budget"]: r["gain"] for r in rows if r["gain"] is not None}
    g_inf, g_2 = gains.get(math.inf), gains.get(2.0)
    if g_inf is None:
        return "No no-noise (eps = infinity) row in the sweep: noise cannot be compared."
    if g_2 is None:
        return f"No eps = 2 row in the sweep: the no-noise gain is {g_inf:+.4f}, nothing to compare it with."
    d = g_inf - g_2
    head = f"Brier gain {g_inf:+.4f} with no noise against {g_2:+.4f} at eps 2 (difference {d:+.4f}; no CI: one sweep, single seed)"
    if g_inf <= 0:
        return f"Noise is not the limiter: {head}; with no noise the gain is still not positive."
    if d > NO_NOISE_MARGIN:
        return f"Noise is the limiter: {head}; the no-noise gain exceeds the eps 2 gain by more than {NO_NOISE_MARGIN}."
    return f"Noise is not the limiter: {head}; the no-noise gain is within {NO_NOISE_MARGIN} of the eps 2 gain."


def eps_infinity(inputs: dict, split: str, w) -> list[dict]:
    """Table `eps_infinity`: gain against the per-alert budget including the no-noise row, from the eps-sweep JSON."""
    spec = inputs.get("eps_sweep") or {}
    if not spec.get("path"):
        raise Missing("eps-sweep JSON not provided (inputs.yaml eps_sweep.path, e.g. data/runs/eps_sweep_hi_medium.json from scripts/eps_sweep.py)")
    p = fn.resolve_path(spec["path"], split)
    if p is None or not p.exists():
        raise Missing(f"eps-sweep JSON {spec['path']} not found")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise fn.AnalysisInputError(f"eps-sweep JSON {p} is not valid JSON: {e}") from e
    if not isinstance(raw, list) or not raw:
        raise fn.AnalysisInputError(f"eps-sweep JSON {p} is not a non-empty list of rows")
    rows = []
    for r in raw:
        b = _budget(r.get("budget"))
        if b is None:
            raise fn.AnalysisInputError(f"eps-sweep JSON {p}: a row has no usable 'budget'")
        gain = r.get("brier_gain")
        if gain is None and r.get("brier_p0") is not None and r.get("brier_final") is not None:
            gain = r["brier_p0"] - r["brier_final"]
        rows.append({"system": r.get("system"), "budget": b, "n": r.get("alerts"), "brier_p0": r.get("brier_p0"), "brier_final": r.get("brier_final"),
                     "gain": fn.fnum(gain), "mean_questions": r.get("mean_questions")})  # fmt: skip
    rows.sort(key=lambda r: (str(r["system"]), r["budget"]))
    out, readings = [], []
    for sysname in dict.fromkeys(r["system"] for r in rows):
        mine = [r for r in rows if r["system"] == sysname]
        g2 = next((r["gain"] for r in mine if r["budget"] == 2.0), None)
        for r in mine:
            out.append({"system": sysname, "eps": "infinity (no noise)" if math.isinf(r["budget"]) else r["budget"], "n alerts": r["n"],
                        "brier p0": r["brier_p0"], "brier final": r["brier_final"], "brier gain": r["gain"], "mean questions": r["mean_questions"],
                        "gain minus gain at eps 2": None if r["gain"] is None or g2 is None else r["gain"] - g2})  # fmt: skip
        readings.append((f"{sysname}: " if len({r["system"] for r in rows}) > 1 else "") + eps_reading(mine))
    reading = " ".join(readings)
    note = (f"Reading (machine-generated, descriptive): {reading} Source: {p.name}, a validation sweep (gain = Brier(p0) - Brier(final), positive is better), "
            f"not a result on this run's split.")  # fmt: skip
    cols = ["system", "eps", "n alerts", "brier p0", "brier final", "brier gain", "mean questions", "gain minus gain at eps 2"]
    w.table("eps_infinity", out, cols, "Brier gain against eps per alert, including the no-noise row (eps = infinity)", note=note)
    w.status["eps_infinity"]["reading"] = reading
    w.status["eps_infinity"]["source"] = str(spec["path"])
    return out


# --------------------------------------------------------------------------------------------------------------------
# review capacity (S1, issue 87)
# --------------------------------------------------------------------------------------------------------------------
def resolve_capacity(panel: Panel) -> tuple[dict, str]:
    """The review capacity: the value in fixed_on_val.json, else (validation only) the placeholder default, recorded for fixed_on_val.json."""
    cap = panel.fixed.get("review_capacity")
    if cap:
        frac = cap.get("fraction") if isinstance(cap, dict) else cap
        if not isinstance(frac, (int, float)) or not 0 < float(frac) <= 1:
            raise fn.AnalysisInputError(f"fixed_on_val.json review_capacity must be a fraction in (0, 1] of the alerts or {{fraction: ...}}, got {cap!r}")
        return (cap if isinstance(cap, dict) else {"kind": "top_fraction", "fraction": float(cap)}), "fixed on validation"
    if panel.split == "test":
        raise Missing("review_capacity is not fixed: run --split val first and pass its fixed_on_val.json (it carries review_capacity)")
    cap = dict(DEFAULT_CAPACITY)
    panel.fixed_derived["review_capacity"] = cap
    return cap, "default placeholder (top 10% of alerts by score), written to fixed_on_val.json"


def _topk_resamples(scores: dict[str, np.ndarray], y: np.ndarray, clusters: np.ndarray, frac: float, n_boot: int, seed: int) -> dict:
    """Point and resampled precision / recall of the top `frac` of the (resampled) alerts, for several scores on the same rows.

    A resample is a multinomial weight per cluster (the same draws for every score, so differences are paired). Ranking by a score is
    fixed (ties by row order), so a resample only needs cumulative weights: the top K = ceil(frac * resampled rows) rows count with
    their weights, the boundary row partly.
    """
    n = len(y)
    _, cl = np.unique(clusters, return_inverse=True)
    k = int(cl.max()) + 1
    orders = {name: np.argsort(-s, kind="stable") for name, s in scores.items()}
    big_k = int(math.ceil(frac * n))
    out = {name: {"k": big_k, "precision": float(y[o][:big_k].sum() / big_k),
                  "recall": float(y[o][:big_k].sum() / y.sum()) if y.sum() > 0 else float("nan"), "prec_s": [], "rec_s": []} for name, o in orders.items()}  # fmt: skip
    chunk = max(1, int(2_000_000 // max(n, 1)))
    for wc in bs._weights(k, n_boot, seed, chunk):
        wr = wc[:, cl].astype("f8")  # (B, n) row weights
        total = wr.sum(axis=1)
        kk = np.ceil(frac * total)
        for name, o in orders.items():
            ww = wr[:, o]
            prev = np.cumsum(ww, axis=1) - ww
            incl = np.clip(kk[:, None] - prev, 0.0, ww)
            tp = (incl * y[o][None, :]).sum(axis=1)
            pos = (ww * y[o][None, :]).sum(axis=1)
            out[name]["prec_s"].append(tp / kk)
            out[name]["rec_s"].append(np.where(pos > 0, tp / np.where(pos > 0, pos, 1), np.nan))
    for name in out:
        out[name]["prec_s"], out[name]["rec_s"] = np.concatenate(out[name]["prec_s"]), np.concatenate(out[name]["rec_s"])
    return out


def review_capacity(panel: Panel, w) -> list[dict]:
    """Table `review_capacity` (S1): precision and recall at the review capacity, per system, with paired CIs against one bank."""
    cap, source = resolve_capacity(panel)
    frac = float(cap["fraction"])
    items: dict[str, np.ndarray] = {}
    for name, s in panel.sources.items():
        items[name] = s.p_e2e
    for key, s in panel.scores.items():
        items[key.label()] = s.p_e2e
    rows = []
    for label, p in items.items():
        m = np.isfinite(p)
        if m.sum() < 2 or panel.y[m].sum() == 0:
            rows.append({"system": label, "n": int(m.sum()), "note": "not_run: fewer than 2 alerts or no positive alert"})
            continue
        r = _topk_resamples({"one bank": panel.p0[m], "system": p[m]}, panel.y[m], panel.clusters[m], frac, panel.n_boot, panel.seed)
        ob, sy = r["one bank"], r["system"]
        dp, dr = sy["prec_s"] - ob["prec_s"], sy["rec_s"] - ob["rec_s"]
        rows.append({"system": label, "n": int(m.sum()), "positives": int(panel.y[m].sum()), "base rate": float(panel.y[m].mean()), "reviewed (K)": sy["k"],
                     "precision": sy["precision"],
                     "precision 95% CI": _fci(fn.ci_tuple(sy["prec_s"]), 4), "recall": sy["recall"], "recall 95% CI": _fci(fn.ci_tuple(sy["rec_s"]), 4),
                     "precision minus one bank": sy["precision"] - ob["precision"], "precision diff 95% CI": _fci(fn.ci_tuple(dp), 4),
                     "recall minus one bank": sy["recall"] - ob["recall"], "recall diff 95% CI": _fci(fn.ci_tuple(dr), 4)})  # fmt: skip
    ok = [r for r in rows if "precision" in r]
    if not ok:
        raise Missing("no system with scores and a positive alert for the review-capacity table")
    ob_row = {"system": "one bank (p0)", "n": len(panel.y), "positives": int(panel.y.sum()), "base rate": float(panel.y.mean()),
              "reviewed (K)": int(math.ceil(frac * len(panel.y)))}  # fmt: skip
    r0 =_topk_resamples({"one bank": panel.p0}, panel.y, panel.clusters, frac, panel.n_boot, panel.seed)["one bank"]
    ob_row.update({"precision": r0["precision"], "precision 95% CI": _fci(fn.ci_tuple(r0["prec_s"]), 4), "recall": r0["recall"],
                   "recall 95% CI": _fci(fn.ci_tuple(r0["rec_s"]), 4), "note": "reference"})  # fmt: skip
    rows = [ob_row, *rows]
    cols = ["system", "n", "positives", "base rate", "reviewed (K)", "precision", "precision 95% CI", "recall", "recall 95% CI", "precision minus one bank",
            "precision diff 95% CI", "recall minus one bank", "recall diff 95% CI", "note"]  # fmt: skip
    note = (f"S1 (secondary / sensitivity, outside the Holm families, never changes a verdict). Capacity: the top {frac:.0%} of the alerts of each comparison "
            f"by score are reviewed ({source}; status: {cap.get('status', 'fixed')}). Precision = positives among the reviewed / reviewed, recall = positives "
            f"among the reviewed / all positives. Ties are broken by alert order; the capacity is re-applied to every bootstrap resample; differences are paired "
            f"(same resampled clusters) against the one-bank score on the same alerts.")  # fmt: skip
    w.table("review_capacity", rows, cols, f"S1: precision and recall at the review capacity (top {frac:.0%} of alerts by score)", note=note)
    w.status["review_capacity"].update(capacity=cap, capacity_source=source, tier="sensitivity (S1)")
    return rows


# --------------------------------------------------------------------------------------------------------------------
# p0-adjusted paired differences (S2, issue 88)
# --------------------------------------------------------------------------------------------------------------------
def answered_per_alert(panel: Panel, key: fn.Key | None) -> np.ndarray | None:
    """Answered questions per alert of one system (mean over seeds), aligned to the master alerts; None if the logs hold no step rows for it."""
    if key is None or panel.run is None:
        return None
    log = panel.run.log
    s = log.filter((pl.col("event") == "step") & (pl.col("system") == key.system) & (pl.col("variant") == key.variant) & (pl.col("mode") == key.mode)
                   & (pl.col("_alert_set") == key.alert_set))  # fmt: skip
    if s.is_empty():
        return None
    if "response_kind" in s.columns:
        s = s.filter(pl.col("response_kind") == "reply")
    v = log.filter((pl.col("event") == "verdict") & (pl.col("system") == key.system) & (pl.col("variant") == key.variant) & (pl.col("mode") == key.mode)
                   & (pl.col("_alert_set") == key.alert_set))  # fmt: skip
    n_seeds = max(1, v["seed"].n_unique())
    cnt = dict(s.group_by("alert_id").len().iter_rows())
    return np.array([cnt.get(a, 0) / n_seeds for a in panel.alerts["alert_id"].to_list()], dtype="f8")


def adjusted_difference(d: np.ndarray, covariates: dict[str, np.ndarray], clusters: np.ndarray, n_boot: int, seed: int) -> dict:
    """Unadjusted and covariate-adjusted mean of per-alert differences `d`, with cluster-bootstrap CIs.

    The adjusted estimate is the intercept of the OLS fit of d on the centred covariates, i.e. the fit evaluated at the covariate means of the
    original alerts: with an intercept in the model this equals mean(d) exactly, so the estimand and the point estimate do not change; what
    changes is the sampling variability. Every bootstrap resample (whole clusters, the same draws for both estimators) refits the regression
    and evaluates it at the ORIGINAL covariate means; covariate imbalance in the resample is thereby removed from the draw. Constant covariates are dropped.
    """
    d = np.asarray(d, dtype="f8")
    n = len(d)
    names = [k for k, x in covariates.items() if np.ptp(x) > 1e-12]
    dropped = [k for k in covariates if k not in names]
    X = np.column_stack([covariates[k] - covariates[k].mean() for k in names]) if names else np.zeros((n, 0))
    Z = np.column_stack([np.ones(n), X])
    q = Z.shape[1]
    _, cl = np.unique(clusters, return_inverse=True)
    k = int(cl.max()) + 1
    outer = (Z[:, :, None] * Z[:, None, :]).reshape(n, q * q)
    a_c, b_c, d_c = np.zeros((k, q * q)), np.zeros((k, q)), np.zeros(k)
    np.add.at(a_c, cl, outer)
    np.add.at(b_c, cl, Z * d[:, None])
    np.add.at(d_c, cl, d)
    cnt = np.bincount(cl, minlength=k).astype("f8")
    beta0 = np.linalg.pinv(Z.T @ Z, rcond=1e-10) @ (Z.T @ d)
    resid = d - Z @ beta0
    sst = float(((d - d.mean()) ** 2).sum())
    unadj_s, adj_s, slope_s = [], [], []
    for wc in bs._weights(k, n_boot, seed, 500):
        unadj_s.append((wc @ d_c) / (wc @ cnt))
        a = (wc @ a_c).reshape(-1, q, q)
        beta = (np.linalg.pinv(a, rcond=1e-10) @ (wc @ b_c)[:, :, None])[:, :, 0]
        adj_s.append(beta[:, 0])
        slope_s.append(beta[:, 1:])
    un, ad, sl = np.concatenate(unadj_s), np.concatenate(adj_s), np.vstack(slope_s)
    ci_un, ci_ad = fn.ci_tuple(un), fn.ci_tuple(ad)
    return {"n": n, "n_clusters": k, "unadjusted": float(d.mean()), "unadjusted_ci": ci_un, "adjusted": float(beta0[0]), "adjusted_ci": ci_ad,
            "width_unadjusted": ci_un[1] - ci_un[0], "width_adjusted": ci_ad[1] - ci_ad[0], "covariates": names, "dropped_constant": dropped,
            "slopes": {nm: {"estimate": float(beta0[1 + i]), "ci": fn.ci_tuple(sl[:, i])} for i, nm in enumerate(names)},
            "r2": float(1 - (resid**2).sum() / sst) if sst > 0 else float("nan"), "samples_unadjusted": un, "samples_adjusted": ad}  # fmt: skip


def adjusted_estimates(panel: Panel, w) -> list[dict]:
    """Table `adjusted_estimates` (S2): the paired Brier differences of H1 and H3, unadjusted and adjusted for p0 and the answered questions."""
    comps: list[tuple[str, str, Scores, np.ndarray, dict]] = []  # (hypothesis, label, our scores, the other system's squared errors, extra vectors for the mask)
    missing: list[str] = []
    try:
        ours = panel.role("ours_compose")
        ups = panel.uppers()
        extra = {"up": next(iter(ups.values())).se_e2e} if ups else {}  # H1's alerts: those the first upper limit also covers
        comps.append(("H1", "Brier(ours) - Brier(one bank)", ours, fn._one_bank_vec(panel), extra))
    except Missing as e:
        missing.append(f"H1: {e}")
    try:
        ours = panel.role("ours_compose")
        best, _ = panel.best_simple()
        others = {f"best simple ({best})": panel.role(best).se_e2e, "enumerator": panel.role("enumerator").se_e2e, "llm fixed list": panel.role("llm_fixed_list").se_e2e}
        for nm, se in others.items():  # H3's alerts: those all four systems share
            comps.append(("H3", f"Brier(ours) - Brier({nm})", ours, se, {f"o{j}": v for j, v in enumerate(others.values())}))
    except Missing as e:
        missing.append(f"H3: {e}")
    if not comps:
        raise Missing("; ".join(missing) or "no H1/H3 system in the run")
    rows, details = [], {}
    for hyp, label, ours, other, extra in comps:
        vec = {"ours": ours.se_e2e, "other": other, **extra}
        m = panel.common(vec)
        d = (ours.se_e2e - other)[m]
        cov = {"p0": panel.p0[m]}
        na = answered_per_alert(panel, ours.key)
        note = SENSITIVITY_LABEL
        if na is not None:
            cov["answered questions"] = na[m]
        else:
            note += "; no step rows in the log, adjusted for p0 only"
        r = adjusted_difference(d, cov, panel.clusters[m], panel.n_boot, panel.seed)
        if r["dropped_constant"]:
            note += "; constant covariate dropped: " + ", ".join(r["dropped_constant"])
        sl = r["slopes"]
        rows.append({"hypothesis": hyp, "comparison": label, "n": r["n"], "clusters": r["n_clusters"], "unadjusted": r["unadjusted"],
                     "unadjusted 95% CI": _fci(r["unadjusted_ci"], 4), "unadjusted CI width": r["width_unadjusted"], "adjusted": r["adjusted"],
                     "adjusted 95% CI": _fci(r["adjusted_ci"], 4), "adjusted CI width": r["width_adjusted"],
                     "width ratio (adjusted / unadjusted)": r["width_adjusted"] / r["width_unadjusted"] if r["width_unadjusted"] > 0 else None,
                     "slope on p0": sl["p0"]["estimate"] if "p0" in sl else None, "slope on p0 95% CI": _fci(sl["p0"]["ci"], 4) if "p0" in sl else None,
                     "slope on answered questions": sl["answered questions"]["estimate"] if "answered questions" in sl else None,
                     "slope on answered 95% CI": _fci(sl["answered questions"]["ci"], 4) if "answered questions" in sl else None,
                     "R2 (in sample)": r["r2"], "note": note})  # fmt: skip
        details[f"{hyp}: {label}"] = {k: v for k, v in r.items() if not k.startswith("samples")}
    cols = ["hypothesis", "comparison", "n", "clusters", "unadjusted", "unadjusted 95% CI", "unadjusted CI width", "adjusted", "adjusted 95% CI",
            "adjusted CI width", "width ratio (adjusted / unadjusted)", "slope on p0", "slope on p0 95% CI", "slope on answered questions",
            "slope on answered 95% CI", "R2 (in sample)", "note"]  # fmt: skip
    note = (f"S2 ({SENSITIVITY_LABEL}; outside the Holm families, never changes a verdict). Per-alert Brier difference (negative = ours better) regressed by OLS on "
            f"p0 and the number of answered questions of our system; the adjusted estimate is the fit at the original covariate means, so it equals the unadjusted "
            f"mean by construction and only its CI differs; cluster bootstrap refits the regression on every resample of whole clusters.")  # fmt: skip
    w.table("adjusted_estimates", rows, cols, f"S2: p0-adjusted paired Brier differences of H1 and H3 ({SENSITIVITY_LABEL})", note=note)
    w.status["adjusted_estimates"].update(tier="sensitivity (S2)", label=SENSITIVITY_LABEL, details=details, not_built=missing)
    return rows
