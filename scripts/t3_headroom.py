"""Headroom study and tuning of the non-LLM question pipeline (docs/t3_headroom.md).  CPU only.  Owner: A1.

    python scripts/t3_headroom.py select   # TRAIN cross-fitted + val fit+cal choices; writes the frozen files
    python scripts/t3_headroom.py eval     # val eval (160 askable alerts), ONCE, after the freeze

`select` never reads a val `eval` label: the eval alerts are dropped from the val split on load.  It writes
configs/fusion/headroom_v1.json (ConditionalFusion params + hash), configs/fusion/headroom_v1_decisions.json (every choice, the
file hashes, a timestamp), configs/fusion/headroom_v1_table.jsonl (the LR table with the hypotheses the fusion uses marked
accepted) and configs/decision/headroom_v1.yaml.  `eval` refuses to run unless the decisions file exists and records its sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ftm.decision.threshold import fit_threshold, stop_bounds  # noqa: E402
from ftm.eval.bootstrap import bootstrap_means, percentile_ci  # noqa: E402
from ftm.fusion.conditional import ConditionalFusion  # noqa: E402
from ftm.fusion.fuse import logit, sigmoid  # noqa: E402
from ftm.fusion.headroom import (  # noqa: E402
    REGIMES,
    Design,
    Logit,
    Scorer,
    cv_predict,
    fit_train,
    load_all,
    log_lr_matrix,
    make_folds,
    metrics,
    oof_log_lr,
    recal_ab,
    run_policy,
    scorer_of,
    subsplit,
)  # fmt: skip

warnings.filterwarnings("ignore")
OUT = Path("outputs/headroom")
FUSION_JSON = Path("configs/fusion/headroom_v1.json")
DECISIONS = Path("configs/fusion/headroom_v1_decisions.json")
TABLE = Path("configs/fusion/headroom_v1_table.jsonl")
CLIP = 0.1
SHRINKS = (0.25, 0.5, 0.75, 1.0)
ALLOC = {"e2": (2.0,), "e1": (1.0,), "e05": (0.5,), "mix": (0.25, 0.5, 1.0, 2.0)}
N_RANDOM = 30


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - y) ** 2))


# ------------------------------------------------------------------------------------------------ val-side fitting
BASES = ("recal", "intercept", "p0")  # (a, b) both fitted | only a fitted (b = 1, a prior shift) | none (a = 0, b = 1)


def fit_ab(base, p0, y, offset=None):
    if base == "p0":
        return 0.0, 1.0
    return recal_ab(p0, y, offset, ridge=1e9 if base == "intercept" else 1.0, clip=CLIP)


def fit_on(base, sc0: Scorer, shrink: float, regime: str, X, cell, p0, y, hyps, stop=None):
    """Fit (a, b) per `base` on the given rows with the evidence of the EIG policy at `shrink` as an offset; returns (Scorer, asks)."""
    a0, b0 = fit_ab(base, p0, y)
    sc = Scorer(a0, b0, sc0.w_h, shrink, sc0.clip)
    asks = run_policy("eig", cell, X, p0, hyps, sc, ALLOC[regime], y=y, stop=stop)
    a, b = fit_ab(base, p0, y, shrink * ((X * asks) @ sc.w_h))
    sc = Scorer(a, b, sc0.w_h, shrink, sc0.clip)
    return sc, run_policy("eig", cell, X, p0, hyps, sc, ALLOC[regime], y=y, stop=stop)


def cross_part(vs, sc0, base, shrink, regime, X, hyps):
    """Pooled predictions of fit->cal and cal->fit: (a, b) fitted on one part, policy and score on the other."""
    pred = np.zeros(len(vs.y))
    for tr_p, te_p in (("fit", "cal"), ("cal", "fit")):
        a, b = np.flatnonzero(vs.part == tr_p), np.flatnonzero(vs.part == te_p)
        sc, _ = fit_on(base, sc0, shrink, regime, X[a], vs.cell[a], vs.p0[a], vs.y[a], hyps)
        asks = run_policy("eig", vs.cell[b], X[b], vs.p0[b], hyps, sc, ALLOC[regime], y=vs.y[b])
        pred[b] = sigmoid(sc.z(vs.p0[b], X[b], asks))
    return pred


def floor_pred(base, vs, fit_idx=None, apply_idx=None):
    """Evidence-free base ('p0 recalibrated' for recal / intercept, clipped raw p0 for p0): fitted on fit_idx, applied to apply_idx."""
    fi = np.arange(len(vs.y)) if fit_idx is None else fit_idx
    ai = np.arange(len(vs.y)) if apply_idx is None else apply_idx
    a, b = fit_ab(base, vs.p0[fi], vs.y[fi])
    return sigmoid(Scorer(a, b, np.zeros(1), 0.0, CLIP).base(vs.p0[ai])), (a, b)


def floor_cross_part(vs, base="recal"):
    pred = np.zeros(len(vs.y))
    for tr_p, te_p in (("fit", "cal"), ("cal", "fit")):
        a, b = np.flatnonzero(vs.part == tr_p), np.flatnonzero(vs.part == te_p)
        pred[b] = floor_pred(base, vs, a, b)[0]
    return pred


# ------------------------------------------------------------------------------------------------ select
def select(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    hyps, tr, va = load_all()
    va = subsplit(va, np.flatnonzero(va.part != "eval"))  # eval labels are never read below this line
    assert not (va.part == "eval").any()
    acc = np.array([h.accepted for h in hyps])
    allh = np.ones(len(hyps), dtype=bool)
    n, y = len(tr.y), tr.y
    res: dict = {"n_train_alerts": n, "n_val_fitcal": len(va.y), "train_label_rate": float(y.mean()),
                 "val_fitcal_label_rate": float(va.y.mean())}  # fmt: skip

    # ---- 1. train cross-fitted comparison of fusion forms (5 interleaved time-block folds; features out-of-fold)
    folds, blk = make_folds(tr, hyps)
    zero = lambda d: Logit(d, l2=1e9, prior_w=0.0)  # noqa: E731  evidence off: recalibrated p0 only
    tab = {}

    def cv(label, kind, use, mk, dense=False):
        d = Design.make(hyps, kind, use)
        tab[label] = {rg: brier(cv_predict(folds, blk, mk, d, rg, n, dense), y) for rg in (["dense"] if dense else list(REGIMES))}
        print(label, {k: round(v, 5) for k, v in tab[label].items()}, flush=True)

    cv("p0_recal", "fam", allh, zero)
    cv("naive_w1_accepted(6)", "fam", acc, lambda d: Logit(d, l2=1e9))
    for l2 in (1, 3, 10, 30, 100):
        cv(f"fam_accepted_l2={l2}", "fam", acc, lambda d, l2=l2: Logit(d, l2=l2))
        cv(f"fam_all_l2={l2}", "fam", allh, lambda d, l2=l2: Logit(d, l2=l2))
    for l2 in (3, 30):
        cv(f"fam_all_prior0_l2={l2}", "fam", allh, lambda d, l2=l2: Logit(d, l2=l2, prior_w=0.0))
        cv(f"hyp_all_prior1_l2={l2}", "hyp", allh, lambda d, l2=l2: Logit(d, l2=l2))
    for l2 in (3, 30):
        cv(f"binfx_all_l2={l2}", "bin", allh, lambda d, l2=l2: Logit(d, l2=l2, prior_w=0.0))
    zc = float(logit(tr.p0).mean())

    def aug(F, p0):
        return np.hstack([F, F * (logit(p0) - zc)[:, None]])

    d = Design.make(hyps, "fam", allh)
    k = d.n_cols
    tab["fam_all_x_p0_interaction"] = {
        rg: brier(cv_predict(folds, blk, lambda dd: Logit(dd, l2=np.r_[np.full(k, 3.0), np.full(k, 100.0)],
                                                          prior_w=np.r_[np.ones(k), np.zeros(k)]), d, rg, n, aug=aug), y)
        for rg in REGIMES}  # fmt: skip
    print("interaction", tab["fam_all_x_p0_interaction"], flush=True)
    cv("dense_fam_all_prior0_l2=10", "fam", allh, lambda dd: Logit(dd, l2=10, prior_w=0.0), dense=True)
    cv("dense_fam_eps2_prior0_l2=10", "fam", np.array([h.eps == 2.0 for h in hyps]), lambda dd: Logit(dd, l2=10, prior_w=0.0),
       dense=True)  # fmt: skip
    res["train_cv_brier"] = tab
    cand = [k_ for k_ in tab if k_.startswith("fam_all_l2=")]
    l2 = int(min(cand, key=lambda k_: np.mean([tab[k_][r] for r in ("e2", "e1", "e05", "mix")])).split("=")[1])
    res["chosen_l2"] = l2
    # interaction: reject unless it beats the plain form by more than 2% of the gain over recalibrated p0
    plain = np.mean([tab[f"fam_all_l2={l2}"][r] for r in ("e2", "e1", "e05")])
    inter = np.mean([tab["fam_all_x_p0_interaction"][r] for r in ("e2", "e1", "e05")])
    base_ = np.mean([tab["p0_recal"][r] for r in ("e2", "e1", "e05")])
    res["interaction_supported"] = bool(plain - inter > 0.02 * (base_ - plain))

    # ---- 2. final train fits
    m_all, des_all, lrs = fit_train(tr, hyps, l2=l2)
    m_acc, des_acc, _ = fit_train(tr, hyps, l2=l2, use=acc)
    m_dense, des_dense, _ = fit_train(tr, hyps, l2=10, dense=True)
    xv = log_lr_matrix(va, hyps, lrs)
    fams = des_all.families

    # ---- 3. val fit+cal: floors and the base / question-set / allocation / shrink grid, scored cross-part (fit->cal, cal->fit)
    yv = va.y
    fl = {}
    res["floors_fitcal"] = {"p0_v2 raw": metrics(va.p0, yv), "p0_v1 raw": metrics(va.p0_v1, yv),
                            "constant(in-sample)": {"brier": brier(yv.mean(), yv)}}  # fmt: skip
    for base in BASES:
        pf_, ab = floor_pred(base, va)
        fl[base] = ab
        res["floors_fitcal"][f"floor[{base}] in-sample"] = {**metrics(pf_, yv), "a": ab[0], "b": ab[1]}
        res["floors_fitcal"][f"floor[{base}] cross-part"] = metrics(floor_cross_part(va, base), yv)
    grid = {}
    models = {"all": (m_all, des_all), "accepted": (m_acc, des_acc)}
    for mname, (m, des) in models.items():
        sc0 = scorer_of(m, des, hyps, 1.0)
        sc0.clip = CLIP
        for base in BASES:
            for rg in ALLOC:
                for s in SHRINKS:
                    key = f"{mname}/{base}/{rg}/{s:g}"
                    grid[key] = brier(cross_part(va, sc0, base, s, rg, xv, hyps), yv)
                    print("grid", key, round(grid[key], 5), flush=True)
    res["val_fitcal_grid_crosspart_brier"] = grid
    res["val_fitcal_floor_crosspart_brier"] = {b_: brier(floor_cross_part(va, b_), yv) for b_ in BASES}
    frozen = {}
    for tag, pick in (("primary", lambda k: True), ("accepted", lambda k: k.startswith("accepted/"))):
        best = min((k for k in grid if pick(k)), key=grid.get)
        mname, base, rg, s = best.split("/")
        s = float(s)
        m, des = models[mname]
        sc0 = scorer_of(m, des, hyps, 1.0)
        sc0.clip = CLIP
        sc, asks = fit_on(base, sc0, s, rg, xv, va.cell, va.p0, yv, hyps)  # (a, b) on all fit+cal
        frozen[tag] = dict(model=mname, base=base, alloc=rg, shrink=s, sc=sc, sc0=sc0, m=m, des=des, asks=asks,
                           crosspart_brier=grid[best], floor_crosspart_brier=res["val_fitcal_floor_crosspart_brier"][base])  # fmt: skip
    res["chosen"] = {t: {k: v for k, v in f.items() if k in ("model", "base", "alloc", "shrink", "crosspart_brier", "floor_crosspart_brier")}
                     for t, f in frozen.items()}  # fmt: skip
    P = frozen["primary"]
    sc, sc0, m, des, mname, rg, s, base = P["sc"], P["sc0"], P["m"], P["des"], P["model"], P["alloc"], P["shrink"], P["base"]
    pf, _ = floor_pred(base, va)
    post = sigmoid(sc.z(va.p0, xv, P["asks"]))

    # ---- 4. freeze: decision threshold and stop bounds (primary), the two fusion files, the table
    def stop_for(sc_, m_, asks_):
        used = np.flatnonzero(sc_.w_h != 0)
        return stop_bounds(((hyps[j].eps, dict(zip(hyps[j].cells, hyps[j].lr, strict=True))) for j in used),
                           [float(np.max(np.abs(sc_.w_h)))], sc_.shrink)  # fmt: skip

    delta, wmax = stop_for(sc, m, None)
    thr = {"f1": fit_threshold(post, yv, "f1"), "cost_1_1": fit_threshold(post, yv, "cost"),
           "capacity_top10pct": float(np.quantile(post, 0.9))}  # fmt: skip
    stop_arg = (float(logit(thr["f1"])), delta * wmax)
    asks_stop = run_policy("eig", va.cell, xv, va.p0, hyps, sc, ALLOC[rg], y=yv, stop=stop_arg)
    res["stop_rule_fitcal"] = {"max_delta": delta, "max_weight": wmax, "asks_no_stop": float(P["asks"].sum(1).mean()),
                               "asks_with_cannot_cross": float(asks_stop.sum(1).mean()),
                               "brier_no_stop": brier(post, yv),
                               "brier_with_stop": brier(sigmoid(sc.z(va.p0, xv, asks_stop)), yv)}  # fmt: skip
    res["thresholds_fitcal"] = thr
    files = {}
    rows0 = [json.loads(line) for line in Path("data/backtest/hi_medium/7613b89b5dcc/lr_table.jsonl").read_text().splitlines() if line]
    for tag, F in frozen.items():
        suffix = "" if tag == "primary" else "_accepted"
        fjson, ftab = FUSION_JSON.with_name(f"headroom_v1{suffix}.json"), TABLE.with_name(f"headroom_v1{suffix}_table.jsonl")
        w_fam = np.array([float(F["m"].w[i]) if any(h.family == f and F["sc"].w_h[j] != 0 for j, h in enumerate(hyps)) else 0.0
                          for i, f in enumerate(fams)])  # fmt: skip
        cf = ConditionalFusion(list(fams), F["sc"].a, F["sc"].b, w_fam, float(l2), F["shrink"], "p0" if F["base"] == "p0" else "fitted",
                               0.0, True, CLIP, {"name": f"headroom_v1{suffix}", "train_alerts": n, "p0": "data/alerts/hi_medium_v2",
                                                 "model": F["model"], "base_kind": F["base"], "alloc": F["alloc"], "l2": l2,
                                                 "table": str(ftab), "ab_fit": "val fit+cal"})  # fmt: skip
        h = cf.to_json(fjson)
        usedfam = {f for f, w in zip(fams, w_fam, strict=True) if w != 0.0}
        rows = []
        for r in rows0:
            r = dict(r)
            r["accepted_ebh"] = r["accepted"]
            r["accepted"] = bool(r["family"] in usedfam and (F["model"] == "all" or r["accepted_ebh"]))
            rows.append(r)
        ftab.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        files[tag] = {"fusion": {"path": str(fjson), "sha256": sha(fjson), "content_hash": h},
                      "table": {"path": str(ftab), "sha256": sha(ftab)}, "weights": dict(zip(fams, w_fam.round(4).tolist(), strict=True))}  # fmt: skip
    order = fixed_order(tr, hyps, xv_train=oof_log_lr(tr, hyps, 5))
    dec = {"name": "headroom_v1", "written_before_eval": True, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "p0_clip": CLIP, "l2": l2, "dense_l2": 10, "thresholds": thr, "chosen_threshold": thr["f1"], "threshold_objective": "f1",
           "stop": {"max_delta": delta, "max_weight": wmax}, "interaction_supported": res["interaction_supported"],
           "fixed_order": order, "n_random_seeds": N_RANDOM, "floor_ab": fl,
           "variants": {t: {"model": F["model"], "base": F["base"], "alloc": F["alloc"], "eps_menu": list(ALLOC[F["alloc"]]),
                            "shrink": F["shrink"], "a": F["sc"].a, "b": F["sc"].b, **files[t]} for t, F in frozen.items()},
           "files": {"train_table": {"path": "data/backtest/hi_medium/7613b89b5dcc/lr_table.jsonl",
                                     "sha256": sha("data/backtest/hi_medium/7613b89b5dcc/lr_table.jsonl")},
                     "val_rows": {"path": "data/backtest/hi_medium/f8b676b7d1fc/alert_rows.parquet",
                                  "sha256": sha("data/backtest/hi_medium/f8b676b7d1fc/alert_rows.parquet")}}}  # fmt: skip
    DECISIONS.write_text(json.dumps(dec, indent=1, sort_keys=True))
    write_decision_yaml(dec)
    res["decisions_sha256"] = sha(DECISIONS)

    # ---- 5. ceilings on val fit+cal (frozen choices; in-sample (a, b), plus cross-part for the realistic rows)
    dense = (m_dense, des_dense)
    res["ceilings_fitcal"] = {tag: ceiling_table(va, xv, hyps, F["sc"], F["sc0"], F["base"], F["alloc"], order, dense, cross=True)
                              for tag, F in frozen.items()}  # fmt: skip
    res["family_weights"] = {t: f["weights"] for t, f in files.items()}
    res["collapse_fitcal"] = {t: collapse(sigmoid(F["sc"].z(va.p0, xv, F["asks"])), va.p0, yv, F["sc"], floor_pred(F["base"], va)[0])
                              for t, F in frozen.items()}  # fmt: skip
    (OUT / "select.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("chosen", "stop_rule_fitcal", "thresholds_fitcal", "collapse_fitcal")}, indent=1))
    print(f"frozen {[f['fusion']['content_hash'] for f in files.values()]}; {time.time() - t0:.0f}s")


def collapse(post, p0, y, sc, p_floor) -> dict:
    z, z0 = logit(post), logit(np.clip(p0, CLIP, 1 - CLIP))
    slope = float(np.polyfit(z0, z, 1)[0])
    mf, mfl = metrics(post, y), metrics(p_floor, y)
    return {"fitted_b": sc.b, "fitted_a": sc.a, "shrink": sc.shrink, "slope_of_posterior_on_logit_p0": slope,
            "posterior_var": mf["var_p"], "floor_var": mfl["var_p"], "var_ratio_vs_floor": mf["var_p"] / max(mfl["var_p"], 1e-12),
            "auc": mf["auc"], "floor_auc": mfl["auc"], "cal_slope": mf.get("cal_slope"), "floor_cal_slope": mfl.get("cal_slope"),
            "mean_post": mf["mean_p"], "base_rate": float(y.mean()), "brier": mf["brier"], "floor_brier": mfl["brier"]}  # fmt: skip


def fixed_order(tr, hyps, xv_train) -> list[int]:
    """Hypotheses ranked by their TRAIN out-of-fold Brier gain over the constant base rate (naive w = 1), best first."""
    y = tr.y
    base = np.full(len(y), y.mean())
    z = float(logit(np.array([y.mean()]))[0])
    gain = np.full(len(hyps), -np.inf)
    for j in range(len(hyps)):
        m = tr.cell[:, j] >= 0
        if m.sum() > 50:
            gain[j] = brier(base[m], y[m]) - brier(sigmoid(z + xv_train[m, j]), y[m])
    return [int(j) for j in np.argsort(-gain, kind="stable")]


def ceiling_table(vs, xv, hyps, sc, sc0, base, rg, order, dense, floor_ab=None, cross=False, seeds=N_RANDOM):
    """Rows of the ceiling table on `vs` for one frozen scorer `sc` (a, b, shrink fixed; floor refit on these rows unless
    `floor_ab` is given).  `dense` = (model, design) of the all-answers fit used by the information ceiling."""
    y, p0 = vs.y, vs.p0
    m_dense, des_dense = dense
    out = {"p0_v1 raw": metrics(vs.p0_v1, y), "p0_v2 raw": metrics(p0, y)}
    for b_ in BASES:
        ab = fit_ab(b_, p0, y) if floor_ab is None else floor_ab[b_]
        out[f"floor: p0 [{b_}]"] = metrics(sigmoid(Scorer(ab[0], ab[1], np.zeros(1), 0.0, CLIP).base(p0)), y)
    menu = ALLOC[rg]
    dsc0 = scorer_of(m_dense, des_dense, hyps, 1.0)
    for nm, ep in (("info ceiling: all 110 programs at eps 2", (2.0,)), ("info ceiling: all 440 (program, eps) answers", ALLOC["mix"])):
        allasks = (vs.cell >= 0) & np.isin([h.eps for h in hyps], list(ep))[None, :]
        best = None
        for s_ in (0.25, 0.5, 0.75, 1.0):
            a_, b_ = fit_ab(base, p0, y, s_ * ((xv * allasks) @ dsc0.w_h))
            pr = sigmoid(Scorer(a_, b_, dsc0.w_h, s_, CLIP).z(p0, xv, allasks))
            if best is None or brier(pr, y) < best[0]:
                best = (brier(pr, y), s_, pr)
        out[nm] = {**metrics(best[2], y), "shrink": best[1], "note": "dense model; shrink and (a, b) refit on these rows (optimistic)"}
    asks = run_policy("eig", vs.cell, xv, p0, hyps, sc, menu, y=y)
    out["EIG (label-free) [FROZEN CHOICE]"] = {**metrics(sigmoid(sc.z(p0, xv, asks)), y), "asks_per_alert": float(asks.sum(1).mean())}
    for r2 in ALLOC:
        if r2 != rg:
            a2 = run_policy("eig", vs.cell, xv, p0, hyps, sc, ALLOC[r2], y=y)
            out[f"EIG alloc {r2}"] = {**metrics(sigmoid(sc.z(p0, xv, a2)), y), "asks_per_alert": float(a2.sum(1).mean())}
    br = [brier(sigmoid(sc.z(p0, xv, run_policy("random", vs.cell, xv, p0, hyps, sc, menu, rng=np.random.default_rng(sd)))), y)
          for sd in range(seeds)]  # fmt: skip
    out[f"random (mean of {seeds} seeds)"] = {"brier": float(np.mean(br)), "brier_sd": float(np.std(br))}
    a4 = run_policy("fixed", vs.cell, xv, p0, hyps, sc, menu, order=order)
    out["fixed best order (train-ranked)"] = {**metrics(sigmoid(sc.z(p0, xv, a4)), y), "asks_per_alert": float(a4.sum(1).mean())}
    a5 = run_policy("oracle_greedy", vs.cell, xv, p0, hyps, sc, menu, y=y)
    out["UNATTAINABLE label-aware oracle, eps-feasible greedy set"] = {**metrics(sigmoid(sc.z(p0, xv, a5)), y),
                                                                        "asks_per_alert": float(a5.sum(1).mean())}  # fmt: skip
    a6 = run_policy("oracle_greedy", vs.cell, xv, p0, hyps, sc, menu, y=y, max_steps=1)
    out["UNATTAINABLE label-aware oracle, best single ask"] = metrics(sigmoid(sc.z(p0, xv, a6)), y)
    if cross:
        out["EIG frozen choice (cross-part, honest)"] = metrics(cross_part(vs, sc0, base, sc.shrink, rg, xv, hyps), y)
        out[f"floor: p0 [{base}] (cross-part, honest)"] = metrics(floor_cross_part(vs, base), y)
    return out


def write_decision_yaml(dec):
    txt = f"""# Decision layer tuned for the headroom_v1 fusion (scripts/t3_headroom.py select); values fitted on HI-Medium val fit+cal only.
# Same keys as default.yaml; load with the same code.  Written before the val eval part was scored.
objective: f1                # f1 | cost | capacity; the thresholds of the other objectives are listed below
threshold: {dec['chosen_threshold']:.6f}      # posterior >= threshold -> escalate (f1 objective, val fit+cal)
thresholds_fitcal:
  f1: {dec['thresholds']['f1']:.6f}
  cost_1_1: {dec['thresholds']['cost_1_1']:.6f}
  capacity_top10pct: {dec['thresholds']['capacity_top10pct']:.6f}
capacity_per_day: null
cost_fn: null
cost_fp: null
stop:
  min_eps: 0.25
  max_delta: {dec['stop']['max_delta']:.6f}      # largest |log LR| per unit eps over the hypotheses the fusion uses (stop_bounds)
  max_weight: {dec['stop']['max_weight']:.6f}     # largest |w_f| times the shrink
fusion_params: configs/fusion/headroom_v1.json
fusion_decisions: configs/fusion/headroom_v1_decisions.json
"""
    Path("configs/decision/headroom_v1.yaml").write_text(txt)


# ------------------------------------------------------------------------------------------------ eval (once)
def evaluate(a):
    if not DECISIONS.exists():
        raise SystemExit("no frozen decisions file: run `select` first")
    dec = json.loads(DECISIONS.read_text())
    for k, f in dec["files"].items():
        if sha(f["path"]) != f["sha256"]:
            raise SystemExit(f"{f['path']} changed after the freeze ({k})")
    for t, v in dec["variants"].items():
        for k in ("fusion", "table"):
            if sha(v[k]["path"]) != v[k]["sha256"]:
                raise SystemExit(f"{v[k]['path']} changed after the freeze ({t})")
    hyps, tr, va = load_all()
    ev = subsplit(va, np.flatnonzero(va.part == "eval"))
    acc = np.array([h.accepted for h in hyps])
    m_dense, des_dense, _ = fit_train(tr, hyps, l2=dec["dense_l2"], dense=True)
    out = {"decisions_sha256": sha(DECISIONS), "n_eval": len(ev.y), "eval_label_rate": float(ev.y.mean()), "variants": {}}
    y, p0 = ev.y, ev.p0
    uniq, inv = np.unique(ev.cluster, return_inverse=True)
    out["clusters"] = int(len(uniq))
    for tag, v in dec["variants"].items():
        m, des, lrs = fit_train(tr, hyps, l2=dec["l2"], use=None if v["model"] == "all" else acc)
        cf = ConditionalFusion.from_json(v["fusion"]["path"])
        assert cf.content_hash() == v["fusion"]["content_hash"]
        xe = log_lr_matrix(ev, hyps, lrs)
        sc = scorer_of(m, des, hyps, v["shrink"], v["a"], v["b"])
        sc.clip = CLIP
        sc0 = scorer_of(m, des, hyps, 1.0)
        sc0.clip = CLIP
        tab = ceiling_table(ev, xe, hyps, sc, sc0, v["base"], v["alloc"], dec["fixed_order"], (m_dense, des_dense),
                            floor_ab={k: tuple(x) for k, x in dec["floor_ab"].items()})  # fmt: skip
        asks = run_policy("eig", ev.cell, xe, p0, hyps, sc, ALLOC[v["alloc"]], y=y)
        post = sigmoid(sc.z(p0, xe, asks))
        refs = {"recal(a,b frozen)": sigmoid(Scorer(*dec["floor_ab"]["recal"], np.zeros(1), 0.0, CLIP).base(p0)),
                f"floor[{v['base']}] (same base)": sigmoid(Scorer(*dec["floor_ab"][v["base"]], np.zeros(1), 0.0, CLIP).base(p0)),
                "p0_v2 raw": p0, "p0_v1 raw": ev.p0_v1}  # fmt: skip
        cis = {}
        for nm, ref in refs.items():
            d = (post - y) ** 2 - (ref - y) ** 2  # negative = fused better
            cis[f"brier(EIG fused) - brier({nm})"] = {"mean": float(d.mean()), "ci95": list(percentile_ci(bootstrap_means(d, inv)))}
        for nm, p in [("EIG fused", post), *refs.items()]:
            cis[f"brier({nm})"] = {"mean": brier(p, y), "ci95": list(percentile_ci(bootstrap_means((p - y) ** 2, inv)))}
        thr = dec["chosen_threshold"]
        esc = post >= thr
        out["variants"][tag] = {"table": tab, "paired_ci": cis, "collapse": collapse(post, p0, y, sc, refs[f"floor[{v['base']}] (same base)"]),
                                "decision_f1_threshold": {"threshold": thr, "escalated": int(esc.sum()), "tp": int((esc & (y > 0.5)).sum()),
                                                          "precision": float((y[esc] > 0.5).mean()) if esc.any() else None,
                                                          "recall": float(esc[y > 0.5].mean())}}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "eval"])
    a = ap.parse_args(argv)
    {"select": select, "eval": evaluate}[a.stage](a)


if __name__ == "__main__":
    main()
