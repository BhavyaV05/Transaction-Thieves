"""Headroom study: per-alert answer matrices from back-test rows, out-of-fold LRs, regularised fusion, offline policy simulation.

Owner: A1 (rewrite of 1 Nov).  Companion of `ftm.fusion.conditional`; the numbers and the protocol are in docs/t3_headroom.md.

A *hypothesis* is one (canonical program, eps) cell of the back-test.  Per split the rows give, for every alert and every
hypothesis the right bank could answer, one DP answer (the noise draw is fixed by the bank's seed), so any policy can be
simulated offline: asking hypothesis h on alert i returns `cell[i, h]`.  Cells are the TRAIN table's bins (`rebin`), so the
val answers are scored with train-fitted cells and nothing about val enters them.

Evidence features: x[i, h] = log LR of the cell, 0 when unasked/unanswered.  On train the LR is *out-of-fold* (estimated on the
other time blocks, same estimator and smoothing as the table), so a fusion fitted on train sees honest, noisy LRs, not the
in-sample ones.  Everything here is label-free except `oof_log_lr`, `fit_*` and the oracle.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ftm.backtest.lr import cells_for
from ftm.dsl.parser import arity, parse, sensitivity
from ftm.fusion.fuse import logit, sigmoid


@dataclass
class Hyp:
    canon: str
    eps: float
    family: str
    kind: str
    bins: tuple
    cells: list  # cell keys, index = cell id
    lr: np.ndarray  # table LR per cell (train, in-sample)
    p_clean: np.ndarray  # P(cell | clean) measured on train noisy answers
    p_launder: np.ndarray
    accepted: bool
    sens: int
    ar: int | None
    qfam: str = ""  # dsl family of the PROGRAM (the table's `family`)


@dataclass
class Split:
    """Per-alert design of one split: alerts in time order."""

    alert_ids: list
    p0: np.ndarray
    p0_v1: np.ndarray
    y: np.ndarray
    part: np.ndarray  # val_part (fit/cal/eval) or "train"
    cell: np.ndarray  # (n_alert, H) int16, -1 = no answer
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # timestamps (datetime64)
    cluster: np.ndarray = field(default_factory=lambda: np.zeros(0))  # laundering-attempt id (own id when none): bootstrap unit


def load_table(path: str | Path) -> list[Hyp]:
    hyps = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        node = parse(r["program_canonical"])
        cells = sorted(r["lr"])
        pc = r["p_cell"]
        hyps.append(Hyp(r["program_canonical"], float(r["eps"]), r["family"], r["kind"], tuple(r["bins"] or ()), cells,
                        np.array([r["lr"][c] for c in cells]), np.array([pc[c][0] for c in cells]),
                        np.array([pc[c][1] for c in cells]), bool(r["accepted"]), sensitivity(node), arity(node)))
    return hyps


def raw_cell(kind: str, bins, value: float) -> str:
    """The table cell of one raw (float) answer; bool answers are stored as 0/1 in the rows."""
    if kind == "kary":
        return str(int(value))
    return cells_for(kind, [bool(value) if kind == "bool" else float(value)], bins)[0]


def build_split(rows, hyps: list[Hyp], alerts, use_answer_bin: bool = False) -> Split:
    """`rows`: polars frame of alert_rows (alert_id, canonical, eps, answer, answer_bin, label); `alerts`: polars frame of the
    alerts (alert_id, p0, p0_v1, label, timestamp, val_part) to keep, any order (sorted by time here)."""

    alerts = alerts.sort("timestamp", "alert_id")
    ids = alerts["alert_id"].to_list()
    idx = {a: i for i, a in enumerate(ids)}
    hix = {(h.canon, h.eps): j for j, h in enumerate(hyps)}
    cell = np.full((len(ids), len(hyps)), -1, dtype="i2")
    cmap = [{c: k for k, c in enumerate(h.cells)} for h in hyps]
    for r in rows.select("alert_id", "canonical", "eps", "answer", "answer_bin").iter_rows():
        i, j = idx.get(r[0]), hix.get((r[1], float(r[2])))
        if i is None or j is None or r[3] is None:
            continue
        h = hyps[j]
        key = r[4] if use_answer_bin else raw_cell(h.kind, h.bins, r[3])
        k = cmap[j].get(key)
        if k is not None:
            cell[i, j] = k
    part = alerts["val_part"].to_numpy() if "val_part" in alerts.columns else np.full(len(ids), "train")
    p1 = alerts["p0_v1"].to_numpy() if "p0_v1" in alerts.columns else np.full(len(ids), np.nan)
    att = alerts["attempt_id"].to_list() if "attempt_id" in alerts.columns else [None] * len(ids)
    cl = np.array([f"a{a}" if a is not None and a >= 0 else f"s{i}" for i, a in enumerate(att)])
    return Split(ids, alerts["p0"].to_numpy().astype("f8"), p1.astype("f8"), alerts["label"].to_numpy().astype("f8"), part,
                 cell, alerts["timestamp"].to_numpy(), cl)  # fmt: skip


def log_lr_matrix(split: Split, hyps: list[Hyp], lr_of=None) -> np.ndarray:
    """x[i, h] = log LR of the answered cell (0 if unanswered); `lr_of[h]` = per-cell LR arrays (default: the table's)."""
    X = np.zeros(split.cell.shape)
    for j, h in enumerate(hyps):
        lr = h.lr if lr_of is None else lr_of[j]
        m = split.cell[:, j] >= 0
        X[m, j] = np.log(lr[split.cell[m, j]])
    return X


def time_blocks(n: int, k: int, per_fold: int = 4) -> np.ndarray:
    """Fold id per alert (rows in time order): `k * per_fold` contiguous blocks dealt round-robin to `k` folds, so every fold
    sees every era (the train label rate drifts from 0 to 0.36 over the period) while neighbours stay together."""
    nb = k * per_fold
    return (np.minimum((np.arange(n) * nb) // n, nb - 1)) % k


def cell_lr(c_clean: np.ndarray, c_launder: np.ndarray, ceiling: float, alpha: float = 0.01) -> np.ndarray:
    """The table's estimator on cell counts: (P(cell | launder) + alpha) / (P(cell | clean) + alpha), clipped to the DP ceiling."""
    p_l = c_launder / max(c_launder.sum(), 1)
    p_c = c_clean / max(c_clean.sum(), 1)
    return np.clip((p_l + alpha) / (p_c + alpha), 1.0 / ceiling, ceiling)


def ceiling_of(h: Hyp) -> float:
    """The DP clip of a hypothesis, recovered from its table LRs (the largest |log LR| the table could have hit)."""
    return float(max(h.lr.max(), 1.0 / h.lr.min()))


def oof_log_lr(split: Split, hyps: list[Hyp], k: int = 5, alpha: float = 0.01) -> np.ndarray:
    """Train x matrix with each alert's LR estimated on the other `k` time blocks (same estimator as the table)."""
    blk = time_blocks(len(split.y), k)
    X = np.zeros(split.cell.shape)
    y = split.y > 0.5
    for j, h in enumerate(hyps):
        c, nc, ceil = split.cell[:, j], len(h.cells), ceiling_of(h)
        for b in range(k):
            tr, te = (blk != b) & (c >= 0), (blk == b) & (c >= 0)
            if not te.any():
                continue
            n1 = np.bincount(c[tr & y], minlength=nc).astype("f8")
            n0 = np.bincount(c[tr & ~y], minlength=nc).astype("f8")
            X[te, j] = np.log(cell_lr(n0, n1, ceil, alpha)[c[te]])
    return X


# ---------------------------------------------------------------- asks (offline policy simulation helpers)

def draw_asks(cell: np.ndarray, hyps: list[Hyp], rng: np.random.Generator, eps_menu: Sequence[float], budget: float = 2.0,
              one_per_family: bool = True, draws: int = 1) -> np.ndarray:
    """Random-policy ask masks, shape (draws, n, H): per alert, uniformly random answered hypotheses at an eps in `eps_menu`
    (a family at most once) while they fit the budget -- the `RandomPlanner` of ftm.planners.simple on the stored answers."""
    n, H = cell.shape
    eps = np.array([h.eps for h in hyps])
    fam = np.array([h.family for h in hyps])
    ok = (cell >= 0) & np.isin(eps, list(eps_menu))[None, :]
    out = np.zeros((draws, n, H), dtype=bool)
    for d in range(draws):
        for i in range(n):
            cand = np.flatnonzero(ok[i])
            left, used = budget, set()
            while len(cand):
                cand = cand[(eps[cand] <= left + 1e-9) & ~np.isin(fam[cand], list(used))] if one_per_family else cand[eps[cand] <= left + 1e-9]
                if not len(cand):
                    break
                j = int(rng.choice(cand))
                out[d, i, j] = True
                left -= eps[j]
                used.add(fam[j])
    return out


# ---------------------------------------------------------------- the regularised logistic

@dataclass
class Design:
    """logit P = a + b*logit(p0) + sum_g w_g f_g(asks); f from `kind`: 'fam' (log-LR summed per family) or 'bin' (one-hot cells)."""

    kind: str  # fam | hyp | bin
    families: list
    fam_of: np.ndarray  # per hypothesis: family column (fam) 
    col_of: list  # per hypothesis: array cell -> column (bin)
    n_cols: int
    use: np.ndarray  # hypotheses allowed as features (bool, H)

    @classmethod
    def make(cls, hyps: list[Hyp], kind: str, use: np.ndarray | None = None) -> Design:
        fams = sorted({h.family for h in hyps})
        fam_of = np.array([fams.index(h.family) for h in hyps])
        use = np.ones(len(hyps), dtype=bool) if use is None else np.asarray(use, dtype=bool)
        col_of, nxt = [], 0
        for h in hyps:
            col_of.append(np.arange(nxt, nxt + len(h.cells)))
            nxt += len(h.cells)
        if kind == "hyp":  # one weight per hypothesis; fam_of is then the hypothesis index
            fam_of = np.arange(len(hyps))
        return cls(kind, fams, fam_of, col_of, {"fam": len(fams), "hyp": len(hyps)}.get(kind, nxt), use)

    def matrix(self, cell: np.ndarray, x: np.ndarray, asks: np.ndarray) -> np.ndarray:
        """(n, n_cols) features of the asked hypotheses (asks bool (n, H)); unasked or unused = 0."""
        m = asks & (cell >= 0) & self.use[None, :]
        F = np.zeros((cell.shape[0], self.n_cols))
        for j in np.flatnonzero(m.any(0)):
            r = np.flatnonzero(m[:, j])
            if self.kind in ("fam", "hyp"):
                F[r, self.fam_of[j]] += x[r, j]
            else:
                F[r, self.col_of[j][cell[r, j]]] = 1.0
        return F


@dataclass
class Logit:
    """Penalised logistic with an unpenalised intercept a, ridge on (b - b0) and (w - w0); optional L1 on w (prox via saga)."""

    design: Design
    l2: float = 10.0  # pseudo-observations on w
    l2_b: float = 10.0
    prior_w: float = 1.0  # 1 for 'fam' log-LR (naive Bayes), 0 for 'bin' / L1
    l1: float = 0.0  # elastic-net: total strength in pseudo-observations, ratio below
    l1_ratio: float = 0.5
    a: float = 0.0
    b: float = 1.0
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inter: float = 0.0  # evidence slope on (z0 - zc): w_eff = w * (1 + inter * (z0 - zc))
    zc: float = 0.0

    def fit(self, p0, F, y, sw=None):
        from scipy.optimize import minimize

        z0 = logit(np.asarray(p0, dtype="f8"))
        n_c = F.shape[1]
        y = np.asarray(y, dtype="f8")
        sw = np.ones(len(y)) if sw is None else sw
        n = sw.sum()
        pr = np.r_[0.0, 1.0, np.broadcast_to(np.asarray(self.prior_w, dtype='f8'), (n_c,))]
        pen = np.r_[0.0, self.l2_b, np.broadcast_to(np.asarray(self.l2, dtype='f8'), (n_c,))]
        D = np.column_stack([np.ones(len(y)), z0, F])
        if self.l1 > 0:
            return self._fit_enet(D, y, sw, pr)

        def loss(th):
            z = D @ th
            r = th - pr
            g = D.T @ (sw * (sigmoid(z) - y)) / n + pen * r / n
            return float((sw * (np.logaddexp(0.0, z) - y * z)).sum() / n + 0.5 * (pen * r * r).sum() / n), g

        res = minimize(loss, pr.copy(), jac=True, method="L-BFGS-B")
        self.a, self.b, self.w = float(res.x[0]), float(res.x[1]), res.x[2:].copy()
        return self

    def _fit_enet(self, D, y, sw, pr):
        from sklearn.linear_model import LogisticRegression

        scale = 30.0  # z0 nearly unpenalised
        X = np.column_stack([D[:, 1] * scale, D[:, 2:]])
        C = 1.0 / max(self.l1, 1e-9)  # sklearn: sum loss + 1/C * penalty; our loss is a sum over n rows
        m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=self.l1_ratio, C=C, max_iter=200, tol=1e-4)
        m.fit(X, y.astype(int), sample_weight=sw)
        self.a = float(m.intercept_[0])
        self.b = float(m.coef_[0, 0] * scale)
        self.w = m.coef_[0, 1:].copy()
        return self

    def logit(self, p0, F):
        z0 = logit(np.asarray(p0, dtype="f8"))
        ev = F @ self.w
        if self.inter:
            ev = ev * (1.0 + self.inter * (z0 - self.zc))
        return self.a + self.b * z0 + ev

    def predict(self, p0, F):
        return sigmoid(self.logit(p0, F))


# ---------------------------------------------------------------- cross-fitted train study

def lr_from(split: Split, hyps: list[Hyp], rows: np.ndarray) -> list[np.ndarray]:
    """Per-hypothesis cell LRs estimated on the alerts `rows` (bool or index) of `split`."""
    y = split.y > 0.5
    out = []
    for j, h in enumerate(hyps):
        c = split.cell[:, j]
        m = np.zeros(len(c), dtype=bool)
        m[rows] = True
        m &= c >= 0
        nc = len(h.cells)
        out.append(cell_lr(np.bincount(c[m & ~y], minlength=nc).astype("f8"), np.bincount(c[m & y], minlength=nc).astype("f8"),
                           ceiling_of(h)))
    return out


def subsplit(split: Split, idx) -> Split:
    return Split([split.alert_ids[i] for i in idx], split.p0[idx], split.p0_v1[idx], split.y[idx], split.part[idx],
                 split.cell[idx], split.t[idx] if len(split.t) else split.t,
                 split.cluster[idx] if len(split.cluster) else split.cluster)  # fmt: skip


@dataclass
class Fold:
    train: Split
    test: Split
    x_train: np.ndarray  # inner out-of-fold log LRs of the training alerts
    x_test: np.ndarray  # test log LRs from the training alerts' LRs
    asks_train: np.ndarray  # (draws, n, H)
    asks_test: dict  # regime name -> (n, H)


REGIMES = {"e2": (2.0,), "e1": (1.0,), "e05": (0.5,), "mix": (0.25, 0.5, 1.0, 2.0)}


def make_folds(split: Split, hyps: list[Hyp], k: int = 5, draws: int = 3, train_menu=REGIMES["mix"], seed: int = 0,
               regimes=REGIMES) -> list[Fold]:  # fmt: skip
    blk = time_blocks(len(split.y), k)
    rng = np.random.default_rng(seed)
    folds = []
    for b in range(k):
        trn, te = np.flatnonzero(blk != b), np.flatnonzero(blk == b)
        sub, tsub = subsplit(split, trn), subsplit(split, te)
        xin = oof_log_lr(sub, hyps, k=4)
        xte = log_lr_matrix(tsub, hyps, lr_from(sub, hyps, np.arange(len(trn))))
        asks_tr = draw_asks(sub.cell, hyps, rng, train_menu, draws=draws)
        asks_te = {r: draw_asks(tsub.cell, hyps, rng, m)[0] for r, m in regimes.items()}
        folds.append(Fold(sub, tsub, xin, xte, asks_tr, asks_te))
    return folds, blk


def cv_predict(folds: list[Fold], blk: np.ndarray, make_model, design: Design, regime: str, n: int, dense: bool = False,
               aug=None) -> np.ndarray:  # fmt: skip
    """Pooled out-of-fold predictions for one model constructor `make_model(design)` and one test regime ('dense' = every
    answered hypothesis, the information ceiling's design)."""
    pred = np.zeros(n)
    for b, f in enumerate(folds):
        if dense:
            at, ae = (f.train.cell >= 0)[None], f.test.cell >= 0
        else:
            at, ae = f.asks_train, f.asks_test[regime]
        F = np.vstack([design.matrix(f.train.cell, f.x_train, a) for a in at])
        p0t = np.tile(f.train.p0, len(at))
        Fe = design.matrix(f.test.cell, f.x_test, ae)
        if aug is not None:  # extra columns that depend on p0 (interaction)
            F, Fe = aug(F, p0t), aug(Fe, f.test.p0)
        m = make_model(design).fit(p0t, F, np.tile(f.train.y, len(at)))
        pred[blk == b] = m.predict(f.test.p0, Fe)
    return pred


# ---------------------------------------------------------------- loading

TRAIN_NS = "data/backtest/hi_medium/7613b89b5dcc"  # K=4, history ledgers, 440 hypotheses on the train alerts
VAL_NS = "data/backtest/hi_medium/f8b676b7d1fc"  # the same 440 hypotheses on every askable val alert (scripts/backtest.py --split val)


def load_all(alerts_dir: str = "data/alerts/hi_medium_v2", train_ns: str = TRAIN_NS, val_ns: str = VAL_NS):
    """(hyps, train Split, val Split): the askable alerts of each split (those with rows), val cells re-binned with the train
    table's bins.  Val `eval` rows are loaded too (the caller must not score them before the freeze)."""
    import polars as pl

    hyps = load_table(f"{train_ns}/lr_table.jsonl")
    out = []
    for name, ns in (("train", train_ns), ("val", val_ns)):
        al = pl.read_parquet(f"{alerts_dir}/{name}.parquet")
        if "val_part" not in al.columns or name == "train":
            al = al.with_columns(pl.lit("train").alias("val_part"))
        rows = pl.read_parquet(f"{ns}/alert_rows.parquet")
        al = al.filter(pl.col("alert_id").is_in(rows["alert_id"].unique().to_list()))
        out.append(build_split(rows, hyps, al))
    return hyps, out[0], out[1]


# ---------------------------------------------------------------- offline policy simulation on stored answers

@dataclass
class Scorer:
    """The fused posterior as a function of the asked evidence: z = base(p0) + shrink * sum_h w[fam(h)] x[i, h]."""

    a: float
    b: float
    w_h: np.ndarray  # per-hypothesis weight (family weight of its family)
    shrink: float
    clip: float = 0.1  # logit(p0) taken at p0 clipped to [clip, 1 - clip] (ConditionalFusion.p0_clip)

    def base(self, p0) -> np.ndarray:
        return self.a + self.b * logit(np.clip(np.asarray(p0, dtype="f8"), self.clip, 1 - self.clip))

    def z(self, p0, x: np.ndarray, asks: np.ndarray) -> np.ndarray:
        return self.base(p0) + self.shrink * ((x * asks) @ self.w_h)


def pad_cells(hyps: list[Hyp], alpha: float = 0.5):
    """(logLR, P(cell|launder), P(cell|clean), valid) as (H, C) arrays for the vectorised EIG; probabilities are those measured
    on the noisy train answers (smoothed by alpha pseudo-counts' worth of mass: tiny, only to avoid zeros)."""
    C = max(len(h.cells) for h in hyps)
    L, P1, P0, V = (np.zeros((len(hyps), C)) for _ in range(4))
    for j, h in enumerate(hyps):
        k = len(h.cells)
        L[j, :k] = np.log(h.lr)
        P1[j, :k], P0[j, :k], V[j, :k] = h.p_launder, h.p_clean, 1.0
    return L, P1, P0, V


def run_policy(policy: str, cell: np.ndarray, x: np.ndarray, p0: np.ndarray, hyps: list[Hyp], sc: Scorer, eps_menu,
               budget: float = 2.0, y: np.ndarray | None = None, rng: np.random.Generator | None = None,
               order: Sequence[int] | None = None, one_per_family: bool = True, max_steps: int = 12,
               stop: tuple[float, float] | None = None) -> np.ndarray:
    """Asks chosen by a policy, as a bool mask (n, H).  policy: 'eig' (label-free, expected Brier reduction per unit eps under
    the fitted scorer, stop when no question is expected to help), 'random', 'fixed' (a global `order` of hypothesis indices,
    first affordable unused-family one), 'oracle_greedy' (LABEL-AWARE: adds the affordable hypothesis whose answer moves the
    posterior most toward the label, while it lowers the squared error; an unattainable bound that selects on noise).
    `stop` = (logit threshold, max |log-odds shift| per unit eps): the cannot_cross rule, an alert stops once its distance to the
    threshold exceeds eps_left * that bound (ftm.decision.threshold)."""
    n, H = cell.shape
    eps = np.array([h.eps for h in hyps])
    fam = np.array([h.family for h in hyps])
    fam_ix = np.unique(fam, return_inverse=True)[1]
    ok0 = (cell >= 0) & np.isin(eps, list(eps_menu))[None, :]
    asks = np.zeros((n, H), dtype=bool)
    left = np.full(n, budget)
    used = np.zeros((n, fam_ix.max() + 1), dtype=bool)
    z = sc.base(p0)
    L, P1, P0, V = pad_cells(hyps)
    wl = sc.shrink * sc.w_h[:, None] * L  # (H, C): the log-odds shift of each cell
    rng = rng or np.random.default_rng(0)
    for _ in range(max_steps):
        ok = ok0 & ~asks & (eps[None, :] <= left[:, None] + 1e-9)
        if one_per_family:
            ok &= ~used[:, fam_ix]
        if stop is not None:
            ok &= (np.abs(z - stop[0]) <= left * stop[1])[:, None]
        if not ok.any():
            break
        if policy == "eig":
            p = sigmoid(z)
            pc = p[:, None, None] * P1[None] + (1 - p)[:, None, None] * P0[None]  # (n, H, C)
            q = sigmoid(z[:, None, None] + wl[None])
            after = (pc * q * (1 - q) * V[None]).sum(2)
            gain = p[:, None] * (1 - p)[:, None] - after
            score = np.where(ok, gain / eps[None, :], -np.inf)
            pick = score.argmax(1)
            go = ok.any(1) & (score[np.arange(n), pick] * eps[pick] > 1e-9)
        elif policy == "random":
            r = np.where(ok, rng.random((n, H)), -1.0)
            pick = r.argmax(1)
            go = ok.any(1)
        elif policy == "fixed":
            rank = np.full(H, H)
            for r_, j in enumerate(order):
                rank[j] = r_
            score = np.where(ok, -rank[None, :].astype("f8"), -np.inf)
            pick = score.argmax(1)
            go = ok.any(1)
        elif policy == "oracle_greedy":
            zn = z[:, None] + sc.shrink * sc.w_h[None, :] * x
            err_new = (sigmoid(zn) - y[:, None]) ** 2
            err_now = ((sigmoid(z) - y) ** 2)[:, None]
            score = np.where(ok, err_now - err_new, -np.inf)
            pick = score.argmax(1)
            go = ok.any(1) & (score[np.arange(n), pick] > 1e-12)
        else:
            raise ValueError(policy)
        if not go.any():
            break
        i = np.flatnonzero(go)
        j = pick[i]
        asks[i, j] = True
        left[i] -= eps[j]
        used[i, fam_ix[j]] = True
        z[i] += sc.shrink * sc.w_h[j] * x[i, j]
    return asks


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"brier": float(np.mean((p - y) ** 2)), "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
           "mean_p": float(p.mean()), "var_p": float(p.var())}  # fmt: skip
    out["auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    z = logit(p)
    if z.std() > 1e-9:  # calibration slope: logistic of y on logit p
        from scipy.optimize import minimize

        def nll(t):
            u = t[0] + t[1] * z
            return float(np.sum(np.logaddexp(0, u) - y * u))

        out["cal_slope"] = float(minimize(nll, [0.0, 1.0]).x[1])
    return out


# ---------------------------------------------------------------- the frozen model

FUSION_VERSION = "headroom_v1"


def fit_train(tr: Split, hyps: list[Hyp], l2: float = 3.0, draws: int = 6, menu=REGIMES["mix"], seed: int = 0,
              dense: bool = False, eps_dense=None, use=None) -> tuple[Logit, Design, list]:  # fmt: skip
    """Per-family evidence weights (prior 1 = naive Bayes, ridge `l2`) fitted on ALL train alerts, jointly with p0.  Features are
    the out-of-fold LRs; the training asks are random-policy draws over `menu` (budget 2/alert), or every answered hypothesis
    (`dense`, restricted to the eps levels in `eps_dense`).  Returns (model, design, train LR arrays for val)."""
    des = Design.make(hyps, "fam", use)
    x = oof_log_lr(tr, hyps, k=5)
    if dense:
        eps = np.array([h.eps for h in hyps])
        ok = np.isin(eps, list(eps_dense or [h.eps for h in hyps]))[None, :]
        asks = [(tr.cell >= 0) & ok]
    else:
        cell = tr.cell if use is None else np.where(np.asarray(use)[None, :], tr.cell, -1)  # only usable questions are asked
        asks = list(draw_asks(cell, hyps, np.random.default_rng(seed), menu, draws=draws))
    F = np.vstack([des.matrix(tr.cell, x, a) for a in asks])
    m = Logit(des, l2=l2 if not dense else l2, prior_w=1.0 if not dense else 0.0).fit(
        np.tile(tr.p0, len(asks)), F, np.tile(tr.y, len(asks)))  # fmt: skip
    return m, des, lr_from(tr, hyps, np.arange(len(tr.y)))


def scorer_of(m: Logit, des: Design, hyps: list[Hyp], shrink: float, a: float | None = None, b: float | None = None) -> Scorer:
    w_h = np.array([m.w[des.fam_of[j]] if des.use[j] else 0.0 for j in range(len(hyps))])
    return Scorer(m.a if a is None else a, m.b if b is None else b, w_h, shrink)


def recal_ab(p0: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, ridge: float = 1.0, clip: float = 0.1,
             ) -> tuple[float, float]:  # fmt: skip
    """Intercept and slope of logit p0 (clipped to [clip, 1 - clip]) fitted on `y` (optionally with a fixed evidence offset),
    ridge pulling the slope to 1."""
    from scipy.optimize import minimize

    z = logit(np.clip(np.asarray(p0, dtype="f8"), clip, 1 - clip))
    off = np.zeros(len(z)) if offset is None else offset

    def f(t):
        u = t[0] + t[1] * z + off
        return float(np.sum(np.logaddexp(0, u) - y * u) + 0.5 * ridge * (t[1] - 1) ** 2)

    r = minimize(f, [0.0, 1.0])
    return float(r.x[0]), float(r.x[1])
