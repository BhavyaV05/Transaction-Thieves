"""Group B: DP-FedAvg over the 30 bank views (Flower DP helpers + Opacus), train split only, val for evaluation.

Owner: T1 (Abhishek).  Spec: tracks.md 3.7, plan.md 6.3, ftm/baselines/README.md (federated section).

Two privacy units, never mixed on one curve:

  unit="client"  (McMahan et al. 2018, arXiv:1710.06963).  One privacy unit = one bank's whole dataset. Each round every
                 client trains locally without any per-example mechanism, its update is clipped to norm S, the K updates are
                 averaged with equal weights and Gaussian noise of std sigma*S/K is added (Flower's `compute_clip_model_update`,
                 `compute_stdv`, `add_gaussian_noise_inplace`). All K clients take part in every round (no cohort sampling), so
                 the accountant is the Gaussian mechanism composed over R rounds (RDP, q = 1). Any model works; the features here
                 include past-only neighbourhood aggregates of the bank's own view ("ctx").
  unit="edge"    DP-SGD per client (Opacus `GradSampleModule` + `DPOptimizer`, own Poisson sampling, RDP accountant).
                 One privacy unit = one transaction. Valid ONLY because the per-example gradient depends on that transaction's own
                 attributes ("tx" features: amount, currencies, format, hour, same bank; no neighbour aggregates). A message-passing
                 GNN would not get this guarantee from per-sample clipping: one edge changes the neighbourhood, hence the loss and
                 gradient, of many other training examples, so the sensitivity of the summed clipped gradient is not bounded by C
                 (it would need degree-bounded sampling / per-node clipping). A transaction that sits in two bank views (both
                 banks are in the top 30) is in two clients' datasets, so the reported epsilon composes the two (RDP x2).

Features are scaled with fixed public constants (never data-dependent statistics, which would leak). Server weights in the edge arm
are the clients' train sizes, treated as public. Hyper-parameters are fixed, not tuned on private data.
Heavy imports (torch, opacus, flwr) are inside functions so CI without them still imports this module.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

DELTA = 1e-5
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]
CATS = ("receiving_currency", "payment_currency", "payment_format")
CTX = ["src_prior_out", "dst_prior_in", "src_prior_in", "dst_prior_out", "src_out_n", "src_out_amt", "src_in_n", "src_in_amt",
       "dst_out_n", "dst_out_amt", "dst_in_n", "dst_in_amt", "src_passthru", "dst_passthru"]  # fmt: skip
HIDDEN = 16


# ---------------------------------------------------------------- accounting
def rdp_epsilon(sigma: float, q: float, steps: int, copies: int = 1, delta: float = DELTA) -> float:
    """(epsilon, delta)-DP of `steps` Gaussian-mechanism steps with noise multiplier `sigma` at sampling rate `q` (RDP accountant,
    Opacus analysis). `copies` composes the same mechanism on that many datasets sharing the protected record (RDP is additive)."""
    from opacus.accountants.analysis import rdp as ra

    if sigma <= 0:
        return math.inf
    r = copies * ra.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=ORDERS)
    return float(ra.get_privacy_spent(orders=ORDERS, rdp=r, delta=delta)[0])


def noise_for_epsilon(eps: float, q: float, steps: int, copies: int = 1, delta: float = DELTA) -> float:
    """Smallest noise multiplier (bisection; epsilon is decreasing in sigma) whose epsilon is <= `eps`."""
    lo, hi = 0.2, 2.0
    while rdp_epsilon(hi, q, steps, copies, delta) > eps:
        hi *= 2
        if hi > 1e5:
            raise ValueError(f"no noise multiplier reaches epsilon={eps}")
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rdp_epsilon(mid, q, steps, copies, delta) > eps else (lo, mid)
    return hi


# ---------------------------------------------------------------- data
def category_codes(frames: list[pl.DataFrame]) -> dict[str, dict[str, int]]:
    """Public vocabulary of the three categoricals (currency and format names carry no private information)."""
    return {c: {v: i for i, v in enumerate(sorted(set().union(*[set(f[c].unique().to_list()) for f in frames])))} for c in CATS}


def tx_features(frame: pl.DataFrame, codes: dict) -> np.ndarray:
    """Features of the transaction ITSELF (float32): log amount, log paid/received ratio, same bank, hour (sin, cos), one-hot
    receiving currency, payment currency and format. Fixed public scaling."""
    f = frame
    num = np.column_stack([
        np.log1p(f["amount_received"].clip(lower_bound=0).to_numpy()) / 10.0,
        np.clip(np.log((f["amount_paid"].to_numpy() + 1e-9) / (f["amount_received"].to_numpy() + 1e-9)), -3, 3) / 3.0,
        (f["from_bank"] == f["to_bank"]).cast(pl.Float32).to_numpy(),
        np.sin(2 * np.pi * f["timestamp"].dt.hour().to_numpy() / 24.0),
        np.cos(2 * np.pi * f["timestamp"].dt.hour().to_numpy() / 24.0),
    ])  # fmt: skip
    hot = []
    for c in CATS:
        idx = f[c].replace_strict(codes[c], default=-1, return_dtype=pl.Int64).to_numpy()
        oh = np.zeros((len(f), len(codes[c])), "f4")
        oh[np.arange(len(f))[idx >= 0], idx[idx >= 0]] = 1.0
        hot.append(oh)
    return np.column_stack([num, *hot]).astype("f4")


def ctx_features(frame: pl.DataFrame, codes: dict) -> np.ndarray:
    """`tx_features` plus past-only neighbourhood aggregates of the bank's own view (counts and 24 h flows of the two endpoints,
    `ftm.retrieval.edge_scorer.timeline_features`), log1p / 10. Rows are returned in the order of `frame`."""
    from ftm.retrieval.edge_scorer import timeline_features

    feats, _ = timeline_features(frame.select("tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received",
                                              "amount_paid", "receiving_currency", "payment_currency", "payment_format"))  # fmt: skip
    ctx = frame.select("tx_id").join(feats.select("tx_id", *CTX), on="tx_id", how="left")
    c = np.sign(ctx.select(CTX).to_numpy()) * np.log1p(np.abs(ctx.select(CTX).to_numpy())) / 10.0
    return np.column_stack([tx_features(frame, codes), c]).astype("f4")


def load_clients(ds: str, feature_set: str, banks: list[int] | None = None, min_train_pos: int = 5, log=print, final: bool = False) -> tuple[list[dict], dict]:
    """One dict per usable bank view: X/y/tx_id of its train rows and its val rows (val_part fit/cal/eval). Reads through ftm.data.io
    (train and val only). A bank with fewer than `min_train_pos` train positives is skipped (as in the GFP one-bank table).

    `final=True` (week 14, needs FTM_FINAL_RUN=1) also reads the TEST rows of every view and adds `Xte`, `yte`, `tx_te` to each client. The
    features of a row use earlier rows of the bank's view only and the category codes come from the train/val rows, so the train and val
    arrays are identical to those of the val run; the test rows never enter training."""
    from ftm.baselines.gfp_gbt import val_parts
    from ftm.data import io

    cfg = io.load_config(ds)
    banks_df = io.read_banks(cfg["views_dir"]).sort("rank")
    todo = banks if banks is not None else banks_df["bank"].to_list()
    v = io.scan_transactions(cfg["parquet_dir"], split="val", columns=["timestamp"]).select(
        pl.col("timestamp").min().alias("a"), pl.col("timestamp").max().alias("b")).collect()  # fmt: skip
    cut = (v["a"][0], v["b"][0])
    cols = ["tx_id", "timestamp", "from_bank", "to_bank", "src_id", "dst_id", "amount_received", "amount_paid", "receiving_currency",
            "payment_currency", "payment_format", "is_laundering", "split"]  # fmt: skip
    splits = ["train", "val", "test"] if final else ["train", "val"]
    frames = {b: io.scan_view(cfg["views_dir"], b, split=splits, columns=cols).collect().sort(["timestamp", "tx_id"]) for b in todo}
    codes = category_codes([f.filter(pl.col("split") != "test") for f in frames.values()])
    clients, skipped = [], {}
    for b, f in frames.items():
        tr = (f["split"] == "train").to_numpy()
        npos = int(f["is_laundering"].cast(pl.Int64).filter(pl.Series(tr)).sum())
        if npos < min_train_pos:
            skipped[b] = npos
            log(f"bank {b}: skipped, {npos} train positives")
            continue
        X = (ctx_features if feature_set == "ctx" else tx_features)(f, codes)
        y = f["is_laundering"].cast(pl.Float32).to_numpy()
        va = (f["split"] == "val").to_numpy()
        vp = np.full(f.height, "", dtype="U4")
        vp[va] = val_parts(f["timestamp"].filter(pl.Series(va)), *cut)
        c = {"bank": b, "Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va], "tx_va": f["tx_id"].to_numpy()[va], "part_va": vp[va]}
        if final:
            te = (f["split"] == "test").to_numpy()
            c |= {"Xte": X[te], "yte": y[te], "tx_te": f["tx_id"].to_numpy()[te]}
        clients.append(c)
    return clients, {"skipped": skipped, "n_features": clients[0]["Xtr"].shape[1], "cut": [str(cut[0]), str(cut[1])]}


# ---------------------------------------------------------------- model and one federated run
def make_model(d: int):
    import torch

    return torch.nn.Sequential(torch.nn.Linear(d, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 1))


def _get(model) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def _set(model, arrs) -> None:
    import torch

    with torch.no_grad():
        for p, a in zip(model.parameters(), arrs, strict=True):
            p.copy_(torch.as_tensor(a))


def local_train(model, X, y, steps: int, q: float, lr: float, pos_weight: float, rng: np.random.Generator, clip: float | None = None,
                sigma: float = 0.0) -> None:
    """`steps` SGD steps on Poisson batches (each row enters a batch independently with probability q).

    clip=None: plain SGD (client-level arm). clip=C, sigma>=0: Opacus DP-SGD (per-example clipping to C, Gaussian noise sigma*C on the
    summed gradient, divided by the expected batch size); sigma=0 with clip set clips without noise.
    """
    import torch

    n, expected = len(X), max(q * len(X), 1.0)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    if clip is not None:
        from opacus import GradSampleModule
        from opacus.optimizers import DPOptimizer

        gsm = GradSampleModule(model, loss_reduction="mean")
        opt = DPOptimizer(opt, noise_multiplier=sigma, max_grad_norm=clip, expected_batch_size=int(round(expected)), loss_reduction="mean")
        opt.generator = torch.Generator().manual_seed(int(rng.integers(2**31)))
        net = gsm
    else:
        net = model
    for _ in range(steps):
        idx = np.flatnonzero(rng.random(n) < q)
        opt.zero_grad()
        if len(idx) > 0:  # an empty Poisson batch (probability ~0 at >= 50 expected rows) is skipped
            xb, yb = Xt[idx], yt[idx]
            w = torch.where(yb == 1, pos_weight, 1.0)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(xb).squeeze(1), yb, reduction="none") * w).mean()
            loss.backward()
            opt.step()


def run_federated(clients: list[dict], unit: str, eps: float | None, *, rounds: int, local_steps: int, q: float = 0.01, lr: float = 0.5,
                  clip: float = 1.0, pos_weight: float = 20.0, seed: int = 0, log=print) -> dict:
    """One federated training run; returns the global model's raw P(laundering) for every client's val rows plus the accounting.

    unit "edge": eps -> noise multiplier sigma for DP-SGD (q, rounds*local_steps steps, 2 copies); aggregation = size-weighted FedAvg.
    unit "client": eps -> sigma for the Gaussian mechanism over `rounds` rounds (q = 1); aggregation = equal-weight mean of updates
    clipped to `clip` plus noise sigma*clip/K. eps=None is the non-private run (edge: no clipping; client: no clipping, no noise).
    """
    import torch
    from flwr.supercore.differential_privacy import (
        add_gaussian_noise_inplace,
        compute_clip_model_update,
        compute_stdv,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)  # Flower's add_gaussian_noise_inplace draws from numpy's global RNG: seeded since week 10 (the runs before it, client-level private arms, are not bit-reproducible)
    rng = np.random.default_rng(seed)
    K, d = len(clients), clients[0]["Xtr"].shape[1]
    steps_total = rounds * local_steps
    private = eps is not None
    if not private:
        sigma, spent = 0.0, math.inf
    elif unit == "edge":
        sigma = noise_for_epsilon(eps, q, steps_total, copies=2)
        spent = rdp_epsilon(sigma, q, steps_total, copies=2)
    else:
        sigma = noise_for_epsilon(eps, 1.0, rounds)
        spent = rdp_epsilon(sigma, 1.0, rounds)
    glob = make_model(d)
    w_glob = _get(glob)
    sizes = np.array([len(c["Xtr"]) for c in clients], "f8")
    for r in range(rounds):
        locals_ = []
        for c in clients:
            m = make_model(d)
            _set(m, w_glob)
            if unit == "edge":
                local_train(m, c["Xtr"], c["ytr"], local_steps, q, lr, pos_weight, rng, clip=clip if private else None, sigma=sigma)
            else:
                local_train(m, c["Xtr"], c["ytr"], local_steps, q, lr, pos_weight, rng)
            w = _get(m)
            if unit == "client" and private:
                compute_clip_model_update(w, w_glob, clip)  # w <- w_glob + clipped(w - w_glob)
            locals_.append(w)
        if unit == "edge":
            wt = sizes / sizes.sum()
            w_glob = [sum(wt[i] * locals_[i][j] for i in range(K)) for j in range(len(w_glob))]
        else:
            upd = [sum(locals_[i][j] - w_glob[j] for i in range(K)) / K for j in range(len(w_glob))]
            if private:
                add_gaussian_noise_inplace(upd, compute_stdv(sigma, clip, K))
            w_glob = [w_glob[j] + upd[j] for j in range(len(w_glob))]
        if (r + 1) % max(rounds // 5, 1) == 0:
            log(f"  round {r + 1}/{rounds} done")
    _set(glob, w_glob)
    with torch.no_grad():
        raw = [torch.sigmoid(glob(torch.as_tensor(c["Xva"])).squeeze(1)).numpy().astype("f8") for c in clients]
        raw_test = [torch.sigmoid(glob(torch.as_tensor(c["Xte"])).squeeze(1)).numpy().astype("f8") for c in clients] if "Xte" in clients[0] else None
    return {"raw": raw, "raw_test": raw_test, "sigma": sigma, "epsilon": spent, "delta": DELTA if private else None, "unit": unit, "rounds": rounds,
            "local_steps": local_steps, "q": q if unit == "edge" else 1.0, "clip": clip if private else None, "n_clients": K,
            "seed": seed}  # fmt: skip


def evaluate(clients: list[dict], raw: list[np.ndarray], return_p0: bool = False) -> dict:
    """Brier / PR-AUC / F1 pooled over the clients' val `eval` rows. One Platt calibrator (2 parameters) is fit on the pooled val `fit`
    rows (post-processing with val labels, as in the other models' protocol); F1 uses the best-F1 threshold on `fit`."""
    from ftm.baselines import gfp_gbt as g

    part = np.concatenate([c["part_va"] for c in clients])
    y, p_raw = np.concatenate([c["yva"] for c in clients]), np.concatenate(raw)
    fit, ev = part == "fit", part == "eval"
    cal, _ = g.fit_calibrator(p_raw[fit], y[fit], has_signal=False)
    p = cal(p_raw)
    thr = g.metrics(y[fit], p[fit])["thr"]
    m = g.metrics(y[ev], p[ev], thr)
    const = float(y[fit].mean())
    brier_const = float(np.mean((const - y[ev]) ** 2))
    out = {k: m.get(k) for k in ("n", "n_pos", "brier", "pr_auc", "f1")} | {
        "brier_const": brier_const, "brier_skill": 1 - m["brier"] / brier_const}  # fmt: skip
    return out | {"p0": p} if return_p0 else out  # p0: calibrated probability of every val row, clients concatenated


def test_p0(clients: list[dict], raw_val: list[np.ndarray], raw_test: list[np.ndarray]) -> np.ndarray:
    """Final run: p0 of every test row (clients concatenated). The Platt calibrator is the one of `evaluate`: fit on the pooled val `fit` rows
    of the same model, then applied to the test raw scores; nothing is fit on test."""
    from ftm.baselines import gfp_gbt as g

    part = np.concatenate([c["part_va"] for c in clients])
    y, p_raw = np.concatenate([c["yva"] for c in clients]), np.concatenate(raw_val)
    fit = part == "fit"
    cal, _ = g.fit_calibrator(p_raw[fit], y[fit], has_signal=False)
    return cal(np.concatenate(raw_test))
