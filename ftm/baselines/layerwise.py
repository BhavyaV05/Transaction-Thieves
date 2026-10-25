"""Group B, AML-specific federated method: layer-wise exchange of representations and adjoints (Ceydeli, Wang, Atasu).

Owner: T1 (Abhishek).  Spec: tracks.md 3.7 (fallback for Tian et al.), plan.md 6.3.
Paper: "Federated Layer-Wise Exchange for Cross-Client Money Laundering Pattern Detection", arXiv:2605.06433 (v2, ICAIF 2026
workshop).  No code is released; this is written from the paper (sections 3, 4 and Appendix C.2), nothing else was consulted.

What the paper specifies, and what is here:

  * Clients own disjoint node sets; client c holds its extended subgraph (its own edges plus the cross-client edges, with the remote
    endpoints present but without their features or labels). Here a client is one bank view, it owns the accounts of its bank, and
    a transaction is labelled and used in the objective of the client of its SENDING bank only (the paper's rule; incoming
    cross-bank transactions are graph context).
  * Forward exchange: before every layer the owner of a boundary node sends its current representation to every client that holds
    the node as a remote node. Without exchange the remote representations are 0 (the paper's definition of the local baseline).
  * Backward exchange: the adjoint of a remote copy is returned to the owner and added to the owner's adjoint before it
    back-propagates through the previous layer. In this single-process simulation the exchange is a gather of the owners' tensors, so
    autograd returns exactly the partial adjoints of Section 4.2 (Theorem 2); forward-only exchange gathers the owners' tensors
    detached (no adjoint returns).
  * Optimisation arms: Fully Local (no parameter aggregation), FedAvg (sample-weighted parameter average after a local epoch,
    here a fixed number of local steps), FedProx (mu = 0.01) and Sync-SGD (one shared parameter set, gradient of every client
    aggregated at every step); each with no exchange, forward exchange (Fwd-LE) or forward and backward exchange (Fwd/Bwd-LE).
  * Model: a 2-layer Multi-PNA edge classifier (Egressy et al. 2024: PNA aggregators mean/min/max/std with the three degree
    scalers, reverse message passing, edge-state updates), constant node features.

Not as in the paper (stated in docs/t1_baselines_val.md):
  * Full-batch message passing over each client's whole extended subgraph, with the loss on a random mini-batch of the client's labelled
    edges, instead of 2-hop neighbour sampling (no sampling gap: with Sync-SGD the exchange is exactly the centralised computation);
  * no ego ids and no port numbering (both are defined per sampled mini-batch);
  * LayerNorm instead of BatchNorm (batch statistics of a client mix owned and remote rows and cannot be averaged across clients);
  * clients are the 30 largest bank views, not a Louvain partition of all 30,470 banks; the accounts of the 30,440 other banks have no
    owner, so their representations are 0 in every mode (nothing can be exchanged for them);
  * 3 seeds, a fixed step budget, one set of hyper-parameters for every arm.
Heavy imports (torch, torch_geometric) are inside functions so CI without them still imports this module.
"""

from __future__ import annotations

import numpy as np
import polars as pl

EXCHANGE = ("none", "fwd", "bwd")  # no exchange | forward only | forward and backward
SYNC = ("local", "fedavg", "fedprox", "sync_sgd")
HIDDEN = 20  # n_hidden of IBM's Multi-PNA configuration (model_settings.json: pna)
LAYERS = 2
EDGE_FEATURES = 43  # federated.tx_features (42) + time since the first transaction


class ClientGraph:
    """One client's extended subgraph in local indices. Arrays only; `to(device)` makes the torch tensors."""

    def __init__(self, bank: int, frame: pl.DataFrame, node_bank: dict, x_edge: np.ndarray, client_of_bank: dict, train_only: bool):
        f = frame.filter(pl.col("split") == "train") if train_only else frame
        keep = (frame["split"] == "train").to_numpy() if train_only else np.ones(frame.height, bool)
        self.bank = bank
        src, dst = f["src_id"].to_numpy(), f["dst_id"].to_numpy()
        self.nodes, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
        self.src, self.dst = inv[: len(src)].astype(np.int64), inv[len(src) :].astype(np.int64)
        self.x_edge = x_edge[keep]
        self.y = f["is_laundering"].cast(pl.Float32).to_numpy()
        self.tx_id = f["tx_id"].to_numpy()
        self.split = f["split"].to_numpy()
        self.labelled = (f["from_bank"] == bank).to_numpy() & (self.split == "train")  # the sender's client trains on the transaction
        nb = np.array([node_bank[int(n)] for n in self.nodes], dtype=np.int64)
        self.owned = nb == bank
        self.owner = np.array([client_of_bank.get(int(b), -1) for b in nb], dtype=np.int64)  # client index of the owner, -1: none


def build_plan(graphs: list[ClientGraph]) -> list[dict]:
    """Per client: local positions of its remote nodes that have an owner, the row of the owner's copy in the concatenated node
    table, and the positions of remote nodes without an owner."""
    offsets = np.cumsum([0] + [len(g.nodes) for g in graphs])
    plan = []
    for c, g in enumerate(graphs):
        pos_owned, rows = [], []
        for o in np.unique(g.owner[(g.owner >= 0) & (g.owner != c)]):
            pos = np.flatnonzero(g.owner == o)
            loc = np.searchsorted(graphs[o].nodes, g.nodes[pos])  # nodes are sorted unique arrays
            pos_owned.append(pos)
            rows.append(offsets[o] + loc)
        plan.append({"pos": np.concatenate(pos_owned) if pos_owned else np.zeros(0, np.int64),
                     "rows": np.concatenate(rows) if rows else np.zeros(0, np.int64),
                     "unowned": np.flatnonzero(g.owner < 0), "remote": np.flatnonzero(~g.owned)})  # fmt: skip
    return plan


def degree_histograms(graphs: list[ClientGraph]) -> tuple[np.ndarray, np.ndarray]:
    """In- and out-degree histograms of the owned nodes over all clients' training graphs: the PNA degree scalers' constants."""
    din, dout = [], []
    for g in graphs:
        n = len(g.nodes)
        din.append(np.bincount(g.dst, minlength=n)[g.owned])
        dout.append(np.bincount(g.src, minlength=n)[g.owned])
    return tuple(np.bincount(np.concatenate(d)).astype(np.int64) for d in (din, dout))


def make_model(deg_in: np.ndarray, deg_out: np.ndarray, hidden: int = HIDDEN, layers: int = LAYERS):
    import torch
    from torch import nn
    from torch_geometric.nn import PNAConv

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            kw = {"in_channels": hidden, "out_channels": hidden, "aggregators": ["mean", "min", "max", "std"],
                  "scalers": ["identity", "amplification", "attenuation"], "edge_dim": hidden, "towers": 5, "pre_layers": 1,
                  "post_layers": 1, "divide_input": False}  # fmt: skip
            self.fwd = PNAConv(deg=torch.as_tensor(deg_in), **kw)
            self.rev = PNAConv(deg=torch.as_tensor(deg_out), **kw)  # reverse message passing
            self.norm = nn.LayerNorm(hidden)
            self.emlp = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        def nodes(self, h, e, src, dst):
            ei = torch.stack([src, dst])
            out = self.fwd(h, ei, e) + self.rev(h, ei.flip(0), e)
            return (h + torch.relu(self.norm(out))) / 2

        def edges(self, h, e, src, dst):  # needs the layer's new representation of BOTH endpoints, remote ones included
            return e + self.emlp(torch.cat([h[src], h[dst], e], -1)) / 2

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.node0 = nn.Parameter(torch.randn(hidden) / hidden**0.5)  # embedding of the constant node feature
            self.edge_emb = nn.Linear(EDGE_FEATURES, hidden)
            self.layers = nn.ModuleList([Layer() for _ in range(layers)])
            self.head = nn.Sequential(nn.Linear(3 * hidden, 50), nn.ReLU(), nn.Linear(50, 25), nn.ReLU(), nn.Linear(25, 1))

    return Net()


def _exchange(h_list, plan, mode, device):
    """Overwrite every client's remote rows: 0 (no exchange), the owner's tensor detached (Fwd-LE) or attached (Fwd/Bwd-LE)."""
    import torch

    table = torch.cat(h_list)
    table = table.detach() if mode == "fwd" else table
    out = []
    for h, p in zip(h_list, plan, strict=True):
        h = h.clone()
        if mode == "none":
            h[p["remote_t"]] = 0.0
        else:
            h[p["unowned_t"]] = 0.0
            if len(p["pos"]):
                h[p["pos_t"]] = table[p["rows_t"]]
        out.append(h)
    return out


def forward(nets, graphs_t, plan_t, mode: str, device, edge_idx=None):
    """All clients in lock-step. nets: one module per client (the same object for Sync-SGD). Returns per client the logits of the
    edges in `edge_idx[c]` (local edge indices) after `LAYERS` layers with the exchange `mode` before every layer and the read-out."""
    import torch
    from torch.utils.checkpoint import checkpoint

    hs = [nets[c].node0.expand(len(g["owned"]), -1).contiguous() for c, g in enumerate(graphs_t)]
    es = [nets[c].edge_emb(g["x"]) for c, g in enumerate(graphs_t)]
    ck = checkpoint if torch.is_grad_enabled() else None  # recompute in the backward pass: all clients' activations do not fit in 8 GB

    def run(fn, *args):
        return ck(fn, *args, use_reentrant=False) if ck else fn(*args)

    hs = _exchange(hs, plan_t, mode, device)  # layer 0 representations of the remote nodes
    for layer in range(LAYERS):
        hs = [run(nets[c].layers[layer].nodes, hs[c], es[c], g["src"], g["dst"]) for c, g in enumerate(graphs_t)]
        hs = _exchange(hs, plan_t, mode, device)  # the layer's output of every boundary node, before the edge update and the next layer
        es = [run(nets[c].layers[layer].edges, hs[c], es[c], g["src"], g["dst"]) for c, g in enumerate(graphs_t)]
    out = []
    for c, g in enumerate(graphs_t):
        idx = edge_idx[c] if edge_idx is not None else torch.arange(len(g["src"]), device=device)
        z = torch.cat([torch.relu(hs[c][g["src"][idx]]), torch.relu(hs[c][g["dst"][idx]]), es[c][idx]], -1)
        out.append(nets[c].head(z).squeeze(-1))
    return out


def to_device(graphs: list[ClientGraph], plan: list[dict], device):
    import torch

    gt = [{"src": torch.as_tensor(g.src, device=device), "dst": torch.as_tensor(g.dst, device=device),
           "x": torch.as_tensor(g.x_edge, device=device), "owned": torch.as_tensor(g.owned, device=device)} for g in graphs]  # fmt: skip
    pt = [{"pos": p["pos"], "pos_t": torch.as_tensor(p["pos"], device=device), "rows_t": torch.as_tensor(p["rows"], device=device),
           "unowned_t": torch.as_tensor(p["unowned"], device=device), "remote_t": torch.as_tensor(p["remote"], device=device)} for p in plan]  # fmt: skip
    return gt, pt


def run(frames: dict[int, pl.DataFrame], x_edge: dict[int, np.ndarray], *, sync: str, exchange: str, steps: int = 400, local_steps: int = 10,
        parts: dict[int, np.ndarray], batch: int = 8192, lr: float = 2e-3, pos_weight: float = 7.0, mu: float = 0.01, eval_every: int = 40,
        seed: int = 0, device: str = "cuda", log=print, fixed_step: int | None = None, frames_test: dict | None = None,
        x_edge_test: dict | None = None) -> dict:  # fmt: skip
    """One training run. Returns per client the raw P(laundering) of its val rows at the step with the best pooled val `fit` AP.

    Final run (week 14): `fixed_step` (the step chosen on val, no selection and no training beyond it) and `frames_test` / `x_edge_test` (the same
    views with the TEST rows appended; features from `x_edge_test`): at that step the val rows are scored on the train+val graph as before and the
    test rows on the train+val+test graph with the same network; the result also holds `raw_test` and `test_idx_tx`.

    `frames`: bank -> its view rows (train and val), sorted by (timestamp, tx_id), with split, is_laundering, from_bank ...;
    `x_edge`: bank -> edge features of those rows; `parts`: bank -> val_part ('fit'/'cal'/'eval') of its val rows, in row order. Training uses train rows only; scoring uses the train+val graph (as the paper)."""
    import copy

    import torch
    from sklearn.metrics import average_precision_score

    assert sync in SYNC and exchange in EXCHANGE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    banks = list(frames)
    client_of_bank = {b: i for i, b in enumerate(banks)}
    node_bank = {}
    for b in banks:
        f = frames[b]
        node_bank.update(zip(f["src_id"].to_list(), f["from_bank"].to_list(), strict=True))
        node_bank.update(zip(f["dst_id"].to_list(), f["to_bank"].to_list(), strict=True))
    g_tr = [ClientGraph(b, frames[b], node_bank, x_edge[b], client_of_bank, True) for b in banks]
    g_ev = [ClientGraph(b, frames[b], node_bank, x_edge[b], client_of_bank, False) for b in banks]
    g_te = None
    if frames_test is not None:  # train+val+test graph: only the test rows are scored on it
        g_te = [ClientGraph(b, frames_test[b], node_bank, x_edge_test[b], client_of_bank, False) for b in banks]
    plan_tr, plan_ev = build_plan(g_tr), build_plan(g_ev)
    gt_tr, pt_tr = to_device(g_tr, plan_tr, device)
    gt_ev, pt_ev = to_device(g_ev, plan_ev, device)
    gt_te, pt_te = to_device(g_te, build_plan(g_te), device) if g_te is not None else (None, None)
    deg_in, deg_out = degree_histograms(g_tr)
    base = make_model(deg_in, deg_out).to(device)
    shared = sync == "sync_sgd"
    nets = [base] * len(banks) if shared else [copy.deepcopy(base) for _ in banks]
    uniq = [base] if shared else nets
    opts = [torch.optim.Adam(n.parameters(), lr=lr) for n in uniq]
    lab = [np.flatnonzero(g.labelled) for g in g_tr]
    n_lab = np.array([len(x) for x in lab], "f8")
    wts = n_lab / n_lab.sum()
    ytr = [torch.as_tensor(g.y, device=device) for g in g_tr]
    val_idx = [np.flatnonzero(g.split == "val") for g in g_ev]
    val_t = [torch.as_tensor(i, device=device) for i in val_idx]
    test_idx = [np.flatnonzero(g.split == "test") for g in g_te] if g_te is not None else None
    raw_test = None
    part = np.concatenate([parts[b] for b in banks])
    best = (-1.0, None, -1)
    hist = []
    glob = {k: p.detach().clone() for k, p in base.named_parameters()}

    def evaluate(step):
        nonlocal best, raw_test
        for n in uniq:
            n.eval()
        with torch.no_grad():
            lo = forward(nets, gt_ev, pt_ev, exchange, device, val_t)
        raw = [torch.sigmoid(x).cpu().numpy().astype("f8") for x in lo]
        yv = np.concatenate([g_ev[c].y[val_idx[c]] for c in range(len(banks))])
        fit = part == "fit"
        ap = float(average_precision_score(yv[fit], np.concatenate(raw)[fit])) if yv[fit].sum() else 0.0
        hist.append({"step": step, "val_fit_ap": ap})
        log(f"  step {step}: val fit AP {ap:.4f}")
        if ap > best[0] or fixed_step is not None:
            best = (ap, raw, step)
            if g_te is not None:  # same network, test rows on the graph that also holds the test edges
                with torch.no_grad():
                    lt = forward(nets, gt_te, pt_te, exchange, device, [torch.as_tensor(i, device=device) for i in test_idx])
                raw_test = [torch.sigmoid(x).cpu().numpy().astype("f8") for x in lt]
        for n in uniq:
            n.train()

    for step in range(1, steps + 1):
        picks = [rng.choice(lab[c], size=min(batch, len(lab[c])), replace=False) if len(lab[c]) else lab[c] for c in range(len(banks))]
        pick_t = [torch.as_tensor(p, device=device) for p in picks]
        for o in opts:
            o.zero_grad()
        logits = forward(nets, gt_tr, pt_tr, exchange, device, pick_t)
        losses = []
        for c, (lg, p) in enumerate(zip(logits, pick_t, strict=True)):
            if len(p) == 0:
                continue
            y = ytr[c][p]
            w = torch.where(y == 1, pos_weight, 1.0)
            losses.append(((torch.nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).mean(), c))
        if shared:
            tot = sum(wts[c] * ls for ls, c in losses)  # the centralised objective: labelled edges weighted by client size
        else:
            tot = sum(ls for ls, _ in losses)  # each client's own objective (remote adjoints reach the owner through autograd)
            if sync == "fedprox":
                tot = tot + sum(mu / 2 * sum(((p - glob[k]) ** 2).sum() for k, p in nets[c].named_parameters()) for c in range(len(banks)))
        tot.backward()
        for o in opts:
            o.step()
        if sync in ("fedavg", "fedprox") and step % local_steps == 0:
            with torch.no_grad():
                pars = [dict(n.named_parameters()) for n in nets]
                glob = {k: sum(wts[c] * pars[c][k] for c in range(len(banks))) for k in glob}
                for c in range(len(banks)):
                    for k, p in pars[c].items():
                        p.copy_(glob[k])
        if fixed_step is not None:
            if step == fixed_step:
                evaluate(step)  # the step is fixed from val: no selection, no further training
                break
            continue
        if step % eval_every == 0 or step == steps:
            if sync in ("fedavg", "fedprox") and step % local_steps != 0:
                continue  # evaluate the aggregated model only
            evaluate(step)
    return {"raw": best[1], "raw_test": raw_test, "test_idx_tx": [g_te[c].tx_id[test_idx[c]] for c in range(len(banks))] if g_te is not None else None,
            "best_step": best[2], "best_val_fit_ap": best[0], "history": hist, "sync": sync, "exchange": exchange,
            "steps": steps, "local_steps": local_steps, "batch": batch, "lr": lr, "pos_weight": pos_weight, "seed": seed,
            "n_clients": len(banks), "val_rows": [len(i) for i in val_idx], "val_idx_tx": [g_ev[c].tx_id[val_idx[c]] for c in range(len(banks))]}  # fmt: skip
