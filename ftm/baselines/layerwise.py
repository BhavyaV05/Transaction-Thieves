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


class Sampler:
    """Neighbour-sampled mini-batches of the clients' extended subgraphs (the paper's training: seed edges, 2-hop sampling, up to `fanout`
    neighbours per node, here with the exchange kept exact).

    One step: every client has a batch of seed edges. A node whose representation after `k` layers is needed (`rem` = k) draws up to `fanout`
    of its incident edges (either direction: reverse message passing) once, its neighbours then need `k - 1` layers. Only nodes the client owns
    are expanded; a remote node's own aggregate is overwritten by the exchange, so it is only a request: the owner must hold that node
    with the same `rem`, which expands it in the owner's graph. The subgraph of a client is the union of the drawn edges. With
    `fanout=None` every incident edge is taken and the seed logits equal the full-batch ones (Theorems 1 and 2 still hold; tested). With a finite
    fan-out the receptive fields differ a little between clients, which is the gap the paper reports for its own sampled runs.
    Works on any objects with `nodes, src, dst, x_edge, owned, owner` (ClientGraph or the test stand-ins)."""

    def __init__(self, graphs, fanout: int | None = 100, hops: int = LAYERS, x_device=None):
        self.g, self.fanout, self.hops = graphs, fanout, hops
        self.ptr, self.inc, self.oloc, self.x = [], [], [], []
        for g in graphs:
            n, m = len(g.nodes), len(g.src)
            key = np.concatenate([g.src, g.dst])
            order = np.argsort(key, kind="stable")
            self.inc.append(np.concatenate([np.arange(m), np.arange(m)])[order])
            self.ptr.append(np.concatenate([[0], np.cumsum(np.bincount(key, minlength=n))]))
            loc = np.full(n, -1, np.int64)  # position of the node in its owner's node table
            for o in np.unique(g.owner[(g.owner >= 0) & ~g.owned]):
                pos = np.flatnonzero(g.owner == o)
                loc[pos] = np.searchsorted(graphs[o].nodes, g.nodes[pos])
            self.oloc.append(loc)
            if x_device is not None:
                import torch

                self.x.append(torch.as_tensor(g.x_edge, device=x_device))

    def _draw(self, c, v, level, rem, em, rng):
        g, ptr = self.g[c], self.ptr[c]
        d = ptr[v + 1] - ptr[v]
        big = np.zeros(len(v), bool) if self.fanout is None else d > self.fanout
        sm, ds = v[~big], d[~big]
        idx = [np.arange(ds.sum()) - np.repeat(np.cumsum(ds) - ds, ds) + np.repeat(ptr[sm], ds)]
        who = [np.repeat(sm, ds)]
        for u in v[big]:  # hubs: a uniform sample without replacement
            idx.append(ptr[u] + rng.choice(ptr[u + 1] - ptr[u], self.fanout, replace=False))
            who.append(np.full(self.fanout, u))
        e, who = self.inc[c][np.concatenate(idx)], np.concatenate(who)
        em[e] = True
        nb = g.src[e] + g.dst[e] - who  # the other endpoint
        rem[nb[rem[nb] < level - 1]] = level - 1

    def sample(self, seeds: list[np.ndarray], rng, exchange: bool):
        """seeds[c]: local edge indices of client c. Returns per-client subgraphs (numpy) and the exchange plan over their node tables."""
        G = self.g
        rem = [np.full(len(g.nodes), -1, np.int8) for g in G]
        drawn = [np.zeros(len(g.nodes), bool) for g in G]
        em = [np.zeros(len(g.src), bool) for g in G]
        for c, s in enumerate(seeds):
            em[c][s] = True
            rem[c][G[c].src[s]] = self.hops
            rem[c][G[c].dst[s]] = self.hops

        def request(level):  # remote copies ask their owner for the representation after `level` layers
            raised = False
            for c, g in enumerate(G):
                v = np.flatnonzero((rem[c] == level) & ~g.owned & (g.owner >= 0)) if exchange else np.zeros(0, np.int64)
                for o in np.unique(g.owner[v]):
                    t = self.oloc[c][v[g.owner[v] == o]]
                    t = t[rem[o][t] < level]
                    rem[o][t] = level
                    raised |= len(t) > 0
            return raised

        for level in range(self.hops, 0, -1):
            while True:
                raised = request(level)
                for c, g in enumerate(G):
                    v = np.flatnonzero((rem[c] == level) & g.owned & ~drawn[c])
                    if len(v):
                        self._draw(c, v, level, rem[c], em[c], rng)
                        drawn[c][v] = True
                        raised = True
                if not raised:
                    break
        request(0)
        subs = []
        for c, g in enumerate(G):
            nodes, edges = np.flatnonzero(rem[c] >= 0), np.flatnonzero(em[c])
            lut = np.full(len(g.nodes), -1, np.int64)
            lut[nodes] = np.arange(len(nodes))
            subs.append({"nodes": nodes, "edges": edges, "lut": lut, "src": lut[g.src[edges]], "dst": lut[g.dst[edges]], "owned": g.owned[nodes],
                         "owner": g.owner[nodes], "seed": np.searchsorted(edges, seeds[c])})  # fmt: skip
        off = np.cumsum([0] + [len(s["nodes"]) for s in subs])
        plan = []
        for c, s in enumerate(subs):
            pos, rows = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
            for o in np.unique(s["owner"][~s["owned"] & (s["owner"] >= 0)]) if exchange else []:
                p = np.flatnonzero(~s["owned"] & (s["owner"] == o))
                r = subs[o]["lut"][self.oloc[c][s["nodes"][p]]]
                assert (r >= 0).all(), "the owner does not hold a requested node"
                pos.append(p)
                rows.append(off[o] + r)
            plan.append({"pos": np.concatenate(pos), "rows": np.concatenate(rows), "unowned": np.flatnonzero(s["owner"] < 0),
                         "remote": np.flatnonzero(~s["owned"])})  # fmt: skip
        return subs, plan

    def to_device(self, subs, plan, device):
        import torch

        gt = []
        for c, s in enumerate(subs):
            e = torch.as_tensor(s["edges"], device=device)
            x = self.x[c][e] if self.x else torch.as_tensor(self.g[c].x_edge[s["edges"]], device=device)
            gt.append({"src": torch.as_tensor(s["src"], device=device), "dst": torch.as_tensor(s["dst"], device=device), "x": x,
                       "owned": torch.as_tensor(s["owned"], device=device)})  # fmt: skip
        pt = [{"pos": p["pos"], "pos_t": torch.as_tensor(p["pos"], device=device), "rows_t": torch.as_tensor(p["rows"], device=device),
               "unowned_t": torch.as_tensor(p["unowned"], device=device), "remote_t": torch.as_tensor(p["remote"], device=device)} for p in plan]  # fmt: skip
        return gt, pt, [torch.as_tensor(s["seed"], device=device) for s in subs]


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
        x_edge_test: dict | None = None, sampled: bool = False, fanout: int | None = 100, eval_batch: int = 8192) -> dict:  # fmt: skip
    """One training run. Returns per client the raw P(laundering) of its val rows at the step with the best pooled val `fit` AP.

    Final run (week 14): `fixed_step` (the step chosen on val, no selection and no training beyond it) and `frames_test` / `x_edge_test` (the same
    views with the TEST rows appended; features from `x_edge_test`): at that step the val rows are scored on the train+val graph as before and the
    test rows on the train+val+test graph with the same network; the result also holds `raw_test` and `test_idx_tx`.

    `sampled`: neighbour-sampled mini-batches (class `Sampler`, `fanout` neighbours per node and hop, `batch` seed edges per client and step) instead of
    full-batch message passing; scoring then also goes through sampled batches of `eval_batch` seed edges per client (a fixed sampling stream).

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
    if sampled:
        sm_tr, sm_ev = Sampler(g_tr, fanout, x_device=device), Sampler(g_ev, fanout)
        sm_te = Sampler(g_te, fanout) if g_te is not None else None
        rng_s = np.random.default_rng(seed + 1)  # sampling stream, apart from the seed-edge picks
    else:
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

    def sub_forward(sm, seeds, rng):
        subs, plan = sm.sample(seeds, rng, exchange != "none")
        gt, pt, pos = sm.to_device(subs, plan, device)
        return forward(nets, gt, pt, exchange, device, pos)

    def score_sampled(sm, idx):  # every listed edge of every client, in chunks of `eval_batch` seeds, the same sampling stream at every call
        rng_e, out = np.random.default_rng(seed + 7919), [[] for _ in banks]
        for k in range(max((len(i) + eval_batch - 1) // eval_batch for i in idx)):
            lo = sub_forward(sm, [i[k * eval_batch : (k + 1) * eval_batch] for i in idx], rng_e)
            for c, x in enumerate(lo):
                out[c].append(torch.sigmoid(x).cpu().numpy().astype("f8"))
        return [np.concatenate(o) for o in out]

    def evaluate(step):
        nonlocal best, raw_test
        for n in uniq:
            n.eval()
        with torch.no_grad():
            if sampled:
                raw = score_sampled(sm_ev, val_idx)
            else:
                raw = [torch.sigmoid(x).cpu().numpy().astype("f8") for x in forward(nets, gt_ev, pt_ev, exchange, device, val_t)]
        yv = np.concatenate([g_ev[c].y[val_idx[c]] for c in range(len(banks))])
        fit = part == "fit"
        ap = float(average_precision_score(yv[fit], np.concatenate(raw)[fit])) if yv[fit].sum() else 0.0
        hist.append({"step": step, "val_fit_ap": ap})
        log(f"  step {step}: val fit AP {ap:.4f}")
        if ap > best[0] or fixed_step is not None:
            best = (ap, raw, step)
            if g_te is not None:  # same network, test rows on the graph that also holds the test edges
                with torch.no_grad():
                    if sampled:
                        raw_test = score_sampled(sm_te, test_idx)
                    else:
                        lt = forward(nets, gt_te, pt_te, exchange, device, [torch.as_tensor(i, device=device) for i in test_idx])
                        raw_test = [torch.sigmoid(x).cpu().numpy().astype("f8") for x in lt]
        for n in uniq:
            n.train()

    for step in range(1, steps + 1):
        picks = [rng.choice(lab[c], size=min(batch, len(lab[c])), replace=False) if len(lab[c]) else lab[c] for c in range(len(banks))]
        pick_t = [torch.as_tensor(p, device=device) for p in picks]
        for o in opts:
            o.zero_grad()
        logits = sub_forward(sm_tr, picks, rng_s) if sampled else forward(nets, gt_tr, pt_tr, exchange, device, pick_t)
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
            "sampled": sampled, "fanout": fanout if sampled else None, "n_clients": len(banks), "val_rows": [len(i) for i in val_idx], "val_idx_tx": [g_ev[c].tx_id[val_idx[c]] for c in range(len(banks))]}  # fmt: skip
