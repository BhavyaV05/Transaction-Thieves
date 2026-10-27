"""C11 explainer comparisons for E8: personalised PageRank, GNNExplainer and PGExplainer as evidence selectors (week 10).

Owner: T1 (Abhishek).  Spec: tracks.md 3.8, plan.md E8.  Each selector scores the candidate edges of one alert (the home bank's
edges in [as_of - 72 h, as_of], `evidence_views.candidates`, nothing later) and `evidence_views` keeps the K = 40 best, always
including the alerted edge, exactly as for the other views. No selector reads a label.

* `ppr_edge_scores`: personalised PageRank on the candidate graph, **undirected, not time-respecting**, restarting at the two
  endpoints of the alerted edge (mass 1/2 each, restart probability 0.15); an edge scores PPR(sender) + PPR(receiver).
* `BankModel`: the home bank's **Multi-GNN one-bank model** (Multi-PNA+EU, the checkpoint `models/multi_gnn/<ds>_bank<b>.pt` of
  `scripts/train_gnn.py`) applied to the alert's candidate subgraph with the exact edge features of the p0 run (z-norm with the train
  graph's statistics, port numbers, ego ids on the alerted edge's two accounts). The model is explained for the class "laundering"
  (target 1 for every alert: no label is read) with `torch_geometric.explain`: `GNNExplainer` (one edge mask, applied to the forward and the
  reverse message-passing relation alike) and `PGExplainer` (a mask MLP on the last layer's node embeddings, trained on **train-split
  alerts only**; PyG's PGExplainer supports node- and graph-level tasks only, so it is run as `task_level="graph"` with the alerted edge as
  `index`, which loses the conditioning on the target edge except through the ego ids).
* The model is applied to the alerted edge's **2-hop neighbourhood in the bank's whole history up to `as_of`** (`BankModel.history_subgraph`:
  the `cap` = 100 most recent edges per node and hop, like the p0 run's [100, 100] neighbour sampler, but causal and without the random draw).
  There its P(laundering) equals the stored p0 raw score exactly whenever no node on the way has more than 100 edges (checked on real alerts
  by `scripts/run_explainers.py --check-p0`). The 72 h candidate window alone would not do: the model needs the history, and its score on the
  window graph correlates only 0.69 (Spearman, 16 alerts of bank 10) with p0. The edge masks are then read off the candidate edges: an
  edge outside the history subgraph (or the window) is never selected.

The Multi-GNN clone is outside the repo (`ftm/baselines/README.md`); `torch`, `torch_geometric` and the clone are imported inside the
functions, so the PPR part and the tests of the selection run in CI.
"""

from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

import numpy as np

CLONE = Path("~/Documents/t1_external/Multi-GNN")
EXPLAINERS = ("gnnexplainer", "pgexplainer")
PPR_ALPHA = 0.15  # restart probability
GNNEXPLAINER_EPOCHS = 100  # PyG's default
PG_EPOCHS = 30  # the epochs of PyG's PGExplainer example
PG_LR = 0.003
PG_TRAIN_ALERTS = 100  # train alerts per bank for PGExplainer (seeded draw; all of them when a bank has fewer)
EGO_AND_PORTS_DIM = 6  # edge features of the model: time, amount, currency, format, in-port, out-port
KEY = ("node", "to", "node")
REV = ("node", "rev_to", "node")


def ppr_edge_scores(src, dst, alert: int, alpha: float = PPR_ALPHA, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """Edge score = PPR(sender) + PPR(receiver) of personalised PageRank on the undirected multigraph of the candidate edges
    (parallel edges add weight, a self-loop counts once), restarting at the alerted edge's two end points with mass 1/2 each.
    Power iteration to L1 change < `tol`. Edges outside the alerted edge's connected component score exactly 0."""
    src, dst = np.asarray(src), np.asarray(dst)
    nodes, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    u, v, n = inv[: len(src)], inv[len(src) :], len(nodes)
    from scipy import sparse

    w = sparse.coo_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    w.setdiag(w.diagonal() / 2)  # a self-loop was added twice
    deg = np.asarray(w.sum(1)).ravel()
    p = sparse.diags(1.0 / np.maximum(deg, 1e-12)) @ w  # row-stochastic
    s = np.zeros(n)
    s[u[alert]] += 0.5
    s[v[alert]] += 0.5
    r = s.copy()
    for _ in range(max_iter):
        r2 = alpha * s + (1 - alpha) * (p.T @ r)
        done = np.abs(r2 - r).sum() < tol
        r = r2
        if done:
            break
    return r[u] + r[v]


def _ckpt(ds: str, bank: int) -> Path:
    return Path("models/multi_gnn") / f"{ds}_bank{bank}.pt"


def pg_path(ds: str, bank: int) -> Path:
    """Trained PGExplainer weights of a bank's model (gitignored)."""
    return Path("models/pgexplainer") / f"{ds}_bank{bank}.pt"


class BankModel:
    """One home bank's Multi-GNN one-bank model, its edge features, and the explainers; CPU or GPU.

    Features are rebuilt exactly as `multi_gnn.train_and_score` built them (`edge_arrays` on the bank's train+val frame sorted by
    (timestamp, tx_id), normalisation with the statistics of the train graph, ports of the train+val graph; the ports are causal: the
    port of an edge depends on earlier edges only) and cached in `data/retrieval/<ds>/explain_cache/bank<b>.npz`.
    """

    def __init__(self, ds: str, bank: int, device: str = "cpu", clone: Path = CLONE, cache: bool = True, final: bool = False):
        import json

        import torch
        from torch_geometric.nn import to_hetero

        from ftm.data import io

        self.ds, self.bank, self.torch, self.final = ds, bank, torch, final
        self.dev = torch.device(device)
        clone = Path(clone).expanduser().resolve()
        if str(clone) not in sys.path:
            sys.path.insert(0, str(clone))
        from training import get_model

        self._load_features(io, cache)
        self._index_incidence()
        ns = _args()
        cfg = _settings(clone, json)
        sample = _dummy_batch(torch)
        model = to_hetero(get_model(sample, cfg, ns), sample.metadata(), aggr="mean")
        model.load_state_dict(torch.load(_ckpt(ds, bank), map_location="cpu")["state"])
        self.model = _Wrap(model).to(self.dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _load_features(self, io, cache: bool) -> None:
        import polars as pl

        from ftm.baselines import multi_gnn

        f = Path("data/retrieval") / self.ds / "explain_cache" / f"bank{self.bank}{'_final' if self.final else ''}.npz"
        if cache and f.exists():
            z = np.load(f)
            self.tx, self.src, self.dst, self.ea, self.ea_rev, self.t, self.t0_us = (z[k] for k in ("tx", "src", "dst", "ea", "ea_rev", "t", "t0_us"))
            self.t0_us = int(self.t0_us)
            return
        cfg = io.load_config(self.ds)
        frame = io.scan_view(cfg["views_dir"], self.bank, split=["train", "val", "test"] if self.final else ["train", "val"], columns=multi_gnn.EDGE_COLS).collect().sort(["timestamp", "tx_id"])
        ntr = int((frame["split"] == "train").sum())
        ei, ea, y, txid, n_nodes, _ = multi_gnn.edge_arrays(frame)
        ns = _args()
        _, mean, std = multi_gnn._hetero(ei[:, :ntr], ea[:ntr], y[:ntr], n_nodes, ns)
        va, _, _ = multi_gnn._hetero(ei, ea, y, n_nodes, ns, mean, std)
        self.tx, self.src, self.dst, self.t = txid, ei[0], ei[1], ea[:, 0].astype("i8")  # seconds since the frame's first row
        self.t0_us = int(np.datetime64(frame["timestamp"].min(), "us").astype("i8"))
        self.ea, self.ea_rev = va[KEY].edge_attr.numpy().copy(), va[REV].edge_attr.numpy().copy()
        assert self.ea.shape[1] == EGO_AND_PORTS_DIM and pl is not None
        if cache:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, tx=self.tx, src=self.src, dst=self.dst, ea=self.ea, ea_rev=self.ea_rev, t=self.t, t0_us=self.t0_us)

    def _index_incidence(self) -> None:
        """CSR of the edges at every node, edges in time order within a node (self-loops once)."""
        n = int(max(self.src.max(), self.dst.max())) + 1
        e = np.arange(len(self.src))
        node, edge = np.concatenate([self.src, self.dst]), np.concatenate([e, e])
        keep = np.concatenate([np.ones(len(e), bool), self.src != self.dst])  # a self-loop is listed once
        node, edge = node[keep], edge[keep]
        o = np.lexsort((edge, node))
        self._inc_edge = edge[o]
        self._inc_ptr = np.searchsorted(node[o], np.arange(n + 1))

    def recent_edges(self, v: int, t_end: int, cap: int) -> np.ndarray:
        """Rows of the `cap` most recent edges at node `v` with time <= t_end."""
        lo, hi = self._inc_ptr[v], self._inc_ptr[v + 1]
        inc = self._inc_edge[lo:hi]
        return inc[: np.searchsorted(self.t[inc], t_end, side="right")][-cap:]

    def seconds(self, as_of) -> int:
        """`as_of` in the model's time unit (seconds since the bank frame's first row)."""
        return int((np.datetime64(as_of, "us").astype("i8") - self.t0_us) // 1_000_000)

    def history_subgraph(self, tx_id: int, as_of, cap: int = 100) -> dict:
        """The alerted edge's 2-hop neighbourhood in the bank's WHOLE history up to `as_of` (nothing later), with at most the `cap` most
        recent incident edges per node and hop: what the p0 run's neighbour sampler ([100, 100]) shows the model, minus the later edges
        and the random draw (equal to the p0 raw score whenever no node on the way has more than `cap` edges). The explainers see the
        context the prediction was made from; the evidence is then cut to the 72 h candidates. Returns {"x", "edge_index",
        "edge_attr", "edge_attr_rev" (tensors), "index" (the alerted edge), "pos" (rows of the bank frame of every edge)}."""
        torch = self.torch
        t_end = self.seconds(as_of)
        a = int(np.flatnonzero(self.tx == tx_id)[0])
        seeds = np.unique([self.src[a], self.dst[a]])
        h1 = np.concatenate([self.recent_edges(v, t_end, cap) for v in seeds])
        n1 = np.unique(np.concatenate([self.src[h1], self.dst[h1], seeds]))
        pos = np.unique(np.concatenate([h1, np.concatenate([self.recent_edges(v, t_end, cap) for v in n1]), [a]]))
        nodes, inv = np.unique(np.concatenate([self.src[pos], self.dst[pos]]), return_inverse=True)
        ei = np.stack([inv[: len(pos)], inv[len(pos) :]])
        x = np.zeros((len(nodes), 2), "f4")
        x[np.searchsorted(nodes, seeds), 1] = 1.0  # ego ids: the alerted edge's two accounts
        t = lambda arr: torch.as_tensor(arr).to(self.dev)  # noqa: E731
        return {"x": t(x), "edge_index": t(ei.astype("i8")), "edge_attr": t(self.ea[pos]), "edge_attr_rev": t(self.ea_rev[pos]),
                "index": int(np.flatnonzero(pos == a)[0]), "pos": pos}  # fmt: skip

    @staticmethod
    def _kw(sub: dict) -> dict:
        return {"edge_attr": sub["edge_attr"], "edge_attr_rev": sub["edge_attr_rev"]}

    def p_launder(self, sub: dict) -> float:
        """The model's P(laundering) for the alerted edge on this subgraph (uncalibrated, as the raw score of the p0 table)."""
        with self.torch.no_grad():
            out = self.model(sub["x"], sub["edge_index"], **self._kw(sub))[sub["index"]]
        return float(self.torch.softmax(out, -1)[1])

    def gnnexplainer(self, sub: dict, epochs: int = GNNEXPLAINER_EPOCHS, seed: int = 0) -> np.ndarray:
        """GNNExplainer edge mask (one value per edge of `sub`; 0 outside the gradient support) for the class 'laundering'."""
        from torch_geometric.explain import Explainer, GNNExplainer

        torch = self.torch
        torch.manual_seed(seed)
        ex = Explainer(model=self.model, algorithm=GNNExplainer(epochs=epochs), explanation_type="phenomenon", node_mask_type=None,
                       edge_mask_type="object", model_config=dict(mode="multiclass_classification", task_level="edge", return_type="raw"))  # fmt: skip
        e = sub["edge_index"].shape[1]
        with torch.enable_grad():
            mask = ex(sub["x"], sub["edge_index"], target=torch.ones(e, dtype=torch.long, device=self.dev), index=sub["index"], **self._kw(sub)).edge_mask
        return mask.detach().cpu().numpy().astype("f8")

    def _pg(self, epochs: int):
        from torch_geometric.explain import Explainer, PGExplainer

        return Explainer(model=self.model, algorithm=PGExplainer(epochs=epochs, lr=PG_LR).to(self.dev), explanation_type="phenomenon",
                         edge_mask_type="object", model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"))  # fmt: skip

    def pg_train(self, subs: list[dict], epochs: int = PG_EPOCHS, seed: int = 0, log=None) -> list[float]:
        """Train the PGExplainer mask MLP on the subgraphs of train alerts (target class 1); returns the mean loss per epoch."""
        torch = self.torch
        torch.manual_seed(seed)
        self.pg = self._pg(epochs)
        hist = []
        for epoch in range(epochs):
            tot = 0.0
            for sub in subs:
                e = sub["edge_index"].shape[1]
                tot += self.pg.algorithm.train(epoch, self.model, sub["x"], sub["edge_index"], target=torch.ones(e, dtype=torch.long, device=self.dev),
                                               index=sub["index"], **self._kw(sub))  # fmt: skip
            hist.append(tot / max(len(subs), 1))
            if log:
                log(f"bank {self.bank} PGExplainer epoch {epoch} loss {hist[-1]:.4f}")
        return hist

    def pg_save(self, path: Path, epochs: int = PG_EPOCHS) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"mlp": self.pg.algorithm.mlp.state_dict(), "epochs": epochs}, path)

    def pg_load(self, path: Path) -> None:
        z = self.torch.load(path, map_location=self.dev)
        self.pg = self._pg(z["epochs"])
        self.pg.algorithm.mlp.load_state_dict(z["mlp"])  # PyG's lazy Linear takes its in_channels from the weights
        self.pg.algorithm._curr_epoch = z["epochs"] - 1

    def pgexplainer(self, sub: dict) -> np.ndarray:
        """Trained PGExplainer edge mask for the class 'laundering'."""
        torch = self.torch
        e = sub["edge_index"].shape[1]
        with torch.no_grad():
            mask = self.pg(sub["x"], sub["edge_index"], target=torch.ones(e, dtype=torch.long, device=self.dev), index=sub["index"], **self._kw(sub)).edge_mask
        return mask.detach().cpu().numpy().astype("f8")

    def scores(self, method: str, cand_tx, tx_id: int, as_of, seed: int = 0, sub: dict | None = None) -> tuple[np.ndarray, float]:
        """Score of every candidate edge (`cand_tx`: tx_ids of the 72 h window) under `method`: the mask value of the edge in the alert's
        history subgraph, 0 for candidates outside it, and the seconds the explainer took."""
        sub = sub or self.history_subgraph(tx_id, as_of)
        t0 = time.time()
        m = self.gnnexplainer(sub, seed=seed) if method == "gnnexplainer" else self.pgexplainer(sub)
        sec = time.time() - t0
        mask = dict(zip(self.tx[sub["pos"]].tolist(), m.tolist(), strict=True))
        return np.array([mask.get(int(t), 0.0) for t in cand_tx]), sec


def _args():
    from types import SimpleNamespace

    from ftm.baselines import multi_gnn

    return SimpleNamespace(**multi_gnn.ARGS, batch_size=2048, num_neighs=list(multi_gnn.N_NEIGHS), seed=1)


def _settings(clone: Path, json):
    from types import SimpleNamespace

    return SimpleNamespace(**json.loads((clone / "model_settings.json").read_text())["pna"]["params"])


def _dummy_batch(torch):
    """A two-node hetero batch with the shapes the model is built from (the PNA degree statistics are loaded from the checkpoint)."""
    from torch_geometric.data import HeteroData

    d = HeteroData()
    d["node"].x = torch.zeros(2, 2)
    for k in (KEY, REV):
        d[k].edge_index = torch.tensor([[0], [1]])
        d[k].edge_attr = torch.zeros(1, EGO_AND_PORTS_DIM + 1)  # + the id column that the model never sees
    return d


def _wrap_cls():
    import torch

    class Wrap(torch.nn.Module):
        """Homogeneous interface for `torch_geometric.explain`: logits [E, 2] of the 'to' edges. One edge mask scales the messages of
        both relations (the reverse relation uses the same edges flipped)."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, edge_index, edge_attr, edge_attr_rev):
            return self.model({"node": x}, {KEY: edge_index, REV: edge_index.flip(0)}, {KEY: edge_attr, REV: edge_attr_rev})[KEY]

    return Wrap


def _Wrap(model):  # noqa: N802
    return _wrap_cls()(model)


@functools.lru_cache(maxsize=2)
def _bank_model(ds: str, bank: int, device: str, final: bool = False) -> BankModel:
    return BankModel(ds, bank, device, final=final)


def live_scores(ds: str, method: str, bank: int, tx_id: int, as_of, cand_tx, device: str = "cpu", final: bool = False) -> np.ndarray:
    """Explainer score of every candidate edge of one alert, computed now (the cache of `scripts/run_explainers.py` is bypassed)."""
    bm = _bank_model(ds, bank, device, final)
    if method == "pgexplainer" and not hasattr(bm, "pg"):
        bm.pg_load(pg_path(ds, bank))
    return bm.scores(method, cand_tx, tx_id, as_of, seed=tx_id % 2**31)[0]
