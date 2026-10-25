"""Group A secondary: MEGA-GNN (Bilgi, Chen, Atasu, arXiv:2412.00241), MEGA-GenAgg. Reproduce on its split, then our split.

Owner: T1 (Abhishek). See tracks.md section 3.5 and ftm/baselines/README.md.

The authors' code lives OUTSIDE the repo (Apache-2.0, https://github.com/hcagri/MEGA-GNN, **archived**: its last code
commit is ef60a71, the head only points to a TMLR successor repository, MIT, that is a different code base) in
~/Documents/t1_external/MEGA-GNN. Only this launcher and `mega_gnn_t1.patch` (apply with `git apply` inside the clone) are in the repo.
The two C++ extensions (`ports_cpp.cpp`, `negative_sampling.cpp`) are rebuilt for Python 3.12 with
`g++ -O3 -shared -std=c++17 -fPIC $(python3.12 -m pybind11 --includes) X.cpp -o X$(python3.12-config --extension-suffix)`;
`genagg` is used from its `genagg/` directory on PYTHONPATH (no pip install).

`repro_command` is the reproduction on the code's own split; `train_and_score` trains MEGA-GenAgg (GIN, GenAgg node aggregation, GIN
parallel-edge aggregation, edge updates, reverse message passing, ego ids, no port numbers) on OUR split exactly like
`multi_gnn.train_and_score` (train graph = train edges only, val scored on train+val, no test edge anywhere).
It must run in its own process: the module names of the two clones (`models`, `training`, ...) collide.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from ftm.baselines.multi_gnn import N_NEIGHS, edge_arrays  # noqa: F401  (same inputs and neighbour sampling as Multi-GNN)

MEGA_GNN_COMMIT = "ef60a71"  # last code commit of the archived hcagri/MEGA-GNN
# MEGA-GenAgg of arXiv:2412.00241 (`assets/commands.md` of the clone): GIN with GenAgg node aggregation, GIN parallel-edge aggregation.
ARGS = dict(model="gin", task="edge_class", emlps=True, reverse_mp=True, reverse_mp_lp=False, adamm_hetero=False, ports=False, ports_batch=False,
            tds=False, ego=True, edge_agg_type="gin", node_agg_type="genagg", flatten_edges=True, data="ours", testing=True, tqdm=False)  # fmt: skip


def repro_command(data_name: str = "Small_HI", seed: int = 0, batch_size: int = 2048, epochs: int = 80) -> list[str]:
    """Command line of the reproduction run on the code's own split (run it from the MEGA-GNN clone, PYTHONPATH=.:genagg).
    Seed 0, not 1: `main.py` replaces seed 1 by a random one."""
    return ["python3.12", "main.py", "--data", data_name, "--model", "gin", "--emlps", "--reverse_mp", "--ego", "--flatten_edges",
            "--edge_agg_type", "gin", "--node_agg_type", "genagg", "--n_epochs", str(epochs), "--save_model", "--task", "edge_class",
            "--testing", "--batch_size", str(batch_size), "--seed", str(seed)]  # fmt: skip


def launch(clone: Path, metrics_file: Path, budget_s: int, **kw) -> subprocess.Popen:
    """Start the reproduction in the background with the patch's time budget and per-epoch metrics file."""
    clone = Path(clone).expanduser()
    env = {**os.environ, "T1_BUDGET_S": str(budget_s), "T1_METRICS": str(metrics_file), "PYTHONPATH": f".:{clone / 'genagg'}",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # fmt: skip
    return subprocess.Popen(repro_command(**kw), cwd=clone, env=env)


def parallel_edge_groups(ei: np.ndarray) -> np.ndarray:
    """Id of the (src, dst) pair of every edge: edges with the same ordered pair are parallel (MEGA's `find_parallel_edges`, vectorised)."""
    key = ei[0].astype("i8") * (int(ei.max()) + 1) + ei[1]
    return np.unique(key, return_inverse=True)[1].reshape(-1).astype("i8")


def _hetero(ei, ea, y, n_nodes, mean=None, std=None):
    """Edge arrays -> z-norm (train-graph statistics) -> reverse-MP hetero object with the parallel-edge groups, as MEGA's `get_data`."""
    import torch
    from data_util import create_hetero_obj

    ea = torch.as_tensor(ea)
    if mean is None:
        mean, std = ea.mean(0, keepdim=True), ea.std(0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
    ns = SimpleNamespace(**ARGS)
    simp = torch.as_tensor(parallel_edge_groups(ei))
    x = torch.zeros(n_nodes, 1)  # z_norm of the all-ones placeholder feature
    return create_hetero_obj(x, torch.as_tensor(y), torch.as_tensor(ei), (ea - mean) / std, ea[:, 0].clone(), ns, simp), mean, std


def train_and_score(frame: pl.DataFrame, clone: Path, budget_s: float, max_epochs: int, fit_mask: np.ndarray,
                    seed: int = 1, batch_size: int = 2048, num_workers: int = 0, log=print, ckpt: Path | None = None, resume: bool = False) -> dict:  # fmt: skip
    """Train MEGA-GenAgg on the train rows of `frame` (time-sorted, split in {train, val}), pick the epoch by weighted AP on the val
    `fit` rows, return raw P(laundering) for every val row. Same contract, epoch rule, scoring (always `eval()` mode) and val
    look-ahead (a val edge is scored on the graph of train + the whole val split) as `multi_gnn.train_and_score`."""
    import json

    import torch
    from sklearn.metrics import average_precision_score
    from torch_geometric.loader import LinkNeighborLoader

    clone = Path(clone).expanduser().resolve()
    for p in (clone, clone / "genagg"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from train_util import AddEgoIds, add_arange_ids
    from training import get_model
    from util import set_seed

    set_seed(seed)
    ns = SimpleNamespace(**ARGS, batch_size=batch_size, num_neighs=list(N_NEIGHS), seed=seed, inference=False, finetune=False)
    cfg = SimpleNamespace(**json.loads((clone / "model_settings.json").read_text())["gin"]["params"])
    sp = frame["split"].to_numpy()
    ntr = int((sp == "train").sum())
    assert (sp[:ntr] == "train").all() and (sp[ntr:] == "val").all(), "frame must be time-sorted with all train rows first"
    ei, ea, y, txid, n_nodes, _ = edge_arrays(frame)
    t0 = time.time()
    tr, mean, std = _hetero(ei[:, :ntr], ea[:ntr], y[:ntr], n_nodes)
    va, _, _ = _hetero(ei, ea, y, n_nodes, mean, std)
    key = ("node", "to", "node")
    add_arange_ids([tr, va])  # column 0 of edge_attr (float32, stripped before the model); exact int64 ids go to `eid`
    for d in (tr, va):
        d[key].eid = torch.arange(d[key].edge_index.shape[1])
    log(f"graphs built in {time.time() - t0:.0f}s: train {ntr} edges, train+val {len(y)} edges, {n_nodes} nodes")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kw = dict(num_neighbors=ns.num_neighs, batch_size=batch_size, transform=AddEgoIds(), num_workers=num_workers, persistent_workers=num_workers > 0)

    def loader(data, pos, shuffle):
        pos = torch.as_tensor(pos)
        return LinkNeighborLoader(data, edge_label_index=(key, data[key].edge_index[:, pos]), edge_label=data[key].y[pos], shuffle=shuffle, **kw)

    def strip(batch):
        """drop the id column in place (the model never sees it)"""
        batch[key].edge_attr = batch[key].edge_attr[:, 1:]
        batch["node", "rev_to", "node"].edge_attr = batch["node", "rev_to", "node"].edge_attr[:, 1:]

    tr_loader = loader(tr, np.arange(ntr), True)
    sample = next(iter(tr_loader))
    strip(sample)
    model = get_model(sample, cfg, ns).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([cfg.w_ce1, cfg.w_ce2]).to(dev))

    def prep(batch, seeds):
        """mask of the batch's seed edges and their graph positions; strips the id column in place."""
        ids = batch[key].eid  # int64: float32 edge_attr ids would be exact only up to 16.7M edges (HI-Medium has 31.9M)
        mask = torch.isin(ids, seeds)
        strip(batch)
        return mask, ids

    @torch.no_grad()
    def score(pos):
        """P(laundering) for graph positions `pos` (val edges), NaN where a seed edge was not in its sampled subgraph."""
        model.eval()
        out = np.full(len(y), np.nan, dtype="f4")
        for batch in loader(va, pos, False):
            seeds = torch.as_tensor(pos)[batch[key].input_id.cpu()]
            mask, ids = prep(batch, seeds)
            batch = batch.to(dev)
            logits = model(batch)[mask.to(dev)]
            out[ids[mask].cpu().numpy()] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        return out

    rng = np.random.default_rng(seed)
    vpos = np.arange(ntr, len(y))
    fpos = np.flatnonzero(fit_mask)
    sel = fpos[(y[fpos] == 1) | (rng.random(len(fpos)) < 0.1)]
    sw = np.where(y[sel] == 1, 1.0, 10.0)

    def fit_ap():
        p = score(sel)[sel]
        ok = ~np.isnan(p)
        return float(average_precision_score(y[sel][ok], p[ok], sample_weight=sw[ok])) if y[sel][ok].sum() else 0.0, int((~ok).sum())

    hist, best = [], (-1.0, None)
    if resume and ckpt is not None and Path(ckpt).exists():  # continue from the best checkpoint (fresh Adam state)
        model.load_state_dict(torch.load(ckpt)["state"])
        ap0, miss0 = fit_ap()
        best = (ap0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        log(json.dumps({"resumed_from": str(ckpt), "fit_ap": ap0, "seed_edges_not_sampled": miss0}))
    t_start = time.time()
    for ep in range(max_epochs):
        model.train()
        tot, nb, stop = 0.0, 0, False
        for batch in tr_loader:
            mask, _ = prep(batch, batch[key].input_id)  # train seeds: graph position == input_id (all train edges are seeds)
            batch = batch.to(dev)
            opt.zero_grad()
            out = model(batch)[mask.to(dev)]
            loss = lossf(out, batch[key].y[mask.to(dev)])
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss), nb + 1
            if nb % 200 == 0 and time.time() - t_start > budget_s:
                stop = True
                break
        ap_e, miss_e = fit_ap()
        rec = {"epoch": ep, "batches": nb, "loss": tot / max(nb, 1), "fit_ap": ap_e,
               "seed_edges_not_sampled": miss_e, "elapsed_s": time.time() - t_start, "partial": stop}  # fmt: skip
        hist.append(rec)
        log(json.dumps(rec))
        if ap_e > best[0]:
            best = (ap_e, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if ckpt is not None:
                torch.save({"epoch": ep, "state": best[1]}, ckpt)
        if stop or time.time() - t_start > budget_s:
            break
    model.load_state_dict(best[1])
    raw = score(vpos)[vpos]
    return {"raw": raw, "tx_id": txid[vpos], "history": hist, "best_fit_ap": best[0],
            "n_not_scored": int(np.isnan(raw).sum()), "train_edges": ntr, "train_pos": int(y[:ntr].sum()),
            "seconds": time.time() - t0}  # fmt: skip
