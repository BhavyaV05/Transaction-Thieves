"""Group A core: Multi-GNN (Egressy et al., AAAI 2024), official IBM code. Reproduce on its split, then our split, one-bank and see-everything.

Owner: T1 (Abhishek). See tracks.md section 3.

The IBM code lives OUTSIDE the repo (Apache-2.0, https://github.com/IBM/Multi-GNN, commit
252b0252afca109d1d216c411c59ff70753b25fc) in ~/Documents/t1_external/Multi-GNN. Only this thin launcher and
the compatibility patch `multi_gnn_t1.patch` (apply with `git apply` inside that clone) are in the repo.
`--repro` runs it on its own split; `train_and_score` below trains the same model (Multi-PNA+EU) on OUR split
(train graph = train edges only, val scored on a graph of train+val edges, no test edge anywhere) and returns
raw scores for the val edges. Calibration and the p0 table are in scripts/train_gnn.py.
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

MULTI_GNN_COMMIT = "252b0252afca109d1d216c411c59ff70753b25fc"
# Multi-PNA with edge updates (EU), reverse message passing, ego IDs and port numbering = the paper's best HI-Small model.
PNA_EU_ARGS = ["--model", "pna", "--emlps", "--reverse_mp", "--ego", "--ports"]


def repro_command(data_name: str = "Small_HI", seed: int = 1, batch_size: int = 2048, epochs: int = 100) -> list[str]:
    """Command line of the reproduction run (run it from the Multi-GNN clone)."""
    return ["python3.12", "main.py", "--data", data_name, *PNA_EU_ARGS, "--seed", str(seed), "--testing", "--save_model",
            "--unique_name", "--n_epochs", str(epochs), "--batch_size", str(batch_size)]  # fmt: skip


def launch(clone: Path, metrics_file: Path, budget_s: int, **kw) -> subprocess.Popen:
    """Start the run in the background with the patch's time budget and per-epoch metrics file."""
    env = {**os.environ, "T1_BUDGET_S": str(budget_s), "T1_METRICS": str(metrics_file),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # fmt: skip
    return subprocess.Popen(repro_command(**kw), cwd=clone, env=env)


EDGE_COLS = ["tx_id", "timestamp", "src_id", "dst_id", "amount_received", "receiving_currency", "payment_format", "split", "is_laundering"]  # fmt: skip
N_NEIGHS = (100, 100)
ARGS = dict(model="pna", emlps=True, reverse_mp=True, ego=True, ports=True, tds=False, data="ours", testing=True, tqdm=False)


def edge_arrays(frame: pl.DataFrame, codes: dict | None = None):
    """Multi-GNN edge features of a time-sorted frame of train/val rows: seconds since the first row, amount received,
    receiving currency code, payment format code (the four columns of its `format_kaggle_files.py`), plus dense node ids.

    Returns (edge_index [2,E] int64, edge_attr [E,4] float32, y [E] int64, tx_id, n_nodes, codes). `codes` from the train graph.
    """
    if (frame["split"] == "test").any():
        raise ValueError("Multi-GNN input must not contain test rows")
    if codes is None:
        codes = {c: {v: i for i, v in enumerate(sorted(frame[c].unique().to_list()))} for c in ("receiving_currency", "payment_format")}
    nodes = np.unique(np.concatenate([frame["src_id"].to_numpy(), frame["dst_id"].to_numpy()]))
    ei = np.stack([np.searchsorted(nodes, frame["src_id"].to_numpy()), np.searchsorted(nodes, frame["dst_id"].to_numpy())])
    ts = (frame["timestamp"].dt.epoch("s") - frame["timestamp"].dt.epoch("s").min()).to_numpy()
    cur = frame["receiving_currency"].replace_strict(codes["receiving_currency"], default=-1, return_dtype=pl.Int64).to_numpy()
    fmt = frame["payment_format"].replace_strict(codes["payment_format"], default=-1, return_dtype=pl.Int64).to_numpy()
    ea = np.column_stack([ts, frame["amount_received"].to_numpy(), cur, fmt]).astype("f4")
    return ei.astype("i8"), ea, frame["is_laundering"].cast(pl.Int64).to_numpy(), frame["tx_id"].to_numpy(), len(nodes), codes


def _hetero(ei, ea, y, n_nodes, ns, mean=None, std=None):
    """GraphData -> ports -> z-norm (train-graph statistics) -> reverse-MP hetero object, as in Multi-GNN's get_data."""
    import torch
    from data_util import GraphData, create_hetero_obj

    d = GraphData(x=torch.ones(n_nodes, 1), y=torch.as_tensor(y), edge_index=torch.as_tensor(ei), edge_attr=torch.as_tensor(ea),
                  timestamps=torch.as_tensor(ea[:, 0]).clone()).add_ports()
    if mean is None:
        mean, std = d.edge_attr.mean(0, keepdim=True), d.edge_attr.std(0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
    d.edge_attr = (d.edge_attr - mean) / std
    d.x = torch.zeros_like(d.x)  # z_norm of the all-ones placeholder feature
    return create_hetero_obj(d.x, d.y, d.edge_index, d.edge_attr, d.timestamps, ns), mean, std


def train_and_score(frame: pl.DataFrame, clone: Path, budget_s: float, max_epochs: int, fit_mask: np.ndarray,
                    seed: int = 1, batch_size: int = 2048, num_workers: int = 0, log=print, ckpt: Path | None = None) -> dict:  # fmt: skip
    """Train Multi-PNA+EU on the train rows of `frame` (time-sorted, split in {train, val}), pick the epoch by AP on
    the val `fit` rows, return raw P(laundering) for every val row.

    Train graph = train edges only. Val edges are scored on the graph of train+val edges (the paper's setup for
    validation), so a val edge's neighbourhood can contain later val edges: a look-ahead of the whole val split
    (reported in ftm/baselines/README.md). Epoch selection uses a weighted AP on all fit positives and 10% of the
    fit negatives. `fit_mask` is a bool array over `frame` rows (val `fit` rows). The model is trained in `train()`
    mode and scored in `eval()` mode (dropout off, running batch-norm statistics).
    """
    import json

    import torch
    from sklearn.metrics import average_precision_score
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.nn import to_hetero

    clone = Path(clone).expanduser().resolve()
    if str(clone) not in sys.path:
        sys.path.insert(0, str(clone))
    from train_util import AddEgoIds, add_arange_ids
    from training import get_model
    from util import set_seed

    set_seed(seed)
    ns = SimpleNamespace(**ARGS, batch_size=batch_size, num_neighs=list(N_NEIGHS), seed=seed)
    cfg = SimpleNamespace(**json.loads((clone / "model_settings.json").read_text())["pna"]["params"])
    sp = frame["split"].to_numpy()
    ntr = int((sp == "train").sum())
    assert (sp[:ntr] == "train").all() and (sp[ntr:] == "val").all(), "frame must be time-sorted with all train rows first"
    ei, ea, y, txid, n_nodes, _ = edge_arrays(frame)
    t0 = time.time()
    tr, mean, std = _hetero(ei[:, :ntr], ea[:ntr], y[:ntr], n_nodes, ns)
    va, _, _ = _hetero(ei, ea, y, n_nodes, ns, mean, std)
    add_arange_ids([tr, va])
    log(f"graphs built in {time.time() - t0:.0f}s: train {ntr} edges, train+val {len(y)} edges, {n_nodes} nodes")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kw = dict(num_neighbors=ns.num_neighs, batch_size=batch_size, transform=AddEgoIds(), num_workers=num_workers, persistent_workers=num_workers > 0)
    key = ("node", "to", "node")

    def loader(data, pos, shuffle):
        pos = torch.as_tensor(pos)
        return LinkNeighborLoader(data, edge_label_index=(key, data[key].edge_index[:, pos]), edge_label=data[key].y[pos], shuffle=shuffle, **kw)

    tr_loader = loader(tr, np.arange(ntr), True)
    model = to_hetero(get_model(next(iter(tr_loader)), cfg, ns), tr.metadata(), aggr="mean").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([cfg.w_ce1, cfg.w_ce2]).to(dev))

    def prep(batch, seeds):
        """mask of the batch's seed edges and their graph positions; strips the id column in place."""
        ids = batch[key].edge_attr[:, 0].long()
        mask = torch.isin(ids, seeds)
        batch[key].edge_attr = batch[key].edge_attr[:, 1:]
        batch["node", "rev_to", "node"].edge_attr = batch["node", "rev_to", "node"].edge_attr[:, 1:]
        return mask, ids

    @torch.no_grad()
    def score(pos, model_eval=True):
        """P(laundering) for graph positions `pos` (val edges), NaN where a seed edge was not in its sampled subgraph."""
        model.train(not model_eval)
        out = np.full(len(y), np.nan, dtype="f4")
        lo = loader(va, pos, False)
        for batch in lo:
            seeds = batch[key].input_id  # index into the label-index tensor, i.e. into `pos`
            seeds = torch.as_tensor(pos)[seeds.cpu()]
            mask, ids = prep(batch, seeds)
            batch = batch.to(dev)
            logits = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[key][mask.to(dev)]
            out[ids[mask].cpu().numpy()] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        return out

    rng = np.random.default_rng(seed)
    vpos = np.arange(ntr, len(y))
    fpos = np.flatnonzero(fit_mask)
    sel = fpos[(y[fpos] == 1) | (rng.random(len(fpos)) < 0.1)]
    sw = np.where(y[sel] == 1, 1.0, 10.0)

    def fit_ap(model_eval):
        p = score(sel, model_eval)[sel]
        ok = ~np.isnan(p)
        return float(average_precision_score(y[sel][ok], p[ok], sample_weight=sw[ok])) if y[sel][ok].sum() else 0.0, int((~ok).sum())

    hist, best = [], (-1.0, None, None)
    t_start = time.time()
    for ep in range(max_epochs):
        model.train()
        tot, nb, stop = 0.0, 0, False
        for batch in tr_loader:
            seeds = batch[key].input_id
            mask, _ = prep(batch, seeds)  # train seeds: graph position == input_id (all train edges are seeds)
            batch = batch.to(dev)
            opt.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[key][mask.to(dev)]
            loss = lossf(out, batch[key].y[mask.to(dev)])
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss), nb + 1
            if nb % 200 == 0 and time.time() - t_start > budget_s:
                stop = True
                break
        ap_e, miss_e = fit_ap(True)
        ap_t, _ = fit_ap(False)
        rec = {"epoch": ep, "batches": nb, "loss": tot / max(nb, 1), "fit_ap_eval_mode": ap_e, "fit_ap_train_mode": ap_t,
               "seed_edges_not_sampled": miss_e, "elapsed_s": time.time() - t_start, "partial": stop}  # fmt: skip
        hist.append(rec)
        log(json.dumps(rec))
        mode_eval = ap_e >= ap_t
        if max(ap_e, ap_t) > best[0]:
            best = (max(ap_e, ap_t), mode_eval, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if ckpt is not None:
                torch.save({"epoch": ep, "state": best[2], "eval_mode": mode_eval}, ckpt)
        if stop or time.time() - t_start > budget_s:
            break
    model.load_state_dict(best[2])
    raw = score(vpos, best[1])[vpos]
    return {"raw": raw, "tx_id": txid[vpos], "history": hist, "best_fit_ap": best[0], "eval_mode": bool(best[1]),
            "n_not_scored": int(np.isnan(raw).sum()), "train_edges": ntr, "train_pos": int(y[:ntr].sum()),
            "seconds": time.time() - t0}  # fmt: skip
