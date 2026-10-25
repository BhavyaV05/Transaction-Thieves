"""T1 layer-wise exchange (Ceydeli et al.): Theorem 1 and 2 on a toy graph, and the no-exchange / forward-only contrasts."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("torch_scatter")

from ftm.baselines import layerwise as lw  # noqa: E402


def _graph(own: np.ndarray, node_ids: np.ndarray, src, dst, x, owner_of_global, me):
    """A client's extended subgraph in local indices (own = global ids of the nodes it owns)."""
    keep = np.isin(src, own) | np.isin(dst, own)
    s, d = src[keep], dst[keep]
    nodes, inv = np.unique(np.concatenate([s, d]), return_inverse=True)
    return SimpleNamespace(nodes=nodes, src=inv[: len(s)], dst=inv[len(s) :], x_edge=x[keep], owned=np.isin(nodes, own),
                           owner=np.array([owner_of_global.get(int(n), -1) for n in nodes]), keep=keep, me=me)  # fmt: skip


def _setup(seed=0, n=14, m=60):
    rng = np.random.default_rng(seed)
    src, dst = rng.integers(0, n, m), rng.integers(0, n, m)
    x = rng.normal(size=(m, lw.EDGE_FEATURES)).astype("f4")
    y = (rng.random(m) < 0.4).astype("f4")
    owners = {i: (0 if i < n // 2 else 1) for i in range(n)}
    parts = [_graph(np.array([i for i in range(n) if owners[i] == c]), None, src, dst, x, owners, c) for c in (0, 1)]
    whole = _graph(np.arange(n), None, src, dst, x, {i: 0 for i in range(n)}, 0)
    return src, dst, y, parts, whole


def _logits(graphs, mode, net_for, picks):
    plan = lw.build_plan(graphs)
    gt, pt = lw.to_device(graphs, plan, "cpu")
    return lw.forward([net_for(c) for c in range(len(graphs))], gt, pt, mode, "cpu", picks)


def _net(parts):
    torch.manual_seed(0)
    din, dout = lw.degree_histograms(parts)
    return lw.make_model(din, dout, hidden=10)


def test_exchange_equals_centralised_forward_and_gradient():
    src, dst, y, parts, whole = _setup()
    net = _net([whole])  # the same parameters everywhere (Sync-SGD), degree constants fixed
    # edges trained at their sender's client: the edge is labelled where the sender is owned
    lab = [np.flatnonzero(np.isin(p.nodes[p.src], np.array([i for i in range(14) if p.me == (0 if i < 7 else 1)]))) for p in parts]
    cen = np.flatnonzero(whole.keep)  # all edges, in the same order as `src`
    sender_client = np.where(src < 7, 0, 1)

    def run(graphs, picks, mode):
        net.zero_grad()
        out = _logits(graphs, mode, lambda c: net, picks)
        return out

    # centralised logits of every edge
    cen_out = run([whole], [torch.arange(len(whole.src))], "bwd")[0]
    # distributed: each client's logits for the edges whose sender it owns
    picks = [torch.as_tensor(ids) for ids in lab]
    dist = run(parts, picks, "bwd")
    for c in (0, 1):
        edges = np.flatnonzero(parts[c].keep)[lab[c]]  # global edge index of those local edges
        assert np.all(sender_client[edges] == c)
        assert torch.allclose(dist[c], cen_out[torch.as_tensor(edges)], atol=1e-5), "Theorem 1: representations equal"
    # gradients of the summed loss of every edge: distributed (backward exchange) equals centralised (Theorem 2)
    yt = torch.as_tensor(y)

    def grads(out_list, edge_lists):
        loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(o, yt[torch.as_tensor(e)], reduction="sum") for o, e in zip(out_list, edge_lists, strict=True))
        net.zero_grad()
        loss.backward()
        return torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])

    g_cen = grads([cen_out], [cen])
    out = run(parts, picks, "bwd")
    g_dist = grads(out, [np.flatnonzero(parts[c].keep)[lab[c]] for c in (0, 1)])
    assert torch.allclose(g_cen, g_dist, atol=1e-4, rtol=1e-3)
    # forward-only exchange keeps the logits but loses the adjoints of the remote copies
    out_f = run(parts, picks, "fwd")
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(out_f, dist, strict=True))
    g_fwd = grads(out_f, [np.flatnonzero(parts[c].keep)[lab[c]] for c in (0, 1)])
    assert not torch.allclose(g_cen, g_fwd, atol=1e-4, rtol=1e-3)
    # without exchange the remote representations are 0 and the logits differ
    out_n = run(parts, picks, "none")
    assert not all(torch.allclose(a, b, atol=1e-4) for a, b in zip(out_n, dist, strict=True))


def test_plan_marks_remote_and_unowned_nodes():
    _, _, _, parts, _ = _setup()
    plan = lw.build_plan(parts)
    for g, p in zip(parts, plan, strict=True):
        assert set(p["remote"]) == set(np.flatnonzero(~g.owned))
        assert len(p["pos"]) == len(p["rows"]) and len(p["unowned"]) == 0  # both clients cover all nodes here
