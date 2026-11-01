"""Time-respecting GNN scoring option (finding 19): the kwargs handed to PyG's LinkNeighborLoader (PyG itself is not needed here)."""

from types import SimpleNamespace

import pytest

from ftm.baselines.multi_gnn import causal_kwargs


class _Store(dict):
    def __missing__(self, k):
        self[k] = SimpleNamespace()
        return self[k]


def test_causal_kwargs_use_seed_edge_times_and_tag_the_reverse_edges():
    torch = pytest.importorskip("torch")
    key = ("node", "to", "node")
    d = _Store()
    d[key].timestamps = torch.tensor([0.0, 10.0, 20.0, 30.0])
    kw = causal_kwargs(d, key, torch.tensor([3, 1]))
    assert kw["time_attr"] == "timestamps" and kw["temporal_strategy"] == "uniform"
    assert kw["edge_label_time"].tolist() == [30.0, 10.0]
    assert d["node", "rev_to", "node"].timestamps.tolist() == [0.0, 10.0, 20.0, 30.0]
