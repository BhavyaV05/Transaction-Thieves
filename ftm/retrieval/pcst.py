"""C11: time-respecting prize-collecting Steiner tree (PCST) evidence selection. Pure numpy, no data access.

Owner: T1 (Abhishek).  Spec: plan.md 6.12, tracks.md 3.8.  Retrieval around an alert in the style of G-Retriever
(He et al. 2024): edge prizes become virtual nodes, `pcst_fast` finds the tree, and the K edges are then picked so
that **every selected edge lies on a time-respecting walk to the alerted edge**:

  an edge e=(u, v, t) may be added when an already selected edge f touches u or v with t_f >= t
  (money moves forward in time along the walk e -> f -> ... -> alert; direction of the edges is ignored).

The caller passes only edges with timestamp <= as_of, and nothing here looks at any other edge. `pcst_fast` is
imported inside the function so CI without it still runs the rest.
"""

from __future__ import annotations

import heapq

import numpy as np


def reachable_to_alert(src, dst, t, alert: int, within=None) -> np.ndarray:
    """Bool mask of the edges that lie on a time-respecting walk ending at edge `alert` (itself included).

    Sweep in decreasing time; `latest[node]` = latest time of an included edge at that node. `within` (bool mask)
    restricts the walk to those edges (used to test a selected set).
    """
    src, dst, t = np.asarray(src), np.asarray(dst), np.asarray(t)
    inc = np.zeros(len(t), bool)
    ok = np.ones(len(t), bool) if within is None else np.asarray(within, bool)
    latest = {int(src[alert]): t[alert], int(dst[alert]): t[alert]}
    inc[alert] = True
    order = np.argsort(-t, kind="stable")
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and t[order[j]] == t[order[i]]:
            j += 1
        grp, changed = [k for k in order[i:j] if ok[k] and not inc[k]], True
        while changed:  # edges with the same timestamp can chain, so repeat until stable
            changed = False
            for k in grp:
                u, v = int(src[k]), int(dst[k])
                if not inc[k] and (latest.get(u, -np.inf) >= t[k] or latest.get(v, -np.inf) >= t[k]):
                    inc[k] = True
                    latest[u], latest[v] = max(latest.get(u, t[k]), t[k]), max(latest.get(v, t[k]), t[k])
                    changed = True
        i = j
    return inc


def is_one_trail(src, dst, t, alert: int, chosen) -> bool:
    """Every chosen edge reaches the alerted edge by a time-respecting walk inside the chosen set (and alert is chosen)."""
    chosen = np.asarray(chosen, bool)
    return bool(chosen[alert] and (reachable_to_alert(src, dst, t, alert, within=chosen) == chosen).all())


def is_connected(src, dst, chosen) -> bool:
    """The chosen edges form one connected component of accounts (direction and time ignored)."""
    idx = np.flatnonzero(chosen)
    if len(idx) == 0:
        return True
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in idx:
        parent[find(int(src[k]))] = find(int(dst[k]))
    return len({find(int(src[k])) for k in idx}) == 1


def greedy_trail(src, dst, t, prize, alert: int, k: int, allowed=None) -> np.ndarray:
    """Bool mask of up to `k` edges: start from the alerted edge and repeatedly add the highest-prize eligible edge
    (eligible = touches a selected edge that is not earlier than it). Always a time-respecting trail. `allowed`
    restricts the pool (default: all edges)."""
    n = len(t)
    ok = np.ones(n, bool) if allowed is None else np.asarray(allowed, bool)
    adj: dict[int, list[int]] = {}
    for e in np.argsort(t, kind="stable"):
        if ok[e]:
            adj.setdefault(int(src[e]), []).append(int(e))
            adj.setdefault(int(dst[e]), []).append(int(e))
    ptr = dict.fromkeys(adj, 0)
    latest: dict[int, float] = {}
    chosen = np.zeros(n, bool)
    heap: list = []

    def take(e):
        chosen[e] = True
        for x in (int(src[e]), int(dst[e])):
            latest[x] = max(latest.get(x, -np.inf), t[e])
            lst = adj.get(x, [])
            while ptr.get(x, 0) < len(lst) and t[lst[ptr[x]]] <= latest[x]:
                c = lst[ptr[x]]
                ptr[x] += 1
                if not chosen[c]:
                    heapq.heappush(heap, (-float(prize[c]), int(c)))

    take(alert)
    while heap and chosen.sum() < k:
        _, e = heapq.heappop(heap)
        if not chosen[e]:
            take(e)
    return chosen


def pcst_edges(src, dst, prize, alert: int, edge_cost: float = 0.05, root_prize: float = 1e3) -> np.ndarray:
    """Bool mask of the edges `pcst_fast` selects (G-Retriever construction), rooted at the alerted edge's sender.

    Edges with prize > edge_cost become a virtual node (prize - edge_cost) joined to both ends at no cost; the others stay
    real edges of cost `edge_cost`. Parallel edges are merged by (unordered account pair); the best prize represents the pair,
    and every edge of a selected pair is marked.
    """
    from pcst_fast import pcst_fast

    src, dst, prize = np.asarray(src), np.asarray(dst), np.asarray(prize, "f8")
    nodes, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    u, v = inv[: len(src)], inv[len(src) :]
    a, b = np.minimum(u, v), np.maximum(u, v)
    key = a.astype(np.int64) * len(nodes) + b
    order = np.lexsort((-prize, key))  # per pair: best prize first
    first = order[np.r_[True, key[order][1:] != key[order][:-1]]]
    pa, pb, pp = a[first], b[first], prize[first]
    n0 = len(nodes)
    node_prize = np.zeros(n0)
    node_prize[u[alert]] = root_prize
    node_prize[v[alert]] = root_prize
    big = pp > edge_cost
    real = ~big & (pa != pb)
    virt = np.flatnonzero(big)
    vid = n0 + np.arange(len(virt))
    edges = np.concatenate([np.column_stack([pa[real], pb[real]]), np.column_stack([pa[virt], vid]), np.column_stack([vid, pb[virt]])]).astype(np.int64)
    costs = np.concatenate([np.full(real.sum(), edge_cost), np.zeros(2 * len(virt))])
    prizes = np.concatenate([node_prize, pp[virt] - edge_cost])
    if len(edges) == 0:
        return np.zeros(len(src), bool)
    _, sel_e = pcst_fast(edges, prizes, costs, int(u[alert]), 1, "gw", 0)
    sel = edges[sel_e]
    chosen_pairs = {(int(x), int(y)) for x, y in sel if x < n0 and y < n0}
    chosen_pairs |= {(int(pa[virt[x - n0]]), int(pb[virt[x - n0]])) for x, y in sel for x in (x, y) if x >= n0}
    return np.fromiter(((int(x), int(y)) in chosen_pairs for x, y in zip(a, b, strict=True)), bool, count=len(a))


def select_pcst_trail(src, dst, t, prize, alert: int, k: int = 40, edge_cost: float = 0.05, bonus: float = 10.0):
    """Scorer + PCST view: reachable pool -> PCST backbone -> time-respecting greedy fill of k edges.

    Returns (mask of the k edges, mask of the PCST edges inside the pool). Edges chosen by PCST get `bonus` added to their
    prize, so they are taken first (as soon as they are eligible); the rest of the k comes from the best remaining
    eligible edges.
    """
    pool = reachable_to_alert(src, dst, t, alert)
    idx = np.flatnonzero(pool)
    m = pcst_edges(np.asarray(src)[idx], np.asarray(dst)[idx], np.asarray(prize)[idx], int(np.flatnonzero(idx == alert)[0]), edge_cost)
    pc = np.zeros(len(t), bool)
    pc[idx[m]] = True
    boosted = np.asarray(prize, "f8") + bonus * pc
    return greedy_trail(src, dst, t, boosted, alert, k, allowed=pool), pc
