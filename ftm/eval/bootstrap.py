"""Cluster bootstrap for every confidence interval in the report (plan.md 9.3).

Owner: T4 (Shreyas).  Spec: plan.md 9.3; tracks.md 6.7.

Clusters = laundering attempts + connected account components: two alerts are in one cluster if they
belong to the same attempt (`attempt_id`) or are linked through shared accounts. Alerts in one cluster
are not independent (same ring, same accounts, one (tx, bank) pair seen by two banks), so a row-level
bootstrap gives intervals that are too narrow. We resample whole clusters with replacement, 10,000 times.

Fast path: every headline statistic (Brier, Brier differences, gap closed) is a function of per-row means
over the *same* rows, so a resample only needs per-cluster sums and counts: one multinomial draw of
cluster weights, then weights @ sums. `cluster_bootstrap` handles any other statistic (PR-AUC) the slow way.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

N_BOOT = 10_000
SEED = 0


def cluster_ids(attempt_id: Sequence, accounts: Sequence[Sequence], hub_cap: int | None = None) -> np.ndarray:
    """Cluster label per row: union of rows sharing an attempt or any account (union-find).

    attempt_id: one entry per row, None/NaN when the row is in no attempt.
    accounts: one or more account-key columns, each one entry per row (e.g. [src_node, dst_node]).
    hub_cap: if set, an account that appears in more than `hub_cap` rows does not link rows (hub
      accounts chain unrelated alerts into one giant component). Attempts always link. None = the plan's
      definition. Use it as a sensitivity check, and fix the value in prereg/ before test.
    Returns dense int labels 0..k-1 in first-seen order.
    """
    n = len(attempt_id)
    parent = np.arange(n)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:  # path compression
            parent[i], i = root, parent[i]
        return root

    def link(columns, skip=frozenset()) -> None:
        first: dict = {}  # one key space for all columns given: src of row i joins dst of row j
        for col in columns:
            if len(col) != n:
                raise ValueError("every key column needs one entry per row")
            for i, key in enumerate(col):
                if key is None or (isinstance(key, float) and np.isnan(key)) or key in skip:
                    continue
                j = first.setdefault(key, i)
                if j != i:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)

    link([attempt_id])
    hubs: frozenset = frozenset()
    if hub_cap is not None:
        counts = Counter(k for col in accounts for k in col if k is not None)
        hubs = frozenset(k for k, c in counts.items() if c > hub_cap)
    link(accounts, hubs)
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def cluster_report(clusters) -> dict:
    """Sizes to print next to every CI: a giant component makes the bootstrap meaningless, so check it."""
    c = np.bincount(np.asarray(clusters))
    return {"n_rows": int(c.sum()), "n_clusters": int(len(c)), "largest": int(c.max()),
            "largest_share": float(c.max() / c.sum()), "singletons": int((c == 1).sum())}  # fmt: skip


def _weights(n_clusters: int, n_boot: int, seed: int, chunk: int):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        yield rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=b)
        done += b


def bootstrap_means(values, clusters, n_boot: int = N_BOOT, seed: int = SEED, chunk: int = 500) -> np.ndarray:
    """Resampled means of each column of `values` (n_rows x k), resampling clusters. Shape (n_boot, k).

    All columns use the same resampled clusters, so differences and ratios of columns are paired.
    """
    v = np.asarray(values, dtype="f8")
    if v.ndim == 1:
        v = v[:, None]
    cl = np.asarray(clusters)
    if len(cl) != len(v):
        raise ValueError("clusters needs one entry per row")
    _, cl = np.unique(cl, return_inverse=True)
    k = cl.max() + 1
    sums = np.zeros((k, v.shape[1]))
    np.add.at(sums, cl, v)
    counts = np.bincount(cl, minlength=k).astype("f8")
    out = [(w @ sums) / (w @ counts)[:, None] for w in _weights(k, n_boot, seed, chunk)]
    return np.vstack(out)


def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters, n_boot: int = N_BOOT,
                      seed: int = SEED) -> np.ndarray:  # fmt: skip
    """Generic slow path: `stat(row_indices)` on each resample of whole clusters. Shape (n_boot,)."""
    cl = np.asarray(clusters)
    _, cl = np.unique(cl, return_inverse=True)
    members = np.split(np.argsort(cl, kind="stable"), np.cumsum(np.bincount(cl))[:-1])
    rng = np.random.default_rng(seed)
    k = len(members)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        out[b] = stat(np.concatenate([members[j] for j in pick]))
    return out


def percentile_ci(samples, level: float = 0.95) -> tuple[float, float]:
    s = np.asarray(samples, dtype="f8")
    s = s[~np.isnan(s)]
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def brier_diff(y, p_a, p_b, clusters, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> dict:
    """Brier(a) - Brier(b) with a paired cluster-bootstrap CI. Negative = a is better."""
    y = np.asarray(y, dtype="f8")
    se = np.column_stack([(np.asarray(p_a, "f8") - y) ** 2, (np.asarray(p_b, "f8") - y) ** 2])
    bm = bootstrap_means(se, clusters, n_boot, seed)
    d = bm[:, 0] - bm[:, 1]
    lo, hi = percentile_ci(d, level)
    return {"diff": float(se[:, 0].mean() - se[:, 1].mean()), "ci": (lo, hi), "samples": d}


def gap_closed_ci(y, p_one_bank, p_system, p_upper, clusters, n_boot: int = N_BOOT, seed: int = SEED,
                  level: float = 0.95, min_denominator: float | None = None) -> dict:  # fmt: skip
    """Gap closed with a cluster-bootstrap CI. Resamples whose denominator is tiny count as NaN and are
    dropped from the CI; `nan_share` says how many, and a large share means: report Brier differences."""
    from ftm.eval.metrics import MIN_GAP_DENOMINATOR, gap_closed

    md = MIN_GAP_DENOMINATOR if min_denominator is None else min_denominator
    y = np.asarray(y, dtype="f8")
    se = np.column_stack([(np.asarray(p, "f8") - y) ** 2 for p in (p_one_bank, p_system, p_upper)])
    bm = bootstrap_means(se, clusters, n_boot, seed)
    den = bm[:, 0] - bm[:, 2]
    g = np.where(den >= md, (bm[:, 0] - bm[:, 1]) / np.where(den == 0, 1, den), np.nan)
    m = se.mean(axis=0)
    point = gap_closed(m[0], m[1], m[2], md)
    nan_share = float(np.isnan(g).mean())
    ci = percentile_ci(g, level) if nan_share < 1 else (float("nan"), float("nan"))
    return {"gap_closed": point, "ci": ci, "nan_share": nan_share, "samples": g}


def bootstrap_pvalue(samples, null: float = 0.0, alternative: str = "greater") -> float:
    """One-sided bootstrap p-value for H1: stat > null ('greater') or stat < null ('less').

    p = share of resamples on the null side, with the +1 correction so it is never exactly 0.
    """
    s = np.asarray(samples, dtype="f8")
    s = s[~np.isnan(s)]
    if alternative == "greater":
        k = (s <= null).sum()
    elif alternative == "less":
        k = (s >= null).sum()
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    return float((k + 1) / (len(s) + 1))
