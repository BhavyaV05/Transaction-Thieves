"""Calibration as a training objective for the fast model: soft label targets and proper scoring losses (C13).

Owner: T3 (Akshat).  Spec: wave 4 (docs/t3_system1.md).  The closest publishable stand-in for TypeSafe's RL-for-calibration
(RLCD, unpublished): instead of one-hot cross-entropy on the label token, train on the FULL label distribution with a proper
scoring rule, with targets that are the one-hot label mixed with a larger model's token distribution (and smoothed):

    target = (1 - mix) * ((1 - smooth) * onehot + smooth / k) + mix * large_probs
    soft_ce:  -sum_j target_j * log softmax_restricted(z)_j        (the logits of the k label tokens only)
    brier:    sum_j (softmax_restricted(z)_j - target_j)^2

Temperature scaling and the conformal sets still go on top (ftm.fastmodel.splits).  Compared with plain SFT + T on cal by
`scripts/score_fast_baseline.py report` (ECE, Brier, log loss, reliability bins per decision).
"""

from __future__ import annotations

import numpy as np

LOSSES = ("sft", "soft_ce", "brier")


def soft_target(label_index: int, k: int, large_probs: list[float] | None = None, mix: float = 0.5, smooth: float = 0.05) -> list[float]:
    """The training target over the k labels. Without `large_probs` (a row the large model was not scored on) only the smoothed one-hot."""
    if not 0 <= label_index < k:
        raise ValueError(f"label index {label_index} outside 0..{k - 1}")
    hot = np.full(k, smooth / k)
    hot[label_index] += 1.0 - smooth
    if large_probs is None:
        return hot.tolist()
    q = np.clip(np.asarray(large_probs, "f8"), 0, None)
    if len(q) != k or q.sum() <= 0:
        raise ValueError("large_probs must have k non-negative entries with a positive sum")
    return ((1 - mix) * hot + mix * q / q.sum()).tolist()


def label_distribution_loss(logits, target, mask, kind: str = "soft_ce"):
    """Mean loss over a batch. logits, target, mask: (B, K) torch tensors; mask is True for the real labels of the row (decisions
    have 2 or 6 labels, padded to K). The softmax is taken over the real label tokens only."""
    import torch

    if kind not in LOSSES[1:]:
        raise ValueError(f"loss must be one of {LOSSES[1:]}")
    z = logits.masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(z, dim=-1)
    if kind == "soft_ce":
        return -(target * logp.masked_fill(~mask, 0.0)).sum(-1).mean()
    p = logp.exp().masked_fill(~mask, 0.0)
    return ((p - target) ** 2).sum(-1).mean()
