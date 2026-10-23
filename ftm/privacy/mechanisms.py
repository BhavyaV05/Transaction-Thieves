"""DP mechanisms for the four answer kinds (C5), their exact likelihoods and the odds-shift ceilings.

Owner: T2 (Bhavya).  Spec: plan.md 6.6, tracks.md 4.5.  Protection unit: one transaction (adjacent datasets differ
by adding or removing one transaction), so a question with sensitivity `s` (ftm.dsl.parser.sensitivity) is
eps-DP when the mechanism below is run with that eps and `s`.

  bool       randomised response               P(keep) = e^eps / (1 + e^eps)
  k-ary      k-ary randomised response         P(keep) = e^eps / (e^eps + k - 1), else uniform over the others
  count      two-sided geometric (discrete Laplace), alpha = e^(-eps / s)   P(noise = z) = (1-a)/(1+a) a^|z|
  set count  simulated DP-PSI cardinality (Purcell et al.): the noise only, no cryptography. Same mechanism as count.

`eps = inf` (or None) is the "no noise" arm of the eps sweep: the exact answer.

The likelihood functions are exact, so back-testing (C7) and the tests can use the true mechanism instead of
estimating it from draws.
"""

from __future__ import annotations

import math

import numpy as np

from ftm.types import AnswerKind

EXACT = math.inf


def is_exact(eps: float | None) -> bool:
    return eps is None or math.isinf(eps)


def _check(eps: float | None) -> None:
    if not is_exact(eps) and not (eps > 0):
        raise ValueError(f"eps must be positive or inf, got {eps!r}")


# --- sampling ---------------------------------------------------------------------------------------------------
def randomized_response(value: bool, eps: float | None, rng: np.random.Generator) -> bool:
    _check(eps)
    if is_exact(eps):
        return bool(value)
    keep = math.exp(eps) / (1.0 + math.exp(eps))
    return bool(value) if rng.random() < keep else (not bool(value))


def kary_randomized_response(value: int, k: int, eps: float | None, rng: np.random.Generator) -> int:
    _check(eps)
    if not (0 <= value < k):
        raise ValueError(f"k-ary value {value} outside 0..{k - 1}")
    if is_exact(eps) or k == 1:
        return int(value)
    keep = math.exp(eps) / (math.exp(eps) + k - 1)
    if rng.random() < keep:
        return int(value)
    other = int(rng.integers(0, k - 1))  # uniform over the k-1 other values
    return other if other < value else other + 1


def two_sided_geometric(eps: float | None, sensitivity: float, rng: np.random.Generator) -> int:
    """One draw of discrete Laplace noise with scale sensitivity/eps (difference of two geometrics)."""
    _check(eps)
    if is_exact(eps) or sensitivity == 0:
        return 0
    alpha = math.exp(-eps / sensitivity)
    p = 1.0 - alpha
    # numpy's geometric counts trials from 1; subtract 1 to get failures from 0
    return int(rng.geometric(p) - 1) - int(rng.geometric(p) - 1)


def noisy_count(value: int, eps: float | None, sensitivity: float, rng: np.random.Generator) -> int:
    return int(value) + two_sided_geometric(eps, sensitivity, rng)


def randomize(kind: AnswerKind, value, eps: float | None, sensitivity: float, rng: np.random.Generator,
              arity: int | None = None):  # fmt: skip
    """Apply the mechanism for `kind` to the exact answer. The one entry point a partner bank uses."""
    if kind == "bool":
        return randomized_response(value, eps, rng)
    if kind == "kary":
        if arity is None:
            raise ValueError("k-ary answers need arity")
        return kary_randomized_response(value, arity, eps, rng)
    if kind in ("count", "set_count"):
        return noisy_count(value, eps, sensitivity, rng)
    raise ValueError(f"unknown answer kind {kind!r}")


# --- exact likelihoods ------------------------------------------------------------------------------------------
def geometric_pmf(z: int, eps: float, sensitivity: float) -> float:
    alpha = math.exp(-eps / sensitivity)
    return (1 - alpha) / (1 + alpha) * alpha ** abs(z)


def likelihood(kind: AnswerKind, eps: float | None, sensitivity: float, true_value, output,
               arity: int | None = None) -> float:  # fmt: skip
    """P(output | true_value) under the mechanism for `kind`."""
    if is_exact(eps):
        return 1.0 if output == true_value else 0.0
    if kind == "bool":
        keep = math.exp(eps) / (1.0 + math.exp(eps))
        return keep if bool(output) == bool(true_value) else 1.0 - keep
    if kind == "kary":
        keep = math.exp(eps) / (math.exp(eps) + arity - 1)
        return keep if output == true_value else (1.0 - keep) / (arity - 1)
    if kind in ("count", "set_count"):
        if sensitivity == 0:
            return 1.0 if output == true_value else 0.0
        return geometric_pmf(int(output) - int(true_value), eps, sensitivity)
    raise ValueError(f"unknown answer kind {kind!r}")


def log_lr(kind: AnswerKind, eps: float | None, sensitivity: float, true_a, true_b, output,
           arity: int | None = None) -> float:  # fmt: skip
    """log P(output | true_a) / P(output | true_b)."""
    pa = likelihood(kind, eps, sensitivity, true_a, output, arity)
    pb = likelihood(kind, eps, sensitivity, true_b, output, arity)
    if pa == 0 and pb == 0:
        return 0.0
    if pb == 0:
        return math.inf
    if pa == 0:
        return -math.inf
    return math.log(pa / pb)


def lr_ceiling(kind: AnswerKind, eps: float | None, sensitivity: float = 1, diff: float = 1) -> float:
    """Largest likelihood ratio any single noisy answer can give between two true answers.

    bool and k-ary: e^eps (any two true values). Count and set count: e^(eps * diff / sensitivity), where `diff` is
    how far apart the two true counts are (diff = sensitivity gives e^eps, the DP guarantee for neighbours; the
    plan's "typical launder/clean difference" is the diff to pass for the ceiling plot).
    """
    if is_exact(eps):
        return math.inf
    if kind in ("bool", "kary"):
        return math.exp(eps)
    if sensitivity == 0:
        return math.inf
    return math.exp(eps * diff / sensitivity)
