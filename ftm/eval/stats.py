"""Multiple testing and equivalence (plan.md 9.3).

Owner: T4 (Shreyas).  Spec: plan.md 9.3; tracks.md 6.7.

- Holm (step-down Bonferroni) within a family. Primary family: H1, H2, H3, H4, H9. Secondary hypotheses
  form their own family. Exploratory ones (H5, H8) are never passed in here.
- Holm orders by p-value, not by the listed order. plan.md says "tested in the order H1 -> ... -> H9 with
  Holm correction"; if a pre-specified order is what is meant, that is fixed-sequence testing, which is
  also here. Which one the pre-registration names is decided at G3 (prereg/hypotheses.yaml).
- Equivalence: "as good as" only if the CI of the difference lies inside +-margin, with
  margin = half the Brier gap between the best simple planner and random (plan.md 9.3).
"""

from __future__ import annotations

from collections.abc import Mapping


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down. Returns per hypothesis: p, adjusted p, reject. Stops at the first non-rejection."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    running, stopped = 0.0, False
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))  # adjusted p is monotone in rank
        reject = not stopped and p <= alpha / (m - i)
        stopped = stopped or not reject
        out[name] = {"p": float(p), "p_adj": float(running), "reject": reject}
    return {name: out[name] for name in pvalues}  # caller's order


def fixed_sequence(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Test in the given order at full alpha; stop at the first non-rejection (the rest are not rejected)."""
    out, stopped = {}, False
    for name, p in pvalues.items():
        tested = not stopped
        reject = tested and p <= alpha
        stopped = stopped or not reject
        out[name] = {"p": float(p), "reject": reject, "tested": tested}
    return out


def equivalence_margin(brier_best_simple: float, brier_random: float) -> float:
    """Half the gap between the best simple planner and random (plan.md 9.3)."""
    return 0.5 * abs(brier_random - brier_best_simple)


def equivalent(ci: tuple[float, float], margin: float) -> bool:
    """True if the whole CI of the difference lies inside (-margin, +margin)."""
    lo, hi = ci
    return -margin < lo and hi < margin


def non_inferior(ci_of_ours_minus_theirs: tuple[float, float], margin: float) -> bool:
    """For Brier (lower is better): ours is not worse by more than `margin` if the CI upper bound < margin."""
    return ci_of_ours_minus_theirs[1] < margin
