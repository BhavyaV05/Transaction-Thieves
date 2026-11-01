"""The production profile of the LLM runs (`--profile tuned` of scripts/run_llm_agent.py): which tuned components a run uses.

Owner: T3 (Akshat).  Spec: docs/t3_stub_audit.md, configs/llm/tuned.yaml.

The profile is a plain config (configs/llm/tuned.yaml) naming the frozen components: p0 v2, the evidence view and edge scorer v2,
the realistic headroom_v1 back-test table, the fitted ConditionalFusion, the decision config (threshold) and the concurrency.
`profile "none"` is the legacy behaviour (every component explicit on the command line, naive fusion unless given), reachable
only by naming it: dev toolbox runs and old ablation recipes.

`pick_evidence_view` is the documented rule that chose the profile's evidence view from the val fit+cal report of
`python -m ftm.retrieval.evidence_views --data hi_medium --parts fit cal --scorer v2` (kept as configs/llm/evidence_view_fitcal_v2.json; never the eval part): tests/test_profile.py re-derives it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "configs" / "llm"
PROFILES = ("tuned", "none")
TRAIL_MIN = 0.99  # a view must give one connected, time-respecting trail on at least this share of alerts
RECALL_TIE = 0.01  # recall within this of the best counts as a tie; the cheaper (fewer tokens) view wins the tie


def load_profile(name: str) -> dict:
    """The dict of configs/llm/<name>.yaml; the legacy profile "none" is empty (nothing defaulted)."""
    if name == "none":
        return {}
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose one of {PROFILES}")
    return yaml.safe_load((PROFILE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def decision_threshold(path: str | Path) -> float:
    """`threshold` of a decision config (configs/decision/*.yaml): escalate at or above it."""
    p = Path(path)
    cfg = yaml.safe_load((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8"))
    if cfg.get("threshold") is None:
        raise ValueError(f"{path} has no fitted threshold")
    return float(cfg["threshold"])


def pick_evidence_view(report: dict, candidates: tuple[str, ...] = ("scorer_pcst", "scorer_greedy"), reference: str = "amount_1hop",
                       token_cap: float = 4000.0) -> str:
    """Rule for the planner's evidence view, read from a val fit+cal `evidence_views.evaluate` report (pattern-edge recall at k=40, edge
    scorer v2). Candidates are the scorer-driven trail views; one qualifies when its trail is connected and time-respecting on
    >= TRAIL_MIN of alerts and its longest evidence fits `token_cap` (the prompt cap minus the reserve). The best recall wins; views
    within RECALL_TIE of it tie and the one with the fewest mean tokens wins. Floor: if the winner's recall is more than RECALL_TIE
    below the `reference` (the week-2 selection, `amount_1hop`), the reference is the default instead, so the scorer never makes the
    evidence worse. `amount` / `scorer` (no trail), `long_prompt` (no selection) and the explainer views are not candidates."""
    rows = {v["view"]: v for v in report["views"]}
    ok = [n for n in candidates if n in rows and rows[n]["one_time_respecting_trail"] >= TRAIL_MIN and rows[n]["max_tokens_est"] <= token_cap]
    if not ok:
        return reference
    best = max(rows[n]["recall_at_k"] for n in ok)
    win = min((n for n in ok if rows[n]["recall_at_k"] >= best - RECALL_TIE), key=lambda n: (rows[n]["mean_tokens_est"], n))
    return win if rows[win]["recall_at_k"] >= rows[reference]["recall_at_k"] - RECALL_TIE else reference
