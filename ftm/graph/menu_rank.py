"""The planner's menu under a token budget: rank by expected Brier reduction, show the top K, say what was dropped.

Owner: T3 (Akshat).  Spec: docs/t3_prompt_budget.md.

With the tuned table every one of the 440 hypotheses is accepted, so the routed menu is about 50 questions with a full bin -> LR
map at four eps levels each, which alone is more than the whole prompt cap. The menu items are ranked by the quantity System 1's
`worth_asking` label uses (`ftm.fusion.gain.expected_gain`: expected Brier reduction at the CURRENT fused posterior under the
fitted fusion, with the table's cell probabilities) and only the top K are shown. An item whose gain is <= 0 is never shown: in
expectation it can only hurt (the fusion overshoots a weak answer). The ranking is recomputed every planner step, from the
posterior after the answers so far; questions already asked and eps levels the budget cannot pay are not candidates.

`MENU_TOP_K = None` everywhere else means the legacy behaviour: the whole routed menu, in table order (`--profile none`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from ftm.fusion.fuse import logit
from ftm.fusion.gain import Cells, UnitFusion, expected_gain
from ftm.graph.policy import MenuItem

DROP_KEYS = ("dropped_nonpositive", "dropped_no_cells", "beyond_top_k")


def _fmt(x: float) -> str:
    return f"{x:g}"


def compact_lr(item: MenuItem) -> str:
    """The bin -> LR map of one (question, eps) in a few tokens: `yes:0.49 no:1.86`, categorical `0:0.83 1:0.99`, numeric
    `<1:1.84 1:1.23 2:1.27 4:0.32` (each number is the LOWER edge of its bin, up to the next one; the first bin is below the
    first edge)."""
    lr = dict(item.lr_bins)
    if set(lr) <= {"true", "false"}:
        return " ".join(f"{'yes' if k == 'true' else 'no'}:{lr[k]:.2f}" for k in ("true", "false") if k in lr)
    e = list(item.edges)
    if e and all(f"b{i}" in lr for i in range(len(e) + 1)):
        return " ".join([f"<{_fmt(e[0])}:{lr['b0']:.2f}", *[f"{_fmt(e[i - 1])}:{lr[f'b{i}']:.2f}" for i in range(1, len(e) + 1)]])
    return " ".join(f"{k}:{v:.2f}" for k, v in lr.items())


def rank_menu(menu: Iterable[MenuItem], model, posterior: float, eps_left: float, asked: set[str], top_k: int | None
              ) -> tuple[list[MenuItem], dict]:
    """(shown items best first, counts). `model`: the fitted ConditionalFusion (`fusion.model`), None for a fusion with no fitted
    weights. `top_k` None keeps every positive-gain item."""
    model = model if model is not None else UnitFusion()
    z = float(logit(min(max(posterior, 1e-9), 1 - 1e-9)))
    scored: list[tuple[float, MenuItem]] = []
    stats = dict.fromkeys(DROP_KEYS, 0) | {"offered": 0}
    for m in menu:
        if m.canonical in asked or m.eps > eps_left + 1e-9:
            continue  # the planner cannot ask it (again / at that price): not an item of the menu at this step
        stats["offered"] += 1
        g = expected_gain(model, z, Cells(m.family, dict(m.lr_bins), m.p_cell))
        if g is None:
            stats["dropped_no_cells"] += 1
        elif not g > 0 or math.isnan(g):
            stats["dropped_nonpositive"] += 1
        else:
            scored.append((g, m))
    scored.sort(key=lambda t: (-t[0], t[1].eps, t[1].canonical))
    keep = scored if top_k is None else scored[:top_k]
    stats["beyond_top_k"] = len(scored) - len(keep)
    stats["kept"] = len(keep)
    stats["gains"] = [round(g, 6) for g, _ in keep]  # best first, aligned with the returned items
    return [m for _, m in keep], stats


def ranked_questions(menu_raw: list[dict], shown: list[MenuItem]) -> list[dict]:
    """The prompt's question list for a ranked menu: one entry per question (the id the schema enumerates), best first,
    carrying only the eps levels that made the top K and, as `lr_text`, one `eps e: <bin -> LR map>` per level in rank order."""
    raw = {q["program"]: q for q in menu_raw}
    by_prog: dict[str, list[MenuItem]] = {}
    for m in shown:  # rank order; a question sits where its best level sits
        by_prog.setdefault(m.canonical, []).append(m)
    return [{**raw[prog], "eps": [m.eps for m in ms], "lr_text": " | ".join(f"eps {m.eps:g}: {compact_lr(m)}" for m in ms), "compact": True}
            for prog, ms in by_prog.items()]
