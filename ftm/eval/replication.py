"""Does an accepted question keep its evidence inside time blocks?  The replication check for the back-test LRs.

Owner: T4 (Shreyas).  Spec: GitHub issue #84, docs/t4_fusion_findings.md section 7.  Script: `scripts/replication_check.py`.

Why.  The back-test estimates LR(cell) = P(cell | launder) / P(cell | clean) from the train answers pooled over time.
When the launder rate and the answer rate both drift with time (the HI-Medium train period: launder rate 5% -> 37% and,
for history questions like `dormant`, the share of `true` answers 88% -> 15%, a ledger warm-up effect) the pooled LR is
a Simpson's-paradox artefact: inside each time block the question carries no signal and the sign of the evidence is
the one that flips on validation. The label-permutation e-value of the acceptance step assumes labels are exchangeable
given the answers, which drift breaks, so such a question can pass e-BH.

What this module computes for ONE question, from its time-ordered answers and labels (`Backtester.collect`):
  cell        the answer cell with the largest |pooled log LR| (first in cell order on a tie): the cell that carries
              the claimed evidence
  pooled      log LR of that cell on all answers, alpha-smoothed exactly as `ftm.backtest.lr.estimate_lr`
              (log[(P(cell|launder)+alpha) / (P(cell|clean)+alpha)]), without the ceiling clip
  blocks      the same log LR inside each of `sign_blocks` equal time blocks (2 = halves)
  mh          Mantel-Haenszel time-stratified log LR of the cell over `mh_blocks` equal time blocks (default 4):
              log of sum_i a_i n0_i / n_i  over  sum_i b_i n1_i / n_i, a_i / b_i = (launder / clean) answers in the
              cell in block i (+ `continuity` each), n1_i / n0_i = launder / clean answers, n_i = n1_i + n0_i
  replicates  True iff pooled and every sign block have the same strictly non-zero sign. Stated explicitly:
              * a log LR of exactly 0.0 (or nan) has no sign, so it does NOT replicate (we cannot confirm a direction);
              * a block with fewer than `min_block_answers` answers or fewer than `min_block_class` launder or clean
                answers cannot be judged, so the question does NOT replicate (reason "thin_block");
              * a question with fewer answers than `sign_blocks` * `min_block_answers` is therefore never confirmed.
              The rule uses train data only; no validation label is involved.

Blocks are `numpy.array_split` of the time-ordered answers: equal counts, not equal durations.

The functions are pure (lists in, numbers out); only `replication_report` touches the back-test engine, and it takes
the collector as an argument so tests need no data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ftm.backtest.lr import cells_for

ALPHA = 0.01  # the LR smoothing of ftm.backtest.lr.estimate_lr / BacktestConfig.alpha


@dataclass(frozen=True)
class ReplicationConfig:
    sign_blocks: int = 2  # time blocks that must agree in sign (2 = halves)
    mh_blocks: int = 4  # strata of the Mantel-Haenszel estimate
    min_block_answers: int = 30  # a block with fewer answers cannot be judged -> no replication
    min_block_class: int = 5  # ... nor one with fewer launder or clean answers than this
    alpha: float = ALPHA
    continuity: float = 0.5  # added to both counts of every MH stratum

    def __post_init__(self):
        if self.sign_blocks < 2 or self.mh_blocks < 1:
            raise ValueError("sign_blocks must be >= 2 and mh_blocks >= 1")


# --- pure pieces --------------------------------------------------------------------------------------------------
def cell_universe(kind: str, bins: Sequence[float] | None = None, keys: Sequence[str] = ()) -> list[str]:
    """The answer cells of a question in the order `estimate_lr` uses. k-ary cells come from `keys` (the LR table's)."""
    if kind == "bool":
        return ["false", "true"]
    if kind in ("count", "set_count"):
        return [f"b{i}" for i in range(len(bins or ()) + 1)]
    return sorted(keys)


def blocks_of(n: int, k: int) -> list[np.ndarray]:
    """Index arrays of `k` equal-count blocks of a time-ordered series of length n."""
    return np.array_split(np.arange(n), k)


def log_lr(c_clean: float, c_launder: float, n_clean: int, n_launder: int, alpha: float = ALPHA) -> float:
    """alpha-smoothed log LR of a cell from its counts; nan when a class is empty (no rate to compare)."""
    if n_clean <= 0 or n_launder <= 0:
        return math.nan
    return float(math.log((c_launder / n_launder + alpha) / (c_clean / n_clean + alpha)))


def cell_counts(cells: Sequence[str], labels: Sequence[bool], universe: Sequence[str]) -> dict[str, list[int]]:
    """cell -> [n clean, n launder]. A cell outside `universe` is counted too, so nothing is dropped silently."""
    cnt: dict[str, list[int]] = {k: [0, 0] for k in universe}
    for c, y in zip(cells, labels, strict=True):
        cnt.setdefault(c, [0, 0])[1 if y else 0] += 1
    return cnt


def cell_log_lr(cells: Sequence[str], labels: Sequence[bool], cell: str, alpha: float = ALPHA) -> float:
    y = np.asarray(labels, dtype=bool)
    inc = np.fromiter((c == cell for c in cells), dtype=bool, count=len(cells))
    return log_lr(int((inc & ~y).sum()), int((inc & y).sum()), int((~y).sum()), int(y.sum()), alpha)


def best_cell(cells: Sequence[str], labels: Sequence[bool], universe: Sequence[str], alpha: float = ALPHA) -> str:
    """The cell with the largest |pooled log LR|; the first in `universe` order on a tie. nan counts as 0."""
    y = np.asarray(labels, dtype=bool)
    n_c, n_l = int((~y).sum()), int(y.sum())
    cnt = cell_counts(cells, labels, universe)
    best, best_v = None, -1.0
    for k in dict.fromkeys([*universe, *cnt]):
        v = log_lr(cnt[k][0], cnt[k][1], n_c, n_l, alpha)
        v = 0.0 if math.isnan(v) else abs(v)
        if v > best_v + 1e-12:
            best, best_v = k, v
    if best is None:
        raise ValueError("no cells")
    return best


def mh_log_lr(cells: Sequence[str], labels: Sequence[bool], cell: str, k: int = 4, continuity: float = 0.5) -> float:
    """Mantel-Haenszel time-stratified log LR of `cell` over `k` equal-count blocks (see the module docstring).

    nan when no stratum holds both classes. 0 when the cell has the same rate in both classes inside every stratum."""
    y = np.asarray(labels, dtype=bool)
    inc = np.fromiter((c == cell for c in cells), dtype=bool, count=len(cells))
    num = den = 0.0
    for idx in blocks_of(len(y), k):
        if len(idx) == 0:
            continue
        yy, ii = y[idx], inc[idx]
        n1, n0 = int(yy.sum()), int((~yy).sum())
        n = n1 + n0
        a = int((ii & yy).sum()) + continuity
        b = int((ii & ~yy).sum()) + continuity
        num += a * n0 / n
        den += b * n1 / n
    if num <= 0 or den <= 0:
        return math.nan
    return float(math.log(num / den))


def block_summary(cells: Sequence[str], labels: Sequence[bool], cell: str, k: int, cfg: ReplicationConfig) -> list[dict]:
    """Per time block: n, launder rate, the share of answers in `cell`, the log LR of `cell` and whether the block is thin."""
    y = np.asarray(labels, dtype=bool)
    inc = np.fromiter((c == cell for c in cells), dtype=bool, count=len(cells))
    out = []
    for idx in blocks_of(len(y), k):
        yy, ii = y[idx], inc[idx]
        n1, n0 = int(yy.sum()), int((~yy).sum())
        n = n1 + n0
        out.append({
            "n": n, "n_launder": n1, "launder_rate": n1 / n if n else math.nan,
            "cell_rate": float(ii.mean()) if n else math.nan,
            "log_lr": log_lr(int((ii & ~yy).sum()), int((ii & yy).sum()), n0, n1, cfg.alpha),
            "thin": bool(n < cfg.min_block_answers or min(n1, n0) < cfg.min_block_class),
        })  # fmt: skip
    return out


def _sign(x: float) -> int:
    return 0 if x is None or math.isnan(x) or x == 0 else (1 if x > 0 else -1)


def sign_replicates(pooled: float, blocks: Sequence[Mapping]) -> tuple[bool, str]:
    """(flag, reason): pooled and every block share one strictly non-zero sign and no block is thin."""
    if any(b["thin"] for b in blocks):
        return False, "thin_block"
    s = _sign(pooled)
    if s == 0:
        return False, "zero_pooled"
    if any(_sign(b["log_lr"]) == 0 for b in blocks):
        return False, "zero_block"
    if any(_sign(b["log_lr"]) != s for b in blocks):
        return False, "sign_flip"
    return True, "ok"


def check_question(answers: Sequence, labels: Sequence[bool], kind: str, bins: Sequence[float] | None,
                   cfg: ReplicationConfig | None = None, universe: Sequence[str] | None = None) -> dict:  # fmt: skip
    """The replication numbers of one question from its time-ordered `answers` and `labels` (pure).

    `kind` and `bins` define the cells (as in the back-test entry); `universe` overrides the cell list (k-ary)."""
    cfg = cfg or ReplicationConfig()
    if len(answers) != len(labels):
        raise ValueError("answers and labels differ in length")
    if len(answers) == 0:
        raise ValueError("no answers")
    cells = cells_for(kind, answers, bins)
    uni = list(universe) if universe is not None else cell_universe(kind, bins, sorted(set(cells)))
    cell = best_cell(cells, labels, uni, cfg.alpha)
    pooled = cell_log_lr(cells, labels, cell, cfg.alpha)
    blocks = block_summary(cells, labels, cell, cfg.sign_blocks, cfg)
    flag, reason = sign_replicates(pooled, blocks)
    return {
        "cell": cell, "pooled_log_lr": pooled, "blocks": blocks, "block_log_lrs": [b["log_lr"] for b in blocks],
        "mh_log_lr": mh_log_lr(cells, labels, cell, cfg.mh_blocks, cfg.continuity), "mh_blocks": cfg.mh_blocks,
        "sign_blocks": cfg.sign_blocks, "n_answers": len(labels), "replicates": flag, "reason": reason,
    }  # fmt: skip


# --- the report over the accepted entries ------------------------------------------------------------------------
Collector = Callable[[object, float], tuple[list, list[bool], int]]


def backtest_collector(banks, alerts) -> Collector:
    """`Backtester.collect` over `alerts` (the train alerts, for the real check)."""
    from ftm.backtest.engine import BacktestConfig, Backtester

    return Backtester(banks, alerts, BacktestConfig()).collect


def replication_report(banks, alerts, entries: Sequence[dict], config: ReplicationConfig | None = None,
                       collector: Collector | None = None, only_accepted: bool = True) -> list[dict]:  # fmt: skip
    """One row per (accepted) entry: question, eps, cell, claimed log LR, pooled, blocks, MH, replicates.

    The answers are asked again with `Backtester.collect(node, eps)` (same reproducible noise as the cache, so the
    counts match the entry's `cells`; a mismatch is flagged in `matches_cache`). `collector` replaces the engine for tests."""
    from ftm.dsl.parser import parse

    cfg = config or ReplicationConfig()
    collect = collector or backtest_collector(banks, alerts)
    rows = []
    for e in entries:
        if only_accepted and not e.get("accepted"):
            continue
        if not e.get("eligible", True) or not e.get("lr"):
            continue
        answers, labels, _ = collect(parse(e["program_canonical"]), e["eps"])
        if not answers:
            continue
        r = check_question(answers, labels, e["kind"], e.get("bins") or None, cfg, universe=list(e["lr"]))
        claimed = math.log(e["lr"][r["cell"]]) if e["lr"].get(r["cell"], 0) > 0 else math.nan
        r |= {
            "question": e["program_canonical"], "eps": e["eps"], "family": e.get("family"), "kind": e["kind"],
            "claimed_log_lr": claimed, "accepted": bool(e.get("accepted")),
            # the re-asked answers reproduce the cache entry (same counts) -> the check judges the claimed evidence
            "matches_cache": bool(len(labels) == e["n_answers"] and sum(labels) == e["n_launder"])
            if "n_answers" in e and "n_launder" in e else None,
        }  # fmt: skip
        rows.append(r)
    return rows


def add_validation(rows: list[dict], entries: Sequence[dict], collector: Collector, alpha: float = ALPHA) -> list[dict]:
    """Add the validation log LR of each row's cell and whether its direction agrees with the train (pooled) direction.

    For reporting only; the replication flag never looks at it. `collector` asks the val alerts."""
    from ftm.dsl.parser import parse

    by_key = {(e["program_canonical"], e["eps"]): e for e in entries}
    for r in rows:
        e = by_key[(r["question"], r["eps"])]
        answers, labels, _ = collector(parse(r["question"]), r["eps"])
        if not answers:
            r |= {"val_log_lr": math.nan, "n_val": 0, "n_val_cell": 0, "agrees_on_val": None}
            continue
        cells = cells_for(r["kind"], answers, e.get("bins") or None)
        v = cell_log_lr(cells, labels, r["cell"], alpha)
        r |= {"val_log_lr": v, "n_val": len(labels), "n_val_cell": sum(c == r["cell"] for c in cells),
              "agrees_on_val": None if _sign(v) == 0 else bool(_sign(v) == _sign(r["pooled_log_lr"]))}  # fmt: skip
    return rows


def summarize(rows: Sequence[Mapping]) -> dict:
    """Counts, mean |log LR| (claimed, pooled, MH) and the val agreement rate of kept vs dropped questions."""
    keep = [r for r in rows if r["replicates"]]
    drop = [r for r in rows if not r["replicates"]]

    def agree(rs):
        v = [r["agrees_on_val"] for r in rs if r.get("agrees_on_val") is not None]
        return {"n": len(v), "agree": int(sum(v)), "rate": (sum(v) / len(v)) if v else None}

    def mean_abs(key):
        v = [abs(r[key]) for r in rows if r.get(key) is not None and not math.isnan(r[key])]
        return float(np.mean(v)) if v else None

    return {
        "n_questions": len(rows), "n_replicating": len(keep), "n_non_replicating": len(drop),
        "mean_abs_claimed": mean_abs("claimed_log_lr"), "mean_abs_pooled": mean_abs("pooled_log_lr"),
        "mean_abs_mh": mean_abs("mh_log_lr"), "val_agreement_replicating": agree(keep),
        "val_agreement_non_replicating": agree(drop),
    }  # fmt: skip


def summary_line(s: Mapping, name: str = "") -> str:
    def pct(a):
        return f"{a['agree']}/{a['n']}" if a["n"] else "n/a"

    def f(x):
        return "n/a" if x is None else f"{x:.2f}"

    return (f"replication [{name}]: {s['n_replicating']} of {s['n_questions']} accepted questions keep their direction "
            f"across time blocks; mean |log LR| claimed {f(s['mean_abs_claimed'])}, pooled {f(s['mean_abs_pooled'])}, "
            f"Mantel-Haenszel {f(s['mean_abs_mh'])}; val direction agrees {pct(s['val_agreement_replicating'])} "
            f"kept vs {pct(s['val_agreement_non_replicating'])} dropped")


def _f(x, spec: str = "+.2f") -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def to_markdown(rows: Sequence[Mapping], summary: Mapping, name: str = "") -> str:
    has_val = any("val_log_lr" in r for r in rows)
    head = "| question | eps | cell | claimed | pooled | blocks | MH | replicates |" + (" val | agrees |" if has_val else "")
    sep = "|---|---|---|---|---|---|---|---|" + ("---|---|" if has_val else "")
    lines = [f"# Replication check: {name}", "", summary_line(summary, name), "", head, sep]
    for r in rows:
        blocks = " / ".join(_f(x) for x in r["block_log_lrs"])
        line = (f"| `{r['question']}` | {r['eps']:g} | {r['cell']} | {_f(r['claimed_log_lr'])} | {_f(r['pooled_log_lr'])} "
                f"| {blocks} | {_f(r['mh_log_lr'])} | {'yes' if r['replicates'] else 'NO (' + r['reason'] + ')'} |")
        if has_val:
            ag = r.get("agrees_on_val")
            line += f" {_f(r.get('val_log_lr'))} | {'' if ag is None else ('yes' if ag else 'no')} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


# --- the filter for G2 / diagnostics -----------------------------------------------------------------------------
def replicates_map(rows: Sequence[Mapping]) -> dict[tuple[str, float], bool]:
    return {(r["question"], r["eps"]): bool(r["replicates"]) for r in rows}


def write_report(path: str | Path, rows: Sequence[Mapping], summary: Mapping, config: ReplicationConfig, **meta) -> None:
    doc = {"config": asdict(config), "summary": summary, "rows": list(rows), **meta}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, default=str, allow_nan=True))


def load_filter(path: str | Path) -> tuple[dict[tuple[str, float], bool], dict]:
    """(replicates map, the file's config) from a JSON written by `write_report` / scripts/replication_check.py."""
    doc = json.loads(Path(path).read_text())
    return replicates_map(doc["rows"]), doc.get("config", {})


def add_filter_args(ap) -> None:
    """The two options shared by scripts/g2_check.py and scripts/fusion_diagnostics.py."""
    ap.add_argument("--replication-filter", default=None, metavar="PATH",
                    help="JSON of scripts/replication_check.py: accepted questions that do not replicate get e_value=1 "
                         "before e-BH (m unchanged). Default: no filter")  # fmt: skip
    ap.add_argument("--replication-blocks", type=int, default=None, metavar="K",
                    help="same filter computed on the train world here (needs data/): K time blocks must agree in sign "
                         "(2 = halves). Not with --replication-filter. Default: no filter")  # fmt: skip


def filter_from_args(a, entries: list[dict], train_loader: Callable[[], tuple], fdr: float = 0.10):
    """Apply the filter the options ask for. Returns (entries, info); info is None when no filter is asked for.

    `train_loader()` -> (banks, train alerts), called only for --replication-blocks."""
    from ftm.eval import g2

    if a.replication_filter and a.replication_blocks:
        raise ValueError("give --replication-filter or --replication-blocks, not both")
    if a.replication_filter:
        rep, cfg = load_filter(a.replication_filter)
        source = {"replication_filter": str(a.replication_filter), "config": cfg}
    elif a.replication_blocks:
        cfg_obj = ReplicationConfig(sign_blocks=a.replication_blocks)
        banks, train = train_loader()
        rep = replicates_map(replication_report(banks, train, entries, cfg_obj))
        source = {"replication_blocks": a.replication_blocks, "config": asdict(cfg_obj)}
    else:
        return entries, None
    return g2.apply_replication_filter(entries, rep, fdr, source)


# --- a synthetic series with a planted time confound (tests and the --synthetic smoke run; never a result) --------
def planted_series(n: int = 800, seed: int = 0, mode: str = "confound", kind: str = "bool") -> tuple[list, list[bool]]:
    """Time-ordered (answers, labels) of a planted pattern.

    confound  the launder rate AND the rate of the answer both jump between the first and second half (5% -> 37% and
              88% -> 15% `true`) and the answer is independent of the label inside each half: Simpson, the pooled LR
              is strong, the halves cancel
    genuine   constant rates and a real label dependence: the same sign in every block
    null      constant rates and no dependence
    kind "count" gives count answers on the same pattern (a high count plays the role of `true`)."""
    if mode not in ("confound", "genuine", "null"):
        raise ValueError("mode must be confound, genuine or null")
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    if mode == "confound":
        labels = rng.random(n) < np.where(t < 0.5, 0.05, 0.37)
        hi = rng.random(n) < np.where(t < 0.5, 0.88, 0.15)  # the same for both classes inside a half
    elif mode == "genuine":
        labels = rng.random(n) < 0.25
        hi = rng.random(n) < np.where(labels, 0.7, 0.3)
    else:
        labels = rng.random(n) < 0.25
        hi = rng.random(n) < 0.5
    answers: list = [bool(x) for x in hi] if kind == "bool" else [int(rng.poisson(8 if h else 2)) for h in hi]
    return answers, [bool(x) for x in labels]


PLANTED = {  # program -> (kind, train mode, val mode): what the planted world answers
    "dormant(ALERT.receiver, days=7)": ("bool", "confound", "null"),
    "returns_home(ALERT.receiver, window=after(72))": ("bool", "genuine", "genuine"),
    "n_counterparties(ALERT.receiver, dir=in, window=before(72))": ("count", "confound", "null"),
    "n_tx(ALERT.receiver, dir=in, window=around(72))": ("count", "genuine", "genuine"),
}


def planted_world(n: int = 800, seed: int = 0, n_val: int = 400, eps: float = 1.0):
    """(entries, train collector, val collector) with no data: four accepted questions (two time-confounded, two genuine),
    entries built by the back-test's own `estimate_lr` / `choose_bins` from the planted train series, e_value huge."""
    from ftm.backtest.lr import choose_bins, estimate_lr
    from ftm.dsl.parser import canonical, parse

    train, val, entries = {}, {}, []
    for i, (prog, (kind, m_tr, m_va)) in enumerate(PLANTED.items()):
        canon = canonical(parse(prog))
        a, y = planted_series(n, seed + i, m_tr, kind)
        train[canon] = (a, y)
        val[canon] = planted_series(n_val, seed + 100 + i, m_va, kind)
        bins = choose_bins(a) if kind == "count" else ()
        est = estimate_lr(kind, a, y, eps, 1.0, None, ALPHA, bins)
        entries.append({
            "program_canonical": canon, "eps": eps, "family": canon.split("(")[0], "kind": kind,
            "eligible": True, "accepted": True, "e_value": 1e12, "log_e": math.log(1e12), "bins": list(bins),
            "lr": est.lr, "cells": {k: list(v) for k, v in est.counts.items()}, "n_answers": len(y),
            "n_launder": int(sum(y)), "n_clean": int(len(y) - sum(y)),
        })  # fmt: skip

    def make(book):
        def collect(node, eps_):
            a, y = book[canonical(node)]
            return list(a), list(y), len(y)

        return collect

    return entries, make(train), make(val)

