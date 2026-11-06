"""Build the shared `CaseMemory` of the Group C ports from labelled TRAIN alerts (prereg/external_list.yaml fixed_choices:
"train alerts, p0 + shape buckets, 2 nearest per class").

Owner: T4. The memory takes train alerts only (CaseMemory refuses anything else). The p0 feature is the train p0 of the alert
table unless `p0_by_alert` overrides it: the alert table's train p0 is out-of-fold and mapped through the val calibrator, so it
runs high (mean 0.37 against a label rate of 0.13, audit finding 17) and the query alerts' val p0 would sit far from it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl

from ftm.external.case_memory import CaseMemory, alert_features
from ftm.types import AlertContext


def load_p0_override(path: Path, col: str = "p0") -> dict[str, float]:
    """alert_id -> p0 from a parquet with columns alert_id and `col` (e.g. W2's recalibrated train p0)."""
    df = pl.read_parquet(path, columns=["alert_id", col])  # time-guard: not transactions
    return {str(a): float(p) for a, p in df.iter_rows() if p is not None}


def build_case_memory(alerts: list[AlertContext], labels: list[int], p0_by_alert: dict[str, float] | None = None) -> CaseMemory:
    """CaseMemory of train alerts. `p0_by_alert` replaces the p0 feature of the alerts it names (unnamed alerts keep theirs)."""
    rows = []
    for a, y in zip(alerts, labels, strict=True):
        if p0_by_alert is not None and a.alert_id in p0_by_alert:
            a = replace(a, p0=p0_by_alert[a.alert_id])
        rows.append({"alert_id": a.alert_id, "split": a.split, "label": int(y), "x": alert_features(a)})
    if rows and len({r["label"] for r in rows}) < 2:
        raise ValueError("the case memory needs both classes (nearest laundering and nearest clean case)")
    return CaseMemory(rows)
