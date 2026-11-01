"""Find and load ONE train / val alert for the demo. Test alerts are refused (their table is never opened).

Owner: T3 (Akshat).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class DemoRefusal(SystemExit):
    """The demo refuses this request (a test id, or an alert that is not in train / val)."""


def runner():
    """scripts/run_llm_agent.py as a module: the demo reuses its alert loader, toolbox and system builders."""
    s = str(ROOT / "scripts")
    if s not in sys.path:
        sys.path.insert(0, s)
    return importlib.import_module("run_llm_agent")


def refuse_if_test(alert_id: str, split: str | None = None) -> None:
    if split == "test" or str(alert_id).upper().startswith("F"):  # test alerts are F000123 (ftm.intake.alerts.test_alert_ids)
        raise DemoRefusal(f"refusing {alert_id}: the demo uses train / val alerts only, never the test split")


def find_alert(data: str, alert_id: str) -> tuple[str, str | None]:
    """(split, val_part) of the alert, looked up in the train and val alert tables only."""
    import polars as pl

    from ftm.data import io

    refuse_if_test(alert_id)
    cfg = io.load_config(data)
    for split in ("train", "val"):
        path = ROOT / io.alerts_dir(cfg, None) / f"{split}.parquet"
        if not path.exists():
            continue
        row = pl.read_parquet(path).filter(pl.col("alert_id") == alert_id)  # time-guard: not transactions
        if row.height:
            return split, (str(row["val_part"][0]) if split == "val" else None)
    raise DemoRefusal(f"alert {alert_id!r} is not a train or val alert of {data} (test ids are refused)")


def load_alert(data: str, alert_id: str, *, k: int = 40, evidence_view: str | None = "scorer_pcst"):
    """(AlertContext, label, evidence_view actually used). The evidence view falls back to the week-2 stub trail when the
    scorer behind it is not available."""
    split, part = find_alert(data, alert_id)
    refuse_if_test(alert_id, split)
    r = runner()
    used = evidence_view
    try:
        got = r.data_alerts(data, split, 1, 0, k, part, {alert_id}, 0, evidence_view)
    except FileNotFoundError:
        used = None
        got = r.data_alerts(data, split, 1, 0, k, part, {alert_id}, 0, None)
    ctx, label = got[0]
    return ctx, label, used
