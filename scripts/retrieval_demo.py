"""Retrieval demo (G4): the four evidence views of one alert, aliased and time-ordered.

Owner: T1 (Abhishek).  Spec: tracks.md 3.8, 3.9 week 9.

  python scripts/retrieval_demo.py --data hi_medium --alert V000553
  python scripts/retrieval_demo.py --data hi_small --alert V000123 --view scorer_pcst --with-labels

Prints, per view, the trail an LLM would read (`ftm.retrieval.evidence_views.view`: aliased accounts, no raw id, no label, nothing after the
alert time) and its size. `--with-labels` (off by default) additionally scores each view against the pattern file: pattern-edge recall among the
candidates and the one-time-respecting-trail check. Labels are used for that scoring only and never printed next to an edge. Train and val alerts only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.data import io  # noqa: E402
from ftm.retrieval import evidence_views as ev  # noqa: E402
from ftm.retrieval import pcst  # noqa: E402

FOUR = ("amount", "scorer", "scorer_pcst", "long_prompt")


def score(data: str, alert_id: str, v: ev.EvidenceView) -> dict:
    """Pattern-edge recall and trail check of one view (labels, scoring only)."""
    cfg = io.load_config(data)
    tx_id, bank, as_of = ev.resolve_alert(alert_id, data)
    cand = ev.candidates(cfg, bank, as_of)
    pat = io.scan_patterns(cfg["parquet_dir"], split=["train", "val"]).select("tx_id", "attempt_id").collect()
    att = pat.filter(pl.col("tx_id") == tx_id)["attempt_id"]
    same = set(pat.filter(pl.col("attempt_id") == att[0])["tx_id"].to_list()) if len(att) else set()
    tx = cand["tx_id"].to_numpy()
    rel = np.isin(tx, list(same - {tx_id}))
    i = int(np.flatnonzero(tx == tx_id)[0])
    chosen = np.isin(tx, v.tx_ids)
    src, dst = cand["src_id"].to_numpy(), cand["dst_id"].to_numpy()
    t = cand["timestamp"].dt.epoch("s").to_numpy().astype("f8")
    return {"pattern_edges_in_window": int(rel.sum()), "pattern_edges_in_view": int((rel & chosen).sum()),
            "recall": float((rel & chosen).sum() / rel.sum()) if rel.sum() else None,
            "one_time_respecting_trail": pcst.is_one_trail(src, dst, t, i, chosen)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hi_small")
    ap.add_argument("--alert", required=True, help="alert id of the train or val alert set, e.g. V000553")
    ap.add_argument("--view", default=None, choices=[*ev.NAMES, None], help="one view (default: the four of E8)")
    ap.add_argument("--k", type=int, default=ev.K)
    ap.add_argument("--with-labels", action="store_true", help="score the views against the pattern file (off by default)")
    ap.add_argument("--full", action="store_true", help="print every line of the long-prompt view (default: first and last 3)")
    a = ap.parse_args()
    names = (a.view,) if a.view else FOUR
    for n in names:
        v = ev.view(a.alert, n, a.k, data=a.data)
        lines = v.text.split("\n")
        print(f"=== {n}: {len(v.tx_ids)} of {v.n_candidates} candidate edges in the 72 h window, about {v.tokens_est} tokens ===")
        if n == "long_prompt" and not a.full and len(lines) > 8:
            print("\n".join([*lines[:3], f"... ({len(lines) - 6} more edges, the newest {len(lines)} of {v.n_candidates} fit the cap) ...", *lines[-3:]]))
        else:
            print(v.text)
        if a.with_labels:
            print("labels (scoring only):", score(a.data, a.alert, v))
        print()


if __name__ == "__main__":
    main()
