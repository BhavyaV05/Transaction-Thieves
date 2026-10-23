"""Case memory shared by the Group C ports (Sharma's "similar past cases", Torres's case memory H).

Owner: T4 (Shreyas). Nearest past alerts by feature-space distance, TRAIN alerts only: a port may never see the
label of a validation or test alert (Torres et al. 4.4: "we index only training-set cases").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ftm.types import AlertContext


@dataclass
class CaseMemory:
    """Nearest past alerts by feature-space distance (TRAIN alerts only, never val/test labels).

    `rows`: dicts with alert_id, split, label and a numeric feature vector `x` (e.g. p0 plus the shape vector).
    Features are standardised on the memory itself.
    """

    rows: list[dict] = field(default_factory=list)

    def __post_init__(self):
        bad = {r["split"] for r in self.rows} - {"train"}
        if bad:
            raise ValueError(f"case memory takes train alerts only, got {sorted(bad)}")
        if self.rows:
            k = len(self.rows[0]["x"])
            self._mu = [sum(r["x"][i] for r in self.rows) / len(self.rows) for i in range(k)]
            self._sd = [max(1e-9, math.sqrt(sum((r["x"][i] - self._mu[i]) ** 2 for r in self.rows) / len(self.rows)))
                        for i in range(k)]  # fmt: skip

    def nearest(self, x: list[float], k: int = 2) -> dict:
        if not self.rows:
            return {"error": "no case memory available for this run"}

        def dist(r):
            return math.sqrt(sum(((a - b) / s) ** 2 for a, b, s in zip(r["x"], x, self._sd, strict=True)))

        out = {}
        for name, lab in (("closest_known_laundering", 1), ("closest_known_clean", 0)):
            near = sorted((r for r in self.rows if r["label"] == lab), key=dist)[:k]
            out[name] = [{"case_id": r["alert_id"], "distance": round(dist(r), 3)} for r in near]
        return out


def alert_features(alert: AlertContext) -> list[float]:
    """p0 plus the ordinal shape buckets, the vector CaseMemory compares (no ids)."""
    sh = alert.summary.get("shape") or {}
    keys = ("in_deg", "out_deg", "n_banks", "amount", "span")
    order = {"in_deg": ["0", "1", "2-3", "4-7", "8+"], "out_deg": ["0", "1", "2-3", "4-7", "8+"],
             "n_banks": ["1", "2", "3-4", "5+"], "amount": ["<100", "100-1k", "1k-10k", "10k-100k", "100k+"],
             "span": ["<6h", "6-24h", "1-3d", ">3d"]}  # fmt: skip
    vec = [float(alert.p0)]
    for k in keys:
        v = str(sh.get(k, "")).split(" ")[0]
        vec.append(float(order[k].index(v)) if v in order[k] else -1.0)
    return vec
