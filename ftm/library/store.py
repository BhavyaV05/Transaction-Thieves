"""Strategy library store (C12): add, search by situation shape, freeze, save/load with a content hash.

Owner: T3 (Akshat).  Spec: plan.md 6.13, tracks.md 5.5.  SCHEMA AND KEYING FROZEN AT G3; contents frozen by G4.

Two keyings:
  - "shape" (the system): strategies are filed by SituationShape.key(); search = exact key, then the same
    typology (or any) nearest by bucket distance.
  - "account" (E11 ablation only): filed by the receiver's alias. It exists to show that an account-keyed
    memory memorises known criminals; never use it in the main system.
A frozen library refuses `add`, and its hash goes in the run log so the test run provably used the frozen one.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from ftm.library.shape import ANY, SHAPE_SCHEMA_VERSION, SituationShape

LIBRARY_SCHEMA_VERSION = 1
Keying = Literal["shape", "account"]


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    kind: Literal["success", "failure"]
    text: str  # what the planner reads (aliases never appear: the DSL uses slots)
    question: str  # canonical DSL program the strategy is about
    family: str | None
    eps: float
    support: int  # number of past alerts behind it
    mean_gain: float  # mean Brier gain of asking `question` on those alerts (negative = it hurt)
    se_gain: float
    accepted: bool  # the question passed back-testing
    shape: dict = field(default_factory=dict)  # SituationShape.as_dict() ("shape" keying)
    account: str | None = None  # receiver alias ("account" keying, E11 only)


class FrozenLibraryError(RuntimeError):
    pass


class StrategyLibrary:
    def __init__(self, keying: Keying = "shape"):
        self.keying = keying
        self.frozen = False
        self._items: list[Strategy] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def strategies(self) -> list[Strategy]:
        return list(self._items)

    def add(self, s: Strategy) -> None:
        if self.frozen:
            raise FrozenLibraryError("the strategy library is frozen (G4); build a new one for exploratory runs")
        if self.keying == "account" and not s.account:
            raise ValueError("account keying needs Strategy.account")
        self._items.append(s)

    def freeze(self) -> str:
        self.frozen = True
        return self.content_hash()

    def content_hash(self) -> str:
        blob = json.dumps([asdict(s) for s in self._items], sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    # -- search ---------------------------------------------------------------------------------------
    def search(self, shape: SituationShape | dict | None = None, *, account: str | None = None, k: int = 3) -> list[Strategy]:
        """Successes first, then failures; exact key, then nearest same-typology shape."""
        if self.keying == "account":
            hits = [s for s in self._items if account is not None and s.account == account]
            return sorted(hits, key=lambda s: (s.kind != "success", -s.mean_gain))[:k]
        if shape is None:
            return []
        shape = shape if isinstance(shape, SituationShape) else SituationShape.from_dict(shape)

        def score(s: Strategy) -> tuple:
            other = SituationShape.from_dict(s.shape)
            same_typ = shape.typology in (ANY, other.typology) or other.typology == ANY
            if not same_typ:
                return (math.inf,)
            exact = other.key() == shape.key()
            dist = sum(abs(a - b) for a, b in zip(shape.vector(), other.vector(), strict=True))
            return (0 if exact else 1, dist, s.kind != "success", -s.mean_gain)

        ranked = sorted((score(s), i, s) for i, s in enumerate(self._items))
        return [s for sc, _, s in ranked if sc[0] != math.inf][:k]

    def search_dicts(self, shape: dict, k: int = 3) -> list[dict]:
        """What ToolBox.search_library returns to an investigator: text plus provenance."""
        return [{"text": s.text, "strategy_id": s.strategy_id, "kind": s.kind, "question": s.question}
                for s in self.search(shape, k=k)]  # fmt: skip

    # -- persistence ------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> str:
        h = self.content_hash()
        doc = {"library_schema_version": LIBRARY_SCHEMA_VERSION, "shape_schema_version": SHAPE_SCHEMA_VERSION,
               "keying": self.keying, "frozen": self.frozen, "content_hash": h,
               "strategies": [asdict(s) for s in self._items]}  # fmt: skip
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
        return h

    @classmethod
    def load(cls, path: str | Path) -> StrategyLibrary:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc["library_schema_version"] != LIBRARY_SCHEMA_VERSION or doc["shape_schema_version"] != SHAPE_SCHEMA_VERSION:
            raise ValueError(f"library {path} has schema {doc['library_schema_version']}/{doc['shape_schema_version']}")
        lib = cls(doc["keying"])
        lib._items = [Strategy(**s) for s in doc["strategies"]]
        if lib.content_hash() != doc["content_hash"]:
            raise ValueError(f"library {path} content hash mismatch: edited by hand?")
        lib.frozen = doc["frozen"]
        return lib


class LibraryToolBox:
    """Wraps any ToolBox so `search_library` answers from `library` (until the harness wires C12 itself).

    Account keying (E11) needs the receiver alias, which the wrapper takes from the shape dict's `receiver` key.
    """

    def __init__(self, tools, library: StrategyLibrary, k: int = 3):
        self._tools, self.library, self.k = tools, library, k

    def __getattr__(self, name):
        return getattr(self._tools, name)

    def local_graph(self, query, as_of):
        return self._tools.local_graph(query, as_of)

    def ask_bank(self, bank, program, eps):
        return self._tools.ask_bank(bank, program, eps)

    def lookup_rules(self, query):
        return self._tools.lookup_rules(query)

    def search_library(self, shape: dict) -> list[dict]:
        if self.library.keying == "account":
            return [{"text": s.text, "strategy_id": s.strategy_id, "kind": s.kind, "question": s.question}
                    for s in self.library.search(account=shape.get("receiver"), k=self.k)]  # fmt: skip
        return self.library.search_dicts(shape, k=self.k)
