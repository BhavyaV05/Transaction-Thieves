"""Expert question specs (configs/questions/expert_specs.yaml): families of templates expanded into numbered
questions. Owner: T4 (Shreyas). T2 turns each spec into a DSL program; until then the specs are the source of truth
for the expert list (plan.md 5.3), and the simple planners (C9) can use their families and kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "configs" / "questions" / "expert_specs.yaml"


@dataclass(frozen=True)
class ExpertQuestion:
    qid: str
    family: str
    kind: str
    text: str
    params: dict
    red_flags: tuple[str, ...]
    typologies: tuple[str, ...]
    sensitivity: int
    expect: str
    status: str = "ok"


def load_families(path: str | Path = SPECS) -> list[dict]:
    return yaml.safe_load(Path(path).read_text())["families"]


def expand(families: list[dict]) -> list[ExpertQuestion]:
    """One question per (family, variant), numbered EQ-001.. in file order."""
    out = []
    for fam in families:
        for params in fam["variants"]:
            out.append(ExpertQuestion(
                qid=f"EQ-{len(out) + 1:03d}", family=fam["family"], kind=fam["kind"],
                text=fam["template"].format(**params), params=dict(params),
                red_flags=tuple(fam["red_flags"]), typologies=tuple(fam.get("typologies") or ()),
                sensitivity=int(fam["sensitivity"]), expect=fam["expect"], status=fam.get("status", "ok"),
            ))  # fmt: skip
    return out


def load_questions(path: str | Path = SPECS) -> list[ExpertQuestion]:
    return expand(load_families(path))
