"""MIPROv2 optimisation of the DSPy ReAct baseline on validation `fit` alerts (T3, item "DSPy-tuned ReAct", plan.md 6.17 / 7.2).

Owner: T3 (Akshat).  Tuning parity: only val `fit` alerts, objective = 1 - Brier of the program's probability, Gemma 4 12B
(the E18 backbone) through our client; the harness world is the one every val run uses (askable alerts, the real partner banks).
The optimised program is saved OUTSIDE git (`outputs/t3_dspy/`); its sha256 goes into the summary of every run that loads it
(`run_llm_agent.py --system dspy_react --dspy-program <file>`), and the final run's dspy_react job loads the same file
(configs/final/jobs.yaml). MIPROv2's own LLM calls and wall time are logged to `compute_log.jsonl` beside the program.

    FTM_LLM_ENDPOINT=http://127.0.0.1:8000/v1 FTM_LLM_API_KEY=... python scripts/optimize_dspy.py --model gemma4-12b --n 40

(the key comes from the environment only). Nothing here reads val `eval` or test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_llm_agent as rla  # noqa: E402

from ftm.external.dspy_react import optimize  # noqa: E402
from ftm.llm.client import HttpChatClient  # noqa: E402
from ftm.types import Limits  # noqa: E402


def fit_examples(data: str, n: int, seed: int, k: int, toolbox: str = "harness") -> list[dict]:
    """The tuning set: n val `fit` alerts (seeded order) the harness can ask about, with their labels and ONE shared toolbox."""
    tools, variant = rla.make_toolbox(toolbox, seed, data, "val")
    only = rla.askable_ids(tools) if variant == "harness" else None
    alerts = rla.data_alerts(data, "val", n, seed, k, "fit", only)
    world = rla.modelled_banks(tools) if variant == "harness" else None
    out = []
    for ctx, label in alerts:
        if world is not None:
            ctx = replace(ctx, summary={**ctx.summary, "partner_banks": [b for b in ctx.summary["partner_banks"] if str(b) in world]})
        out.append({"alert": ctx, "tools": tools, "label": label, "part": "fit"})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="hi_medium")
    ap.add_argument("--n", type=int, default=40, help="val `fit` alerts to tune on")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "t3_dspy")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    llm = HttpChatClient.from_env(endpoint=a.endpoint, model=a.model,
                                  default_extra={"chat_template_kwargs": {"enable_thinking": False}})  # thinking off, as everywhere
    examples = fit_examples(a.data, a.n, a.seed, a.k)
    t0 = time.perf_counter()
    program = optimize(llm, examples, limits=Limits(), auto=a.auto, seed=a.seed, compute_log=a.out / "compute_log.jsonl")
    f = a.out / "react_miprov2.json"
    program.save(str(f))
    import dspy

    meta = {"program": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "dspy": dspy.__version__, "auto": a.auto,
            "seed": a.seed, "n_examples": len(examples), "part": "fit", "model": getattr(llm, "model", None), "data": a.data,
            "alert_ids_sha256": hashlib.sha256("\n".join(e["alert"].alert_id for e in examples).encode()).hexdigest(),
            "wall_seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
