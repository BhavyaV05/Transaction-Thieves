"""End-to-end demo of the triage system on ONE train / val alert (terminal, or a static HTML page for slides).

    # present recorded REAL runs (no GPU, no LLM): the default way to show it; repeat --replay for a side-by-side comparison
    python scripts/demo.py --replay outputs/demo/V000438/planner --replay outputs/demo/V000438/react_adapted --alert V000438 --pause 1.5
    # run the investigation now against the model endpoint (vLLM): needs FTM_LLM_ENDPOINT / --endpoint, as scripts/run_llm_agent.py
    python scripts/demo.py --data hi_medium --alert V000438 --compare react_adapted --record-dir outputs/demo/V000438
    # scripted stand-in for the LLM, only with the explicit flag, labelled "SCRIPTED MOCK LLM" in the output (no GPU; never a result)
    python scripts/demo.py --data hi_medium --alert V000438 --llm mock --compare react_adapted
    python scripts/demo.py --replay outputs/demo/V000438/planner --alert V000438 --pause 1.5
    # one self-contained HTML page
    python scripts/demo.py --replay outputs/demo/V000438/planner --alert V000438 --html outputs/demo/V000438.html

The run uses the production profile (configs/llm/tuned.yaml: p0 v2, edge scorer v2 evidence view, headroom_v1 table and fusion) and the
escalation threshold of configs/decision/headroom_v1.yaml. Train / val alerts only; test ids (F......) are refused. The true label is
hidden unless --reveal-label.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402

from ftm.demo.alerts import DemoRefusal  # noqa: E402
from ftm.demo.build import build_live, build_replay  # noqa: E402
from ftm.demo.render import export_html, render  # noqa: E402


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alert", required=True, help="a train or val alert id, e.g. V000438 (test ids are refused)")
    ap.add_argument("--data", default="hi_medium")
    ap.add_argument("--replay", type=Path, action="append", default=None,
                    help="render from an existing run dir (runlog.parquet + traces.jsonl ...); repeat for a side-by-side comparison")
    ap.add_argument("--live", action="store_true", help="run the investigation now (the default when no --replay is given)")
    ap.add_argument("--llm", choices=["mock", "endpoint"], default="endpoint",
                    help="endpoint (default) = the served model, as run_llm_agent; mock = the scripted stand-in, explicit and labelled in the output")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backtest", type=Path, default=None, help="back-test table for the replay's answer bins (default: the run's own). A live run always uses the profile's table")
    ap.add_argument("--compare", default="", help="comma-separated adapted arms to run beside ours, e.g. react_adapted")
    ap.add_argument("--mode", default="compose", help="our planner's mode (run_llm_agent --mode)")
    ap.add_argument("--evidence-view", default=None, help="evidence view of the trail (ftm.retrieval.evidence_views.view; default: the profile's)")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps-per-alert", type=float, default=None)
    ap.add_argument("--record-dir", type=Path, default=None, help="keep the recorded run dirs here (<dir>/planner, <dir>/<arm>); default: a temp dir")
    ap.add_argument("--threshold", type=float, default=None,
                    help="escalate at or above this probability (default: configs/decision/headroom_v1.yaml, fitted on val fit+cal)")
    ap.add_argument("--clear-below", type=float, default=None,
                    help="optional: add a human-review band between this and the threshold (default: none, the deployed decision is two-way)")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between steps on a terminal (0 = print all at once)")
    ap.add_argument("--width", type=int, default=None, help="console width (default: the terminal's; 120 for --html)")
    ap.add_argument("--evidence-rows", type=int, default=12, help="evidence rows to show (0 = all)")
    ap.add_argument("--reveal-label", action="store_true", help="show the true label at the end (train / val only)")
    ap.add_argument("--html", type=Path, default=None, help="also write a self-contained HTML page of the walkthrough here")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, console: Console | None = None) -> int:
    a = parse(argv)
    try:
        if a.replay:
            demo = build_replay(a.replay, a.alert, data=a.data if a.data else None, backtest=a.backtest, k=a.k,
                                threshold=a.threshold, clear_below=a.clear_below)
        else:
            import os

            if a.llm == "endpoint" and not (a.endpoint or os.environ.get("FTM_LLM_ENDPOINT")):
                print("no model endpoint: give --endpoint / FTM_LLM_ENDPOINT, or --replay RUN_DIR to show a recorded real run, or --llm mock "
                      "for the scripted stand-in (labelled, never a result)", file=sys.stderr)
                return 2
            rec = a.record_dir or Path(tempfile.mkdtemp(prefix=f"demo-{a.alert}-"))
            demo = build_live(a.alert, data=a.data, llm=a.llm, backtest=a.backtest, compare=[c for c in a.compare.split(",") if c],
                              record_dir=rec, mode=a.mode, k=a.k, evidence_view=a.evidence_view, seed=a.seed,
                              endpoint=a.endpoint, model=a.model, eps_per_alert=a.eps_per_alert, threshold=a.threshold,
                              clear_below=a.clear_below)
    except DemoRefusal as e:
        print(str(e), file=sys.stderr)
        return 2
    console = console or Console(width=a.width)
    render(demo, console, pause=a.pause if console.is_terminal else 0.0, reveal_label=a.reveal_label, evidence_rows=a.evidence_rows)
    if a.html:
        a.html.parent.mkdir(parents=True, exist_ok=True)
        export_html(demo, str(a.html), reveal_label=a.reveal_label, width=a.width or 120, evidence_rows=a.evidence_rows)
        console.print(f"wrote {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
