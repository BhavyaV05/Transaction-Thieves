"""The final analysis pipeline: H1-H10, Holm, tables, figures, numbers.json, report fill-in.

Owner: T4 (Shreyas).  Spec: prereg/hypotheses.yaml, plan.md 7-9, docs/t4_final_analysis.md.  Run from the repo root.

    # validation dry run (writes outputs/analysis/<run_id>/ incl. fixed_on_val.json)
    python scripts/final_analysis.py --split val --run outputs/final/<run_id>
    python scripts/final_analysis.py --split val --logs outputs/runs/<id>/runlog.parquet outputs/runs/<id2>/runlog.parquet
    # the final run: the caller sets FTM_FINAL_RUN=1 (this script never sets it), and hands over what validation fixed
    FTM_FINAL_RUN=1 python scripts/final_analysis.py --split test --run outputs/final/<run_id> --fixed outputs/analysis/<val_run>/fixed_on_val.json
    # report copies with the numbers in place (never in place), unfilled placeholders listed
    python scripts/final_analysis.py --fill docs/report --numbers outputs/analysis/<run_id>/numbers.json --out outputs/analysis/<run_id>/report_filled
    # the alert table of configs/analysis/inputs.yaml, from T1's alert parquet and the transactions (needs data/)
    python scripts/final_analysis.py --build-alert-table hi_medium --split val --alerts-out outputs/analysis/inputs/alerts_val.parquet

Nothing is printed until the whole run has been loaded and validated. `--split test` needs FTM_FINAL_RUN=1 already set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm import guards  # noqa: E402
from ftm.eval import bootstrap as bs  # noqa: E402
from ftm.eval import final as fn  # noqa: E402
from ftm.eval import final_tables as ft  # noqa: E402

OUTLINE = ROOT / "docs" / "report" / "00_outline.md"


def do_fill(report: str, numbers_path: Path, out: Path, link_base: Path | None) -> int:
    numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    r = ft.fill_report(Path(report), numbers, out, link_base=link_base)
    print(f"filled {sum(r['filled'].values())} placeholders ({len(r['filled'])} ids) into copies in {out}")
    if r["unfilled"]:
        print(f"{len(r['unfilled'])} placeholders could not be filled:")
        for u in r["unfilled"]:
            print(f"  {u['file']}: {{{{TBD:{u['id']}}}}}  ({str(u['reason'])[:110]})")
    (out / "unfilled.json").write_text(json.dumps(r["unfilled"], indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default=None)
    ap.add_argument("--run", default=None, help="run directory with MANIFEST.json (scripts/run_final.py)")
    ap.add_argument("--logs", nargs="+", default=None, help="run-log parquet files, when there is no manifest")
    ap.add_argument("--inputs", default=None, help="configs/analysis/inputs.yaml")
    ap.add_argument("--systems", default=None, help="configs/analysis/systems.yaml")
    ap.add_argument("--fixed", default=None, help="fixed_on_val.json of the validation run (required for --split test)")
    ap.add_argument("--outdir", default=None, help="default outputs/analysis/<run_id>")
    ap.add_argument("--n-boot", type=int, default=bs.N_BOOT)
    ap.add_argument("--seed", type=int, default=bs.SEED)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--fill", default=None, help="report directory (docs/report): write copies with the placeholders filled")
    ap.add_argument("--numbers", default=None, help="with --fill: an existing numbers.json (fill only, no analysis)")
    ap.add_argument("--out", default=None, help="with --fill: directory for the filled copies (default <outdir>/report_filled)")
    ap.add_argument("--build-alert-table", metavar="DATASET", default=None)
    ap.add_argument("--alerts-out", default=None)
    a = ap.parse_args(argv)

    if a.build_alert_table:
        if not a.split or not a.alerts_out:
            ap.error("--build-alert-table needs --split and --alerts-out")
        p = fn.build_alert_table(a.build_alert_table, a.split, a.alerts_out)
        print(f"wrote {p}")
        return 0
    if a.fill and a.numbers:
        if not a.out:
            ap.error("--fill with --numbers needs --out")
        return do_fill(a.fill, Path(a.numbers), Path(a.out), Path(a.numbers).parent)
    if not a.split:
        ap.error("--split is required")
    if not a.run and not a.logs:
        ap.error("give --run (a directory with MANIFEST.json) or --logs")
    if a.split == "test":
        guards.require_final_run("the final analysis on the test split")  # the caller sets FTM_FINAL_RUN=1; this script never does

    try:
        inputs, systems = fn.load_configs(a.inputs, a.systems)
        run = fn.load_run(a.split, a.run, a.logs, systems.get("never_results"))
        panel = fn.build_panel(run, inputs, systems, a.fixed, a.n_boot, a.seed)
    except (fn.AnalysisInputError, fn.Missing) as e:
        print(f"final_analysis: refusing to analyse, nothing computed.\n{e}", file=sys.stderr)
        return 2

    outdir = Path(a.outdir) if a.outdir else ROOT / "outputs" / "analysis" / run.run_id
    out = ft.run_all(panel, inputs, outdir, figures=not a.no_figures, outline=OUTLINE)
    res = out["hypotheses"]
    if a.split == "val":
        fixed = fn.fixed_from_run(panel, res)
        (outdir / "fixed_on_val.json").write_text(json.dumps(fixed, indent=1, default=str), encoding="utf-8")

    print(f"run {run.run_id} ({a.split}): {len(run.jobs)} job(s), {len(panel.y)} alerts, {int(panel.y.sum())} positives, "
          f"{panel.cluster_info['n_clusters']} clusters (largest share {panel.cluster_info['largest_share']:.3f})")
    for wmsg in panel.warnings:
        print(f"WARNING: {wmsg}")
    print(ft.md_table(ft.hypotheses_table(res), ["hypothesis", "tier", "estimate", "95% CI", "p (Holm)", "verdict"]))
    print("\n" + out["power"])
    skipped = {k: v["reason"] for k, v in out["writer"].status.items() if v["status"] == "not_run"}
    if skipped:
        print("\nnot produced:")
        for k, r in skipped.items():
            print(f"  {k}: {r[:140]}")
    print(f"\nwrote {outdir}")
    if a.fill:
        fill_out = Path(a.out) if a.out else outdir / "report_filled"
        do_fill(a.fill, outdir / "numbers.json", fill_out, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
