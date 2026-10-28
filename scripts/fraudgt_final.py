"""FraudGT in the final run: the dump-only rescoring in the patched checkout, then the calibration into p0 (one command for `configs/final/jobs.yaml`).

Owner: T1 (Abhishek).  Run by `scripts/run_final.py` (step `t1.fraudgt`: FTM_FINAL_RUN=1 and T1_EVAL_TEST=1 are set there, never here):

  python scripts/fraudgt_final.py                 # [--dump DIR] [--checkout DIR] [--data-dir DIR]; defaults from configs/t1_final/fraudgt.yaml
  python scripts/fraudgt_final.py --print         # show the two commands and their environment, run nothing

1. in the FraudGT checkout, `T1_DUMP_ONLY=1 T1_OUT=<dump> python -m fraudGT.main ...`: loads `<dump>/best.pt` (the val run, 5070 Ti), rescores val,
   asserts the stored `pred_val.csv` reproduces (< 1e-3) and writes `<dump>/pred_test.csv`; nothing is trained;
2. `python scripts/fraudgt_scores.py --final --data hi_small --dump <dump>`: stored calibrator on `pred_test.csv`, `data/scores/hi_small/final/p0_fraudgt.parquet`.
Each step's non-zero exit stops the wrapper (its test scores must not be used).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "t1_final" / "fraudgt.yaml"
DEFAULT_DATA_DIR = "~/Documents/t1_external/fraudgt_data"


def build(cfg: dict, dump: str | None = None, checkout: str | None = None, data_dir: str | None = None) -> list[dict]:
    """The two commands as {argv, cwd, env}; `env` holds only what the wrapper adds (T1_EVAL_TEST must come from the caller's environment)."""
    dump = Path(dump or cfg.get("dump_dir") or "data/fraudgt/val_run").expanduser()
    dump = dump if dump.is_absolute() else ROOT / dump
    checkout = Path(checkout or cfg["checkout"]).expanduser()
    data_dir = Path(data_dir or cfg.get("data_dir") or DEFAULT_DATA_DIR).expanduser()
    rescore = [sys.executable, "-m", "fraudGT.main", "--cfg", cfg["config"], "--repeat", "1", "out_dir", f"{dump}/final", "dataset.dir", str(data_dir), "wandb.use", "False"]
    calibrate = [sys.executable, "scripts/fraudgt_scores.py", "--final", "--data", "hi_small", "--dump", str(dump)]
    return [{"argv": rescore, "cwd": str(checkout), "env": {"T1_DUMP_ONLY": "1", "T1_OUT": str(dump)}, "needs": [dump / "best.pt", dump / "pred_val.csv"]},
            {"argv": calibrate, "cwd": str(ROOT), "env": {}, "needs": [dump / "pred_test.csv"]}]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--dump", default=None, help="the val run directory with best.pt and pred_val.csv (default: dump_dir of the config)")
    ap.add_argument("--checkout", default=None, help="the patched FraudGT checkout (default: checkout of the config)")
    ap.add_argument("--data-dir", default=None, help="dataset.dir of FraudGT (the directory that holds AML/)")
    ap.add_argument("--print", action="store_true", help="print the commands and environment, run nothing")
    a = ap.parse_args()
    steps = build(yaml.safe_load(Path(a.config).read_text()), a.dump, a.checkout, a.data_dir)
    if a.print:
        for s in steps:
            print(f"cd {s['cwd']} && {' '.join(f'{k}={v}' for k, v in s['env'].items())} {' '.join(s['argv'])}")
        return
    from ftm import guards

    guards.require_final_run("the FraudGT test scores")
    if os.environ.get("T1_EVAL_TEST") != "1":
        raise SystemExit("T1_EVAL_TEST=1 is not set: it comes from the `fraudgt` step's env in configs/final/jobs.yaml (scripts/run_final.py); this script never sets it")
    for s in steps:
        missing = [str(p) for p in s["needs"] if not p.exists()]
        if missing:
            raise SystemExit(f"missing {missing}: the val run's dump (5070 Ti, ftm/baselines/README.md) or the rescoring step did not write them")
        rc = subprocess.run(s["argv"], cwd=s["cwd"], env={**os.environ, **s["env"]}).returncode
        if rc != 0:
            raise SystemExit(f"{' '.join(s['argv'][:3])} exited with {rc}: do not use FraudGT test scores")


if __name__ == "__main__":
    main()
