"""Package existing VALIDATION run logs as a run directory that scripts/final_analysis.py --split val accepts.

Owner: T4.  Reads only validation logs (split == val is checked, anything else aborts); never touches the test split.

    python scripts/build_val_run.py --llm outputs/t3_val_v3/eval --nonllm outputs/analysis/val_runs_nonllm \
        --out outputs/analysis/val_run_v3

--llm     a folder of <arm>/c*/runlog.parquet (T3's validation eval chunks: the 160 askable eval alerts, chunks of 80)
--nonllm  a folder of <arm>/runlog.parquet written by `scripts/final_non_llm.py --split val --alert-ids <the 160 ids>`
Chunks of one arm are concatenated into <out>/logs/<arm>/runlog.parquet. A few arms of T3's val run share system+variant
(ours_compose/harness for the E16 students, the batched and library-by-account arms); VARIANT_OF_DIR gives each its own
variant label, the one configs/analysis/systems.yaml selects on, because load_run refuses two jobs with one key.
MANIFEST.json follows ftm/final/runner.py (manifest_version 1); every job is alerts=subset (the one set of 160 ids).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from ftm.eval.final import ids_sha256  # noqa: E402

VARIANT_OF_DIR = {
    "e16_student_s0": "student_s0",
    "e16_student_s1": "student_s1",
    "e16_student_s2": "student_s2",
    "e16_prompted": "prompted_qwen3_8b",
    "ours_library_account": "library_account",
    "ours_batched_2_1": "batched_2_1",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", type=Path, required=True)
    ap.add_argument("--nonllm", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", default="val_v3")
    a = ap.parse_args(argv)

    groups: list[tuple[str, list[Path], str]] = []
    for d in sorted(p for p in a.llm.iterdir() if p.is_dir()):
        chunks = sorted(d.glob("c*/runlog.parquet"))
        if chunks:
            groups.append((d.name, chunks, "llm"))
    if a.nonllm:
        for d in sorted(p for p in a.nonllm.iterdir() if p.is_dir()):
            if (d / "runlog.parquet").exists():
                groups.append((d.name, [d / "runlog.parquet"], "non_llm"))

    jobs, all_ids = [], None
    for name, chunks, kind in groups:
        df = pl.concat([pl.read_parquet(c) for c in chunks], how="diagonal_relaxed")
        splits = df["split"].unique().to_list()
        if splits != ["val"]:
            raise SystemExit(f"{name}: expected split val only, found {splits}")
        if "label" in df.columns:
            df = df.drop("label")
        if name in VARIANT_OF_DIR:
            df = df.with_columns(pl.lit(VARIANT_OF_DIR[name]).alias("variant"))
        ver = df.filter(pl.col("event") == "verdict")
        ids = sorted(ver["alert_id"].unique().to_list())
        if ver.height != len(ids):
            raise SystemExit(f"{name}: duplicate verdict rows")
        if all_ids is None:
            all_ids = ids
        elif ids != all_ids:
            raise SystemExit(f"{name}: {len(ids)} alerts, not the same set as the other arms ({len(all_ids)})")
        outp = a.out / "logs" / name / "runlog.parquet"
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(outp)
        systems, variants, modes = (df[c].fill_null("").unique().to_list() for c in ("system", "variant", "mode"))
        machines = [m for m in df["machine"].unique().to_list() if m] if "machine" in df.columns else []
        jobs.append({
            "job_id": f"val.{name}", "system": systems[0], "variant": variants[0], "alerts": "subset", "kind": kind,
            "mode": modes[0] or None, "seeds": sorted(int(s) for s in df["seed"].fill_null(0).unique().to_list()),
            "machine_class": "gpu" if kind == "llm" else "cpu", "machine": machines[0] if machines else None, "status": "done",
            "runlog_path": str(outp.relative_to(a.out)), "runlog_sha256": sha256_file(outp), "n_alerts": len(ids),
        })  # fmt: skip
    assert all_ids is not None, "no run logs found"
    (a.out / "subset_ids.txt").write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    manifest = {
        "manifest_version": 1, "run_id": a.run_id, "synthetic": False, "split": "val", "freeze_sha256": None,
        "subset_sha256": ids_sha256(all_ids), "subset_ids_path": "subset_ids.txt", "full_test_sha256": None, "full_test_ids_path": None,
        "note": "validation packaging of T3's outputs/t3_val_v3/eval chunks and scripts/final_non_llm.py --split val logs (scripts/build_val_run.py)",
        "jobs": jobs,
    }  # fmt: skip
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {a.out / 'MANIFEST.json'}: {len(jobs)} jobs, {len(all_ids)} alerts each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
