"""The final analysis: load the run logs of a (validation or final) run, attach labels, and test H1-H10 as pre-registered.

Owner: T4 (Shreyas).  Spec: prereg/hypotheses.yaml, prereg/external_list.yaml, plan.md 7, 8, 9; docs/t4_final_analysis.md.
Script: scripts/final_analysis.py.  Tables, figures, numbers.json and the report fill-in: ftm/eval/final_tables.py.

The same code runs on a validation dry run and on the final test run, so the test tables and figures are pre-built and the
final numbers arrive "in place, not as new code" (docs/schedule_nov10.md).

Contract:
  * Run logs (ftm.eval.runlog schema) never carry labels into the analysis: a `label` column is dropped on load. Labels, clusters
    and the upper limits come from the alert table and score tables of configs/analysis/inputs.yaml.
  * A run is loaded and validated as a whole (`load_run`) before any number is computed; every problem is collected and
    raised together as `AnalysisInputError`.
  * `--split test` is refused unless FTM_FINAL_RUN=1 is already set (`ftm.guards.require_final_run`); `--split val` refuses any
    log row whose split is not val.
  * Every hypothesis is either tested or reported `not_run` with its reason. Holm runs within a family (primary, secondary);
    a `not_run` member keeps its slot in the family with p = 1 (the family size is pre-registered).
  * Values that prereg/ marks `fix_on_val` (best simple planner, equivalence margin, H9 margin, best ported external, hub cap)
    are derived on validation and written to fixed_on_val.json; the test run reads that file and refuses to derive them itself.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from ftm import guards
from ftm.eval import bootstrap as bs
from ftm.eval import metrics as mt
from ftm.eval import runlog as rl
from ftm.eval import stats as st
from ftm.fusion import fuse

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "analysis"
PRIMARY = ["H1", "H2", "H3", "H4", "H9"]
SECONDARY = ["H6", "H7", "H10"]
EXPLORATORY = ["H5", "H8"]
ALPHA = 0.05
UPPERS = ("gfp_see_everything", "best_sota_see_everything")
MDE_MULTIPLIER = 2.80  # z(0.975) + z(0.80): two-sided alpha 0.05, power 0.8
PLAN_MDE = {"n": 4000, "brier": 0.011}  # plan.md 9.3: with ~4,000 test alerts, Brier improvements of about 0.011


class AnalysisInputError(ValueError):
    """The run or its inputs are incomplete or inconsistent; nothing is analysed."""


class Missing(Exception):
    """A role or input a hypothesis needs is not available: the hypothesis is `not_run` with this reason."""


# --------------------------------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_configs(inputs: str | Path | None = None, systems: str | Path | None = None) -> tuple[dict, dict]:
    return load_yaml(inputs or CONFIG_DIR / "inputs.yaml"), load_yaml(systems or CONFIG_DIR / "systems.yaml")


def resolve_path(p: str | Path | None, split: str = "val", base: Path | None = None) -> Path | None:
    if p is None:
        return None
    q = Path(str(p).replace("{split}", split))
    if q.is_absolute():
        return q
    for root in ([base] if base else []) + [Path.cwd(), ROOT]:
        if (root / q).exists():
            return root / q
    return (base or ROOT) / q


def ids_sha256(ids) -> str:
    """sha256 of the sorted alert ids joined by newlines (the subset hash; prereg/subset_manifest.txt)."""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------------------
# manifest and run logs
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class JobSpec:
    path: Path
    system: str | None = None
    variant: str | None = None
    mode: str | None = None
    alert_set: str = "any"  # subset | full_test | any (no manifest)
    seeds: list[int] | None = None
    sha256: str | None = None
    subset_sha256: str | None = None
    job_id: str | None = None
    kind: str | None = None
    n_alerts: int | None = None


def _first(d: dict, names, default=None):
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return default


def _norm_alert_set(v) -> str:
    s = str(v or "").lower()
    if "subset" in s or "1000" in s or "1k" in s:
        return "subset"
    if "full" in s or "test" in s or s == "all":
        return "full_test"
    return "any"


DONE_STATUSES = {"done", "complete", "completed", "ok", "success", "finished", "present"}


def parse_manifest(path: str | Path) -> tuple[list[JobSpec], dict]:
    """Tolerant reader of outputs/final/<run_id>/MANIFEST.json (or MANIFEST.partial.json): a list of jobs, or {"jobs": [...], run keys}.

    Jobs that are not done yet (status missing/pending/failed, or no run-log path) are not an error: they come back in
    head["missing_jobs"] as {job_id, system, variant, alerts, status}, so the analysis marks those arms and goes on."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    head = {} if isinstance(raw, list) else {k: v for k, v in raw.items() if k != "jobs"}
    jobs_raw = raw if isinstance(raw, list) else _first(raw, ["jobs", "runs", "items"], [])
    if isinstance(jobs_raw, dict):
        jobs_raw = [{"name": k, **v} for k, v in jobs_raw.items()]
    jobs, missing = [], []
    for j in jobs_raw:
        p = _first(j, ["runlog", "run_log", "runlog_path", "log", "log_path", "path", "parquet", "file"])
        status = str(j.get("status") or "").lower()
        jid = _first(j, ["job_id", "name", "id"])
        if status and status not in DONE_STATUSES or p is None:
            missing.append({"job_id": jid, "system": j.get("system"), "variant": j.get("variant"),
                            "alerts": _norm_alert_set(_first(j, ["alert_set", "alerts", "eval_set", "subset"])),
                            "status": status or "no run-log path"})  # fmt: skip
            continue
        seeds = _first(j, ["seeds", "seed"])
        seeds = None if seeds is None else [int(s) for s in (seeds if isinstance(seeds, list) else [seeds])]
        mode = j.get("mode")
        jobs.append(JobSpec(resolve_path(p, base=path.parent) or Path(p), j.get("system"), j.get("variant"), mode,
                            _norm_alert_set(_first(j, ["alert_set", "alerts", "eval_set", "subset"])), seeds,
                            _first(j, ["sha256", "runlog_sha256"]), j.get("subset_sha256"), jid, j.get("kind"), j.get("n_alerts")))  # fmt: skip
    head["missing_jobs"] = missing
    head["job_table"] = [{"job_id": js.job_id, "system": js.system, "variant": js.variant, "alerts": js.alert_set, "status": "present",
                          "n_alerts": js.n_alerts} for js in jobs] + [{**m} for m in missing]  # fmt: skip
    return jobs, head


@dataclass
class Run:
    split: str
    run_id: str
    jobs: list[JobSpec]
    log: pl.DataFrame  # all jobs, `label` dropped; extra columns: _alert_set, _job
    manifest: dict | None
    warnings: list[str] = field(default_factory=list)
    subset_sha256: str | None = None
    subset_ids: tuple[str, ...] = ()
    missing_jobs: list[dict] = field(default_factory=list)  # arms of the manifest that are not done yet (partial run)
    job_table: list[dict] = field(default_factory=list)
    synthetic: bool = False  # a rehearsal run (manifest synthetic=true): never a result, labels may sit in the logs
    synth_labels: pl.DataFrame | None = None


def _read_log(job: JobSpec, idx: int, keep_label: bool = False) -> pl.DataFrame:
    df = pl.read_parquet(job.path)  # time-guard: not transactions
    problems = rl.validate(df.columns, df["event"].unique().to_list() if "event" in df.columns else [])
    if problems:
        raise AnalysisInputError(f"{job.path}: {'; '.join(problems)}")
    if keep_label and "label" in df.columns:
        df = df.rename({"label": "_label"})  # synthetic rehearsals only: no alert table exists for them
    df = df.drop([c for c in ("label",) if c in df.columns])  # labels are attached by the analysis, never taken from a log
    fills = {"variant": "", "mode": ""}
    for c, v in fills.items():
        df = df.with_columns(pl.col(c).fill_null(v)) if c in df.columns else df.with_columns(pl.lit(v).alias(c))
    df = df.with_columns(pl.col("seed").fill_null(0) if "seed" in df.columns else pl.lit(0).alias("seed"))
    return df.with_columns(pl.lit(job.alert_set).alias("_alert_set"), pl.lit(idx).alias("_job"))


def load_run(split: str, run_dir: str | Path | None = None, logs: list[str | Path] | None = None,
             never_results: list[str] | None = None) -> Run:  # fmt: skip
    """Read the manifest (or the given logs), every run log, and check the run as a whole. Raises AnalysisInputError."""
    if split == "test":
        guards.require_final_run("the final analysis on the test split")
    elif split != "val":
        raise AnalysisInputError("--split is val or test")
    problems: list[str] = []
    warnings: list[str] = []
    manifest: dict | None = None
    run_id = "adhoc"
    if run_dir is not None:
        mpath = Path(run_dir) / "MANIFEST.json"
        if not mpath.exists() and (Path(run_dir) / "MANIFEST.partial.json").exists():
            mpath = Path(run_dir) / "MANIFEST.partial.json"  # arms finish one at a time (schedule phase C)
            warnings.append("only MANIFEST.partial.json exists: the run is not complete, missing arms are marked")
        if mpath.exists():
            jobs, manifest = parse_manifest(mpath)
            run_id = str(manifest.get("run_id") or Path(run_dir).name)
        elif not logs:
            raise AnalysisInputError(f"{mpath} not found and no --logs given")
        else:
            jobs, run_id = [], Path(run_dir).name
    else:
        jobs = []
    if logs:
        if jobs:
            raise AnalysisInputError("give either a run directory with a MANIFEST.json or --logs, not both")
        jobs = [JobSpec(Path(p)) for p in logs]
        run_id = run_id if run_dir is not None else "logs"
    missing_jobs = list((manifest or {}).get("missing_jobs") or [])
    if not jobs:
        raise AnalysisInputError("no jobs are done yet: the manifest lists "
                                 + (f"{len(missing_jobs)} job(s), none with a run log" if missing_jobs else "none") + " and no --logs were given")
    synthetic = bool((manifest or {}).get("synthetic"))
    if missing_jobs:
        warnings.append(f"{len(missing_jobs)} job(s) of the manifest are not done and are marked missing: "
                        + ", ".join(str(m.get("job_id") or f"{m.get('system')}/{m.get('variant')}") for m in missing_jobs[:12])
                        + (" ..." if len(missing_jobs) > 12 else ""))  # fmt: skip
    frames = []
    for i, j in enumerate(jobs):
        if not j.path.exists():
            problems.append(f"job {i} ({j.system}/{j.variant}): run log {j.path} is missing")
            continue
        if j.sha256 and file_sha256(j.path).lower() != str(j.sha256).lower():
            problems.append(f"job {i} ({j.system}/{j.variant}): sha256 of {j.path.name} differs from the manifest")
            continue
        frames.append(_read_log(j, i, keep_label=synthetic))
    if problems:
        raise AnalysisInputError("run is incomplete:\n  - " + "\n  - ".join(problems))
    log = pl.concat(frames, how="diagonal_relaxed")
    ok_splits = {split, "synthetic"} if synthetic else {split}  # a rehearsal run logs split=synthetic and is analysed like val
    bad_split = log.filter(~pl.col("split").is_in(sorted(ok_splits)))["split"].unique().to_list()
    if bad_split:
        raise AnalysisInputError(f"--split {split} but the logs hold rows of split {bad_split}: a val analysis never reads test rows")
    if split == "test" and never_results:
        bad = log.filter(pl.col("variant").is_in(never_results))["variant"].unique().to_list()
        if bad:
            raise AnalysisInputError(f"variants {bad} are never results (random answers or mock banks): refused on --split test")
    problems += _check_jobs(log, jobs)
    subset_ids: tuple[str, ...] = ()
    subset_hash = None
    sub = [i for i, j in enumerate(jobs) if j.alert_set == "subset"]
    if sub:
        sets = [frozenset(log.filter((pl.col("_job") == i) & (pl.col("event") == "verdict"))["alert_id"].to_list()) for i in sub]
        if len(set(sets)) > 1:
            problems.append("subset jobs ran on different alert sets (subset hash must be equal across jobs): "
                            + ", ".join(f"job {i}: {len(s)} alerts" for i, s in zip(sub, sets, strict=True)))
        subset_ids = tuple(sorted(sets[0]))
        hashes = {jobs[i].subset_sha256 for i in sub if jobs[i].subset_sha256}
        top = (manifest or {}).get("subset_sha256")
        hashes |= {top} if top else set()
        if len(hashes) > 1:
            problems.append(f"subset_sha256 differs across the manifest's jobs: {sorted(hashes)}")
        subset_hash = next(iter(hashes), None)
        mine = ids_sha256(subset_ids)
        if subset_hash and subset_hash != mine:
            warnings.append(f"manifest subset_sha256 {subset_hash[:12]} differs from the sha256 of the sorted alert ids found in the "
                            f"logs ({mine[:12]}); the hash definitions may differ, the alert sets themselves agree across jobs")
    problems += _check_against_id_files(log, jobs, manifest, Path(run_dir) if run_dir is not None else None, warnings)
    if problems:
        raise AnalysisInputError("run is inconsistent:\n  - " + "\n  - ".join(problems))
    synth_labels = None
    if synthetic and "_label" in log.columns:
        synth_labels = (log.filter((pl.col("event") == "verdict") & pl.col("_label").is_not_null())
                        .group_by("alert_id").agg(pl.col("_label").first().cast(pl.Int64).alias("label"), pl.col("p0").first()))  # fmt: skip
    if "_label" in log.columns:
        log = log.drop("_label")
    table = (manifest or {}).get("job_table") or []
    return Run(split, run_id, jobs, log, manifest, warnings, subset_hash or (ids_sha256(subset_ids) if subset_ids else None),
               subset_ids, missing_jobs, table, synthetic, synth_labels)  # fmt: skip


def _read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _check_against_id_files(log: pl.DataFrame, jobs: list[JobSpec], manifest: dict | None, run_dir: Path | None, warnings: list[str]) -> list[str]:
    """MANIFEST subset_ids_path / full_test_ids_path (relative to the run dir): their hashes must be the manifest's, and every subset
    job must have run exactly the frozen subset (and every full_test job exactly the full test set)."""
    out: list[str] = []
    if not manifest or run_dir is None:
        return out
    for key, hkey, kind in (("subset_ids_path", "subset_sha256", "subset"), ("full_test_ids_path", "full_test_sha256", "full_test")):
        rel = manifest.get(key)
        if not rel:
            continue
        f = Path(rel) if Path(rel).is_absolute() else run_dir / rel
        if not f.exists():
            warnings.append(f"manifest {key} {rel} not found: the {kind} alert set could not be checked against the frozen ids")
            continue
        ids = _read_ids(f)
        if manifest.get(hkey) and ids_sha256(ids) != manifest[hkey]:
            out.append(f"{rel} does not hash to the manifest's {hkey} ({ids_sha256(ids)[:12]} vs {str(manifest[hkey])[:12]})")
            continue
        want = set(ids)
        inj = set(log.filter(pl.col("injection_id").is_not_null())["_job"].unique().to_list()) if "injection_id" in log.columns else set()
        for i, j in enumerate(jobs):
            if j.alert_set != kind or i in inj:  # injection jobs run on their own attack items, not on test alerts
                continue
            got = set(log.filter((pl.col("_job") == i) & (pl.col("event") == "verdict"))["alert_id"].to_list())
            if got != want:
                out.append(f"job {i} ({j.job_id or j.system}): its {len(got)} alerts are not the frozen {kind} ids ({len(want)}): "
                           f"{len(want - got)} missing, {len(got - want)} extra")  # fmt: skip
    return out


def _check_jobs(log: pl.DataFrame, jobs: list[JobSpec]) -> list[str]:
    out = []
    v = log.filter(pl.col("event") == "verdict")
    if v.is_empty():
        return ["the run logs hold no verdict rows"]
    for c in ("alert_id", "p0"):
        if c not in v.columns:
            out.append(f"verdict rows lack the column {c!r}")
    if out:
        return out
    dup = v.group_by(["_job", "system", "variant", "mode", "seed", "alert_id"]).len().filter(pl.col("len") > 1)
    if dup.height:
        out.append(f"{dup.height} (system, variant, seed, alert) have more than one verdict row, e.g. {dup.row(0)[:6]}")
    for i, j in enumerate(jobs):
        jv = v.filter(pl.col("_job") == i)
        if jv.is_empty():
            out.append(f"job {i} ({j.system}/{j.variant}): its run log has no verdict rows")
            continue
        # the manifest names are the plan's (e.g. system ours, variant compose); the logs' own columns are what is analysed
        if j.n_alerts is not None and jv["alert_id"].n_unique() != int(j.n_alerts):
            out.append(f"job {i} ({j.job_id or j.system}): manifest n_alerts {j.n_alerts} but the log has {jv['alert_id'].n_unique()} alerts with a verdict")
        if j.seeds is not None and set(j.seeds) != set(jv["seed"].to_list()):
            out.append(f"job {i} ({j.system}/{j.variant}): manifest seeds {sorted(j.seeds)} but the log has {sorted(set(jv['seed'].to_list()))}")
        # every (system, variant, seed) of the job must cover the same alerts
        cover = jv.group_by(["system", "variant", "mode", "seed"]).agg(pl.col("alert_id").n_unique().alias("n"))
        if cover["n"].n_unique() > 1:
            out.append(f"job {i} ({j.system}/{j.variant}): the seeds/arms cover different numbers of alerts: {cover['n'].to_list()}")
    # one-bank p0 must be the same number for the same alert in every job
    p0 = v.filter(pl.col("p0").is_not_null()).group_by("alert_id").agg((pl.col("p0").max() - pl.col("p0").min()).alias("d"))
    if p0.height and p0["d"].max() > 1e-6:
        out.append(f"p0 differs between jobs for {int((p0['d'] > 1e-6).sum())} alerts (every system starts from the same one-bank score)")
    return out


# --------------------------------------------------------------------------------------------------------------------
# alert table and score sources
# --------------------------------------------------------------------------------------------------------------------
def join_alert_table(alerts: pl.DataFrame, tx: pl.DataFrame, patterns: pl.DataFrame | None = None) -> pl.DataFrame:
    """Alert parquet of T1 (alert_id, tx_id, bank, p0, label, ...) + the transactions' account ids (tx_id, src_id, dst_id)."""
    out = alerts.join(tx.select("tx_id", "src_id", "dst_id"), on="tx_id", how="left")
    if patterns is not None and "attempt_id" not in out.columns:
        out = out.join(patterns.select("tx_id", "attempt_id").unique("tx_id"), on="tx_id", how="left")
    return out.with_columns(pl.col("bank").alias("home_bank")) if "bank" in out.columns and "home_bank" not in out.columns else out


def build_alert_table(ds: str, split: str, out: str | Path) -> Path:
    """Write the alert table for `split` from T1's alert parquet and the transaction ids (needs data/; val or test)."""
    from ftm.data import io

    if split == "test":
        guards.require_final_run("the test alert table")
    cfg = io.load_config(ds)
    alerts = pl.read_parquet(Path(cfg["alerts_dir"]) / f"{split}.parquet")  # time-guard: not transactions
    tx = io.scan_transactions(cfg["parquet_dir"], split=split, columns=["tx_id", "src_id", "dst_id", "day"]).collect()
    pat = io.scan_patterns(cfg["parquet_dir"], split=split).select("tx_id", "attempt_id").collect()
    df = join_alert_table(alerts, tx.select("tx_id", "src_id", "dst_id"), pat)
    if split == "test":  # headline = bulk days, the sparse tail is reported separately (decision 4)
        from ftm.data.splits import test_part

        df = df.join(tx.select("tx_id", "day"), on="tx_id", how="left")
        df = df.with_columns(test_part(cfg).alias("test_part"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out


def _filter(df: pl.DataFrame, flt: dict | None) -> pl.DataFrame:
    for col, vals in (flt or {}).items():
        if col not in df.columns:
            raise AnalysisInputError(f"alert table has no column {col!r} needed by the headline filter")
        df = df.filter(pl.col(col).is_in(vals if isinstance(vals, list) else [vals]))
    return df


def load_alerts(cfg: dict, split: str, part: str = "headline") -> pl.DataFrame:
    a = cfg.get("alerts") or {}
    path = resolve_path(a.get("path"), split)
    if path is None:
        raise Missing("inputs.yaml alerts.path is not set (labels, clusters and account ids come from the alert table)")
    if not path.exists():
        raise AnalysisInputError(f"alert table {path} not found")
    df = pl.read_parquet(path)  # time-guard: not transactions
    for c in ("alert_id", "label"):
        if c not in df.columns:
            raise AnalysisInputError(f"alert table {path} lacks the column {c!r}")
    if df["alert_id"].n_unique() != df.height:
        raise AnalysisInputError("alert_id is not unique in the alert table")
    for c in a.get("account_cols", ["src_id", "dst_id"]):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    if "attempt_id" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("attempt_id"))
    if df["label"].null_count():
        raise AnalysisInputError("the alert table has null labels")
    flt = (a.get(part) or {}).get(split) if isinstance(a.get(part), dict) and split in (a.get(part) or {}) else a.get(part)
    return _filter(df, flt if isinstance(flt, dict) else {}).sort("alert_id")


def load_score(spec: dict, alerts: pl.DataFrame, split: str) -> np.ndarray | None:
    """Per-alert probabilities of a score table aligned to `alerts` (NaN where the table has none); None if no path."""
    path = resolve_path(spec.get("path"), split)
    if path is None:
        return None
    if not path.exists():
        raise AnalysisInputError(f"score table {path} not found")
    df = pl.read_parquet(path)  # time-guard: not transactions
    for col, val in (spec.get("filter") or {}).items():
        if col not in df.columns:
            raise AnalysisInputError(f"{path.name}: no column {col!r} to filter on")
        df = df.filter(pl.col(col).is_in(val if isinstance(val, list) else [val]))
    key, value = list(spec.get("key") or ["alert_id"]), spec.get("value", "p0")
    for c in key + [value]:
        if c not in df.columns:
            raise AnalysisInputError(f"{path.name}: no column {c!r}")
    for c in key:
        if c not in alerts.columns:
            raise AnalysisInputError(f"the alert table lacks {c!r}, the join key of {path.name}")
    df = df.select(key + [pl.col(value).alias("_v")]).unique(key, keep="first")
    j = alerts.select(key).join(df, on=key, how="left")
    return j["_v"].cast(pl.Float64).to_numpy().astype("f8")


# --------------------------------------------------------------------------------------------------------------------
# per-system scores
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Key:
    system: str
    variant: str
    mode: str
    alert_set: str

    def label(self) -> str:
        return "/".join(x for x in (self.system, self.variant, self.mode, self.alert_set) if x and x != "any")


@dataclass
class Scores:
    """Per-alert results of one system, aligned to the master alert order (NaN where the system has no result)."""

    name: str
    se_e2e: np.ndarray
    se_qo: np.ndarray | None
    p_e2e: np.ndarray
    p_qo: np.ndarray | None
    no_verdict_rate: float = float("nan")
    cost: dict = field(default_factory=dict)
    stop_reasons: dict = field(default_factory=dict)
    key: Key | None = None
    source: str = "runlog"

    def present(self) -> np.ndarray:
        return np.isfinite(self.se_e2e)


def _fusion_from(params: dict | None) -> fuse.Fusion:
    if not params:
        return fuse.Fusion("naive")
    f = fuse.Fusion(params.get("mode", "naive"), families=list(params.get("families", [])), a=float(params.get("a", 1.0)),
                    w=np.asarray(params.get("w", []), dtype="f8"), c=float(params.get("c", 0.0)),
                    ridge=float(params.get("ridge", 1.0)), fitted=bool(params.get("fitted", True)), prior=params.get("prior", "naive"))  # fmt: skip
    return f


def question_only(p0: np.ndarray, alert_idx: np.ndarray, steps: pl.DataFrame, n_master: int, model: fuse.Fusion) -> np.ndarray:
    """p0 plus the run's answers through fusion, for one (system, variant, seed): summed accepted log-LR per family."""
    out = p0.copy()
    s = steps.filter(pl.col("log_lr").is_not_null() & pl.col("family").is_not_null())
    if s.is_empty():
        return out
    pos = {a: i for i, a in enumerate(alert_idx)}
    g = s.group_by(["alert_id", "family"]).agg(pl.col("log_lr").sum().alias("x"))
    fams = list(model.families) if model.families else sorted(g["family"].unique().to_list())
    fi = {f: j for j, f in enumerate(fams)}
    X = np.zeros((len(p0), len(fams)))
    extra = np.zeros(len(p0))  # families the fusion never saw keep their back-tested weight 1
    for a, f, x in g.iter_rows():
        i = pos.get(a)
        if i is None:
            continue
        if f in fi:
            X[i, fi[f]] += x
        else:
            extra[i] += x
    pred = model.predict(p0, X) if X.shape[1] else p0
    return fuse.sigmoid(fuse.logit(pred) + extra)


def score_runs(run: Run, alerts: pl.DataFrame, fusion_params: dict | None = None) -> tuple[dict[Key, Scores], dict]:
    """Scores for every (system, variant, mode, alert_set) of the run, on the master alerts. Returns (scores, info)."""
    ids = alerts["alert_id"].to_list()
    pos = {a: i for i, a in enumerate(ids)}
    n = len(ids)
    y = alerts["label"].cast(pl.Float64).to_numpy()
    p0_master = alerts["p0"].cast(pl.Float64).to_numpy() if "p0" in alerts.columns else np.full(n, np.nan)
    model = _fusion_from(fusion_params)
    log = run.log
    inj_keys = _injection_keys(log)
    v_all = log.filter(pl.col("event") == "verdict")
    s_all = log.filter(pl.col("event") == "step")
    out: dict[Key, Scores] = {}
    info = {"dropped_unknown_alerts": 0, "injection_keys": sorted(inj_keys)}
    for kt, grp in v_all.group_by(["system", "variant", "mode", "_alert_set"], maintain_order=True):
        key = Key(*kt)
        if (key.system, key.variant) in inj_keys:
            continue
        vs = grp.filter(pl.col("alert_id").is_in(ids))
        info["dropped_unknown_alerts"] += grp.height - vs.height
        if vs.is_empty():
            continue
        steps = s_all.filter((pl.col("system") == key.system) & (pl.col("variant") == key.variant) & (pl.col("mode") == key.mode)
                             & (pl.col("_alert_set") == key.alert_set))  # fmt: skip
        sums = {k: np.zeros(n) for k in ("se_e2e", "se_qo", "p_e2e", "p_qo")}
        cnt = np.zeros(n)
        nov = tot = 0
        for seed, sg in vs.group_by("seed", maintain_order=True):
            idx = np.array([pos[a] for a in sg["alert_id"].to_list()])
            p0_log = sg["p0"].cast(pl.Float64).to_numpy()
            p0 = np.where(np.isnan(p0_log), p0_master[idx], p0_log)
            if np.isnan(p0).any():
                raise AnalysisInputError(f"{key.label()}: verdict rows lack p0 and the alert table has none")
            e2e, rate = mt.end_to_end_probability(sg["probability"].to_list(), p0)
            nov += rate * len(idx)
            tot += len(idx)
            st_seed = steps.filter(pl.col("seed") == seed[0]) if not steps.is_empty() else steps
            qo = question_only(p0, np.array(sg["alert_id"].to_list()), st_seed, len(idx), model)
            yy = y[idx]
            sums["se_e2e"][idx] += (e2e - yy) ** 2
            sums["se_qo"][idx] += (qo - yy) ** 2
            sums["p_e2e"][idx] += e2e
            sums["p_qo"][idx] += qo
            cnt[idx] += 1
        ok = cnt > 0
        res = {k: np.where(ok, v / np.where(ok, cnt, 1), np.nan) for k, v in sums.items()}
        cost = {}
        for c in ("llm_calls", "tokens_in", "tokens_out", "gpu_seconds", "eps_spent", "sim_days", "trials"):
            if c in vs.columns:
                vals = vs[c].drop_nulls()
                cost[c] = float(vals.mean()) if len(vals) else None
        if not steps.is_empty():
            asked = steps.filter(pl.col("alert_id").is_in(ids))
            cost["questions_per_alert"] = asked.height / (vs.height or 1)
            if "response_kind" in asked.columns:
                cost["answered_per_alert"] = asked.filter(pl.col("response_kind") == "reply").height / (vs.height or 1)
        else:
            cost["questions_per_alert"] = 0.0
        stop = {k: int(c) for k, c in vs["stop_reason"].value_counts().iter_rows() if k is not None} if "stop_reason" in vs.columns else {}
        out[key] = Scores(key.label(), res["se_e2e"], res["se_qo"], res["p_e2e"], res["p_qo"], nov / tot if tot else float("nan"),
                          cost, stop, key)  # fmt: skip
    return out, info


def _injection_keys(log: pl.DataFrame) -> set[tuple[str, str]]:
    if "injection_id" not in log.columns:
        return set()
    s = log.filter((pl.col("event") == "step") & pl.col("injection_id").is_not_null())
    return {(a, b) for a, b in s.select("system", "variant").unique().iter_rows()}


def source_scores(name: str, p: np.ndarray, y: np.ndarray, p0: np.ndarray) -> Scores:
    se = (p - y) ** 2
    return Scores(name, se, None, p, None, 0.0, {}, {}, None, "table")


# --------------------------------------------------------------------------------------------------------------------
# the panel: roles -> scores on the master alerts
# --------------------------------------------------------------------------------------------------------------------
def _match(value: str, pat) -> bool:
    if pat is None:
        return True
    pats = pat if isinstance(pat, list) else [pat]
    return any(fnmatch.fnmatchcase(value or "", str(p)) for p in pats)


def select_keys(keys, sel: dict) -> list[Key]:
    return [k for k in keys if _match(k.system, sel.get("system")) and _match(k.variant, sel.get("variant"))
            and _match(k.mode, sel.get("mode")) and _match(k.alert_set, sel.get("alert_set"))]  # fmt: skip


@dataclass
class Panel:
    split: str
    alerts: pl.DataFrame
    y: np.ndarray
    p0: np.ndarray
    clusters: np.ndarray
    cluster_info: dict
    scores: dict[Key, Scores]
    sources: dict[str, Scores]
    systems_cfg: dict
    n_boot: int = bs.N_BOOT
    seed: int = bs.SEED
    run: Run | None = None
    fixed: dict = field(default_factory=dict)
    fixed_derived: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ---- roles
    def selector(self, role: str) -> dict:
        r = (self.systems_cfg.get("roles") or {}).get(role)
        if r is None:
            raise Missing(f"role {role!r} is not defined in systems.yaml")
        return r

    def role(self, role: str, prefer: str | None = None) -> Scores:
        if role == "one_bank":
            return source_scores("one_bank", self.p0, self.y, self.p0)
        if role in self.sources:
            return self.sources[role]
        sel = self.selector(role)
        if "source" in sel:
            if sel["source"] in self.sources:
                return self.sources[sel["source"]]
            raise Missing(f"role {role}: score table {sel['source']!r} is not provided (inputs.yaml scores.{sel['source']}.path)")
        return self.pick(sel, role, prefer)

    def pick(self, sel: dict, label: str, prefer: str | None = None) -> Scores:
        found = select_keys(self.scores, sel)
        if not found:
            raise Missing(f"role {label}: no run-log rows match {json.dumps(sel)}{self._pending_hint(sel)}")
        combos = {(k.system, k.variant, k.mode) for k in found}
        if len(combos) > 1:
            raise Missing(f"role {label}: ambiguous, {json.dumps(sel)} matches {sorted('/'.join(c) for c in combos)}; narrow the selector in systems.yaml")
        by_set = {k.alert_set: k for k in found}
        for want in ([prefer] if prefer else []) + ["full_test", "any", "subset"]:
            if want in by_set:
                return self.scores[by_set[want]]
        raise Missing(f"role {label}: no usable alert set among {sorted(by_set)}")

    def _pending_hint(self, sel: dict) -> str:
        """For a partial run: name the manifest jobs that are not done and could be this arm."""
        miss = (self.run.missing_jobs if self.run else None) or []
        if not miss:
            return ""
        pats = sel.get("system") if isinstance(sel.get("system"), list) else [sel.get("system")]
        toks = [str(x).replace("*", "") for x in pats if x]
        hits = [m for m in miss if not toks or any(t and t in f"{m.get('job_id')} {m.get('system')}" for t in toks)]
        if not hits:
            return ""
        return " (pending arm: manifest job " + ", ".join(str(m.get("job_id") or m.get("system")) for m in hits[:4]) + " is not done yet)"

    def try_role(self, role: str, prefer: str | None = None) -> Scores | None:
        try:
            return self.role(role, prefer)
        except Missing:
            return None

    def uppers(self) -> dict[str, Scores]:
        return {u: s for u in UPPERS if (s := self.try_role(u)) is not None}

    def best_simple(self) -> tuple[str, str]:
        """(role name, how it was chosen): the value fixed on validation, or derived on this run when split == val."""
        fixed = self.fixed.get("best_simple_planner")
        if fixed:
            return fixed, "fixed on validation"
        if self.split == "test":
            raise Missing("best_simple_planner is not fixed: run --split val first and pass its fixed_on_val.json")
        names = [r for r in self.systems_cfg.get("simple_planners", []) if self.try_role(r) is not None]
        if not names:
            raise Missing("no simple planner (random, fixed_order, eig, learned) in the run")
        brier = {r: float(np.nanmean(self.role(r).se_e2e)) for r in names}
        best = min(brier, key=brier.get)
        self.fixed_derived["best_simple_planner"] = best
        return best, "derived on this validation run"

    # ---- common alerts and paired bootstrap
    def common(self, vectors: dict[str, np.ndarray]) -> np.ndarray:
        m = np.ones(len(self.y), dtype=bool)
        for v in vectors.values():
            m &= np.isfinite(v)
        if not m.any():
            raise Missing("the systems compared have no alert in common")
        return m

    def boot(self, vectors: dict[str, np.ndarray], n_boot: int | None = None) -> dict:
        """Point means and paired resampled means of per-alert squared errors (or any per-alert values) on the common alerts."""
        m = self.common(vectors)
        names = list(vectors)
        cols = np.column_stack([vectors[k][m] for k in names])
        cl = self.clusters[m]
        bm = bs.bootstrap_means(cols, cl, n_boot or self.n_boot, self.seed)
        return {"mask": m, "n": int(m.sum()), "n_pos": int(self.y[m].sum()), "n_clusters": int(len(np.unique(cl))),
                "point": dict(zip(names, cols.mean(axis=0), strict=True)),
                "samples": {k: bm[:, i] for i, k in enumerate(names)}, "clusters": cl}  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# panel construction
# --------------------------------------------------------------------------------------------------------------------
def load_fixed(path: str | Path | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise AnalysisInputError(f"fixed_on_val file {p} not found")
    return json.loads(p.read_text(encoding="utf-8"))


def build_panel(run: Run, inputs: dict, systems: dict, fixed_path: str | Path | None = None, n_boot: int = bs.N_BOOT,
                seed: int = bs.SEED) -> Panel:  # fmt: skip
    split = run.split
    try:
        alerts = load_alerts(inputs, split, "headline")
    except Missing as e:
        if run.synthetic and run.synth_labels is not None:  # a rehearsal run has no alert table: its labels sit in its own logs
            alerts = run.synth_labels.with_columns(pl.lit(None).alias("attempt_id"), pl.lit(None).alias("src_id"), pl.lit(None).alias("dst_id")).sort("alert_id")
        else:
            raise AnalysisInputError(str(e)) from e
    fixed = load_fixed(fixed_path or inputs.get("fixed_on_val"))
    if split == "test" and not fixed:
        raise AnalysisInputError("--split test needs the values fixed on validation (fixed_on_val.json from a --split val run): "
                                 "the best simple planner, the equivalence margins and the hub cap are not derived on test")
    sc = inputs.get("scores") or {}
    y = alerts["label"].cast(pl.Float64).to_numpy()
    p0 = load_score(sc.get("one_bank") or {}, alerts, split) if (sc.get("one_bank") or {}).get("path") else None
    if p0 is None:
        if "p0" not in alerts.columns:
            raise AnalysisInputError("no one-bank score: set scores.one_bank.path or give the alert table a p0 column")
        p0 = alerts["p0"].cast(pl.Float64).to_numpy()
    if np.isnan(p0).any():
        raise AnalysisInputError(f"the one-bank score is missing for {int(np.isnan(p0).sum())} alerts")
    alerts = alerts.with_columns(pl.Series("p0", p0))
    hub = fixed.get("hub_cap", (inputs.get("alerts") or {}).get("cluster_hub_cap"))
    acc = [alerts[c].to_list() for c in (inputs.get("alerts") or {}).get("account_cols", ["src_id", "dst_id"])]
    clusters = bs.cluster_ids(alerts["attempt_id"].to_list(), acc, hub_cap=hub)
    cinfo = {**bs.cluster_report(clusters), "hub_cap": hub}
    scores, info = score_runs(run, alerts, _read_json(resolve_path((inputs.get("fusion") or {}).get("params"), split)))
    sources: dict[str, Scores] = {}
    for name, spec in sc.items():
        if name == "one_bank" or not spec or not spec.get("path"):
            continue
        p = load_score(spec, alerts, split)
        if p is not None:
            sources[name] = source_scores(name, p, y, p0)
    panel = Panel(split, alerts, y, p0, clusters, cinfo, scores, sources, systems, n_boot, seed, run, fixed)
    panel.warnings += run.warnings
    if info["dropped_unknown_alerts"]:
        panel.warnings.append(f"{info['dropped_unknown_alerts']} verdict rows are for alerts outside the headline alert set (dropped)")
    if cinfo["largest_share"] > 0.2 and hub is None:
        panel.warnings.append(f"largest cluster holds {cinfo['largest_share']:.1%} of the alerts and no hub cap is fixed (prereg: set one if > 20%)")
    return panel


def _read_json(p: Path | None) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p is not None and p.exists() else None


# --------------------------------------------------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------------------------------------------------
def fnum(x) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def ci_tuple(samples, level: float = 0.95) -> tuple[float, float]:
    s = np.asarray(samples, dtype="f8")
    return bs.percentile_ci(s, level) if np.isfinite(s).any() else (float("nan"), float("nan"))


def diff_stat(B: dict, a: str, b: str) -> dict:
    """Brier(a) - Brier(b): point, CI, bootstrap samples (negative = a better)."""
    d = B["samples"][a] - B["samples"][b]
    return {"point": float(B["point"][a] - B["point"][b]), "ci": ci_tuple(d), "samples": d}


def gap_stat(B: dict, ob: str, sysn: str, up: str, min_den: float = mt.MIN_GAP_DENOMINATOR) -> dict:
    """Gap closed of `sysn` between `ob` and `up`: point, CI, share of resamples with a tiny denominator."""
    s = B["samples"]
    den = s[ob] - s[up]
    g = np.where(den >= min_den, (s[ob] - s[sysn]) / np.where(den == 0, 1, den), np.nan)
    nan_share = float(np.isnan(g).mean())
    return {"point": mt.gap_closed(B["point"][ob], B["point"][sysn], B["point"][up], min_den),
            "ci": ci_tuple(g) if nan_share < 1 else (float("nan"), float("nan")), "nan_share": nan_share, "samples": g,
            "denominator": float(B["point"][ob] - B["point"][up])}  # fmt: skip


def p_greater(samples, null: float = 0.0) -> float:
    return bs.bootstrap_pvalue(samples, null, "greater")


def p_less(samples, null: float = 0.0) -> float:
    return bs.bootstrap_pvalue(samples, null, "less")


def mde(samples) -> float:
    """Smallest true Brier difference detectable with 80% power at two-sided 5% from the bootstrap SE of the difference."""
    s = np.asarray(samples, dtype="f8")
    return float(MDE_MULTIPLIER * np.nanstd(s))


def power_statement(n_alerts: int, n_subset: int | None, mde_value: float | None) -> str:
    ref = lambda n: PLAN_MDE["brier"] * math.sqrt(PLAN_MDE["n"] / n)  # noqa: E731
    parts = [f"With {n_alerts} alerts the plan's rule of thumb gives a detectable Brier difference of about {ref(n_alerts):.4f} "
             f"({PLAN_MDE['brier']} at {PLAN_MDE['n']} alerts, scaled by 1/sqrt(n))."]  # fmt: skip
    if n_subset:
        parts.append(f"On the {n_subset}-alert external subset it is about {ref(n_subset):.4f}.")
    if mde_value is not None and math.isfinite(mde_value):
        parts.append(f"From the cluster-bootstrap standard error of the paired difference between our system and one bank on this "
                     f"run, the smallest difference detectable with 80% power at two-sided 5% is {mde_value:.4f} Brier.")  # fmt: skip
    return " ".join(parts)


def _fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _fci(ci, nd: int = 3) -> str:
    return f"[{_fmt(ci[0], nd)}, {_fmt(ci[1], nd)}]"


def _res(hid: str, tier: str, claim: str, rule: str, **kw) -> dict:
    return {"id": hid, "tier": tier, "claim": claim, "rule": rule, "status": "tested", "verdict": None, "reason": None, "p": None,
            "p_adj": None, "holm_reject": None, "rule_met": None, "components": [], "notes": [], **kw}  # fmt: skip


def _not_run(hid: str, tier: str, claim: str, rule: str, reason: str) -> dict:
    return _res(hid, tier, claim, rule, status="not_run", verdict="not_run", reason=reason)


def _clean(d: dict) -> dict:
    """Drop bootstrap samples and numpy types so a result can be written as JSON."""
    out = {}
    for k, v in d.items():
        if k == "samples":
            continue
        if isinstance(v, dict):
            out[k] = _clean(v)
        elif isinstance(v, list):
            out[k] = [_clean(x) if isinstance(x, dict) else x for x in v]
        elif isinstance(v, tuple):
            out[k] = [fnum(x) for x in v]
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------------------------------------------------
# margins
# --------------------------------------------------------------------------------------------------------------------
def margins(panel: Panel) -> dict:
    """Equivalence margin (Brier) and the H9 margin in gap-closed units. Fixed on validation; derived only on val."""
    if "equivalence_margin" in panel.fixed:
        m = float(panel.fixed["equivalence_margin"])
        mg = panel.fixed.get("margin_gap")
        return {"equivalence_margin": m, "margin_gap": None if mg is None else float(mg), "source": "fixed on validation"}
    if panel.split == "test":
        raise Missing("the equivalence margin is not fixed: run --split val first and pass its fixed_on_val.json")
    best, _ = panel.best_simple()
    vec = {"random": panel.role("random").se_e2e, "best": panel.role(best).se_e2e}
    B = panel.boot(vec, n_boot=2)
    margin = st.equivalence_margin(B["point"]["best"], B["point"]["random"])
    mg = None
    up = panel.uppers().get("gfp_see_everything")
    if up is not None:
        v = {"ob": np.where(np.isfinite(up.se_e2e), (panel.p0 - panel.y) ** 2, np.nan), "up": up.se_e2e}
        den = panel.boot(v, n_boot=2)["point"]
        d = den["ob"] - den["up"]
        mg = margin / d if d >= mt.MIN_GAP_DENOMINATOR else None
    panel.fixed_derived.update({"equivalence_margin": margin, "margin_gap": mg})
    return {"equivalence_margin": margin, "margin_gap": mg, "source": "derived on this validation run"}


# --------------------------------------------------------------------------------------------------------------------
# hypotheses
# --------------------------------------------------------------------------------------------------------------------
def _one_bank_vec(panel: Panel) -> np.ndarray:
    return (panel.p0 - panel.y) ** 2


def _gap_vs_uppers(panel: Panel, system_vec: np.ndarray, extra: dict | None = None, label: str = "ours") -> dict:
    """Gap closed of one system against each available upper limit, on the alerts all of them share."""
    ups = panel.uppers()
    if not ups:
        raise Missing("no upper limit provided (scores.gfp_see_everything / scores.best_sota_see_everything)")
    out = {}
    for u, s in ups.items():
        vec = {"ob": _one_bank_vec(panel), "sys": system_vec, "up": s.se_e2e, **(extra or {})}
        B = panel.boot(vec)
        out[u] = {"B": B, "gap": gap_stat(B, "ob", "sys", "up"), "n": B["n"]}
    return out


def h1(panel: Panel) -> dict:
    rule = "95% CI lower bound > 0 AND point estimate >= 0.30"
    claim = "At eps = 2 per alert, private questions close at least 30% of the gap (Brier)."
    ours = panel.role("ours_compose")
    res = _gap_vs_uppers(panel, ours.se_e2e)
    r = _res("H1", "primary", claim, rule, statistic="gap_closed(one_bank, ours_compose_eps2, upper)", threshold=0.30)
    ps, met = [], []
    for u, o in res.items():
        g = o["gap"]
        p = p_greater(g["samples"], 0.0)
        ok = bool(np.isfinite(g["ci"][0]) and g["ci"][0] > 0 and np.isfinite(g["point"]) and g["point"] >= 0.30)
        ps.append(p)
        met.append(ok)
        r["components"].append({"name": f"gap closed vs {u}", "estimate": g["point"], "ci": g["ci"], "p": p, "rule_met": ok, "n": o["n"],
                                "nan_share": g["nan_share"], "denominator": g["denominator"]})  # fmt: skip
    first = next(iter(res.values()))["B"]
    r.update(n_alerts=first["n"], n_pos=first["n_pos"], n_clusters=first["n_clusters"], p=max(ps), rule_met=all(met),
             estimate=r["components"][0]["estimate"], ci=r["components"][0]["ci"])  # fmt: skip
    if len(res) < len(UPPERS):
        r["notes"].append("tested against " + ", ".join(res) + " only; the other upper limit is not provided (intersection-union over the available ones)")
    d = diff_stat(first, "sys", "ob")
    r["mde_brier"] = mde(d["samples"])
    r["brier_diff_vs_one_bank"] = {"point": d["point"], "ci": d["ci"]}
    return r


def h2(panel: Panel) -> dict:
    rule = "95% CI lower bound >= 0.70"
    claim = "Questions get at least 70% of the benefit of sharing raw records, while sharing none."
    ours, raw = panel.role("ours_compose"), panel.role("raw_share")
    B = panel.boot({"ob": _one_bank_vec(panel), "ours": ours.se_e2e, "raw": raw.se_e2e})
    s = B["samples"]
    den = s["ob"] - s["raw"]
    ratio = np.where(den >= mt.MIN_GAP_DENOMINATOR, (s["ob"] - s["ours"]) / np.where(den == 0, 1, den), np.nan)
    d0 = B["point"]["ob"] - B["point"]["raw"]
    point = (B["point"]["ob"] - B["point"]["ours"]) / d0 if d0 >= mt.MIN_GAP_DENOMINATOR else float("nan")
    ci = ci_tuple(ratio)
    p = p_greater(ratio, 0.70)
    ok = bool(np.isfinite(ci[0]) and ci[0] >= 0.70)
    r = _res("H2", "primary", claim, rule, statistic="(B(one_bank) - B(ours_eps2)) / (B(one_bank) - B(raw_share_3_banks))", threshold=0.70)
    r.update(n_alerts=B["n"], n_pos=B["n_pos"], n_clusters=B["n_clusters"], estimate=point, ci=ci, p=p, rule_met=ok,
             nan_share=float(np.isnan(ratio).mean()))  # fmt: skip
    r["components"].append({"name": "ratio of benefits", "estimate": point, "ci": ci, "p": p, "rule_met": ok})
    if d0 < mt.MIN_GAP_DENOMINATOR:
        r["notes"].append("raw sharing does not beat one bank (tiny denominator): the ratio is undefined, report raw Brier differences")
    r["mde_brier"] = mde(s["ob"] - s["ours"])
    return r


def h3(panel: Panel) -> dict:
    rule = "all three hold: non-inferior to the best simple planner and to the enumerator (CI upper bound < margin), better than the fixed list (CI upper bound < 0)"
    claim = "LLM composing its own questions is at least as good as the best simple planner AND the enumerator, and better than itself with a fixed list."
    m = margins(panel)
    best, how = panel.best_simple()
    prefer = None
    vec = {"llm": panel.role("ours_compose", prefer).se_e2e, "best": panel.role(best).se_e2e,
           "enum": panel.role("enumerator").se_e2e, "fixed": panel.role("llm_fixed_list").se_e2e}  # fmt: skip
    B = panel.boot(vec)
    r = _res("H3", "primary", claim, rule, statistic="intersection-union of three bootstrap tests", margin=m["equivalence_margin"],
             margin_source=m["source"], best_simple_planner=best, best_simple_planner_source=how)  # fmt: skip
    ps, oks = [], []
    for name, other, kind in ((f"B(llm) - B(best simple: {best})", "best", "non_inferior"), ("B(llm) - B(enumerator)", "enum", "non_inferior"),
                              ("B(llm) - B(llm fixed list)", "fixed", "superior")):  # fmt: skip
        d = diff_stat(B, "llm", other)
        if kind == "non_inferior":
            p, ok = p_less(d["samples"], m["equivalence_margin"]), st.non_inferior(d["ci"], m["equivalence_margin"])
        else:
            p, ok = p_less(d["samples"], 0.0), bool(d["ci"][1] < 0)
        ps.append(p)
        oks.append(ok)
        r["components"].append({"name": name, "kind": kind, "estimate": d["point"], "ci": d["ci"], "p": p, "rule_met": ok})
    r.update(n_alerts=B["n"], n_pos=B["n_pos"], n_clusters=B["n_clusters"], p=max(ps), rule_met=all(oks),
             estimate=r["components"][0]["estimate"], ci=r["components"][0]["ci"])  # fmt: skip
    r["mde_brier"] = mde(B["samples"]["llm"] - B["samples"]["best"])
    return r


def _useless_questions(panel: Panel, min_asked: int = 30) -> dict:
    """FDR audit (H4): among accepted questions asked >= 30 times, the share whose realised LR in the claimed direction is <= 1."""
    ours = panel.role("ours_compose")
    key = ours.key
    if key is None:
        raise Missing("the FDR audit needs the run-log step rows of ours_compose")
    log = panel.run.log
    s = log.filter((pl.col("event") == "step") & (pl.col("system") == key.system) & (pl.col("variant") == key.variant)
                   & (pl.col("mode") == key.mode) & (pl.col("_alert_set") == key.alert_set)
                   & pl.col("log_lr").is_not_null() & pl.col("program_canonical").is_not_null())  # fmt: skip
    if "response_kind" in s.columns:
        s = s.filter(pl.col("response_kind") == "reply")
    lab = panel.alerts.select("alert_id", pl.col("label").cast(pl.Int64).alias("_y"))
    s = s.join(lab, on="alert_id", how="inner")
    if s.is_empty():
        raise Missing("no answered, back-tested step rows in the ours_compose logs")
    qcol = ["program_canonical", "eps"] if "eps" in s.columns else ["program_canonical"]
    rows, useless = [], 0
    for qk, g in s.group_by(qcol):
        g = g.unique(["alert_id", "seed"] + ["step"], keep="first") if "step" in g.columns else g
        n_ans = g.height
        if n_ans < min_asked:
            continue
        n1, n0 = max(int((g["_y"] == 1).sum()), 0), max(int((g["_y"] == 0).sum()), 0)
        if n1 == 0 or n0 == 0:
            continue
        cells = g["answer_value"].fill_null("").to_list() if "answer_value" in g.columns else [""] * n_ans
        gy = g["_y"].to_numpy()
        cell_ids = np.array(cells)
        score = 0.0
        for c in np.unique(cell_ids):
            m = cell_ids == c
            c1, c0 = (gy[m] == 1).sum(), (gy[m] == 0).sum()
            real = math.log(((c1 + 0.5) / (n1 + 0.5)) / ((c0 + 0.5) / (n0 + 0.5)))  # smoothed realised log LR of this answer
            claimed = float(g["log_lr"].to_numpy()[m].mean())
            score += m.sum() * math.copysign(1.0, claimed) * real if claimed != 0 else 0.0
        mean_real = score / n_ans
        is_useless = mean_real <= 0
        useless += int(is_useless)
        rows.append({"question": qk[0], "eps": qk[1] if len(qk) > 1 else None, "asked": n_ans, "realised_log_lr_in_claimed_direction": mean_real,
                     "useless": bool(is_useless)})  # fmt: skip
    k, n = useless, len(rows)
    if n == 0:
        raise Missing(f"no accepted question was answered at least {min_asked} times, so the FDR audit cannot be evaluated")
    from scipy.stats import beta

    upper = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return {"k_useless": k, "n_questions": n, "share": k / n, "upper95": upper, "rows": rows, "min_asked": min_asked}


def h4(panel: Panel) -> dict:
    rule = "ours better than no back-testing (CI upper bound < 0) AND FDR audit upper 95% bound < 0.10"
    claim = "Back-testing matters: without it results get worse; with it, under 10% of accepted questions are useless on later data."
    ours, nb = panel.role("ours_compose"), panel.role("no_backtest")
    vec = {"ours": ours.se_e2e, "nb": nb.se_e2e}
    dec = panel.try_role("llm_decides")
    if dec is not None:
        vec["dec"] = dec.se_e2e
    B = panel.boot(vec)
    d = diff_stat(B, "ours", "nb")
    p = p_less(d["samples"], 0.0)
    sup = bool(d["ci"][1] < 0)
    r = _res("H4", "primary", claim, rule, statistic="B(ours_compose) - B(no_backtest); FDR audit",
             threshold_fdr=0.10)  # fmt: skip
    r.update(n_alerts=B["n"], n_pos=B["n_pos"], n_clusters=B["n_clusters"], estimate=d["point"], ci=d["ci"], p=p)
    r["components"].append({"name": "B(ours_compose) - B(no_backtest)", "estimate": d["point"], "ci": d["ci"], "p": p, "rule_met": sup})
    if dec is not None:
        sf = diff_stat(B, "dec", "nb")
        r["components"].append({"name": "single factor: B(llm_decides) - B(no_backtest) (reported, not part of the test)",
                                "estimate": sf["point"], "ci": sf["ci"], "p": None, "rule_met": None})  # fmt: skip
    r["mde_brier"] = mde(d["samples"])
    try:
        audit = _useless_questions(panel)
    except Missing as e:
        r["fdr_audit"] = {"status": "not_run", "reason": str(e)}
        r["rule_met"] = None
        r["notes"].append(f"FDR audit not evaluable: {e}")
        r["audit_missing"] = True
        return r
    ok = audit["upper95"] < 0.10
    r["fdr_audit"] = {k: v for k, v in audit.items() if k != "rows"}
    r["fdr_audit"]["rows"] = audit["rows"]
    r["components"].append({"name": "FDR audit: share of accepted questions with LR <= 1 on this period", "estimate": audit["share"],
                            "ci": (float("nan"), audit["upper95"]), "p": None, "rule_met": ok, "n": audit["n_questions"]})  # fmt: skip
    r["rule_met"] = bool(sup and ok)
    return r


def best_ported_external(panel: Panel) -> tuple[str, str]:
    """The ported external agent with the highest end-to-end gap closed (point, vs GFP): fixed on validation."""
    if panel.fixed.get("best_ported_external"):
        return panel.fixed["best_ported_external"], "fixed on validation"
    if panel.split == "test":
        raise Missing("best_ported_external is not fixed: choose it on validation (G4) and pass fixed_on_val.json")
    up = panel.uppers().get("gfp_see_everything") or next(iter(panel.uppers().values()), None)
    if up is None:
        raise Missing("no upper limit provided, so the best external agent cannot be chosen")
    best, best_g = None, -np.inf
    ob = _one_bank_vec(panel)
    for name, spec in (panel.systems_cfg.get("externals") or {}).items():
        if "ported" not in spec:
            continue
        s = panel.try_pick(spec["ported"], name)
        if s is None:
            continue
        m = np.isfinite(s.se_e2e) & np.isfinite(up.se_e2e)
        if not m.any():
            continue
        g = mt.gap_closed(ob[m].mean(), s.se_e2e[m].mean(), up.se_e2e[m].mean())
        if np.isfinite(g) and g > best_g:
            best, best_g = name, g
    if best is None:
        raise Missing("no ported external agent with results in the run")
    panel.fixed_derived["best_ported_external"] = best
    return best, "derived on this validation run"


def _try_pick(self: Panel, sel: dict, label: str, prefer: str | None = None) -> Scores | None:
    try:
        return self.pick(sel, label, prefer)
    except Missing:
        return None


Panel.try_pick = _try_pick  # type: ignore[attr-defined]


def h9(panel: Panel) -> dict:
    rule = "95% CI lower bound of gap_closed(ours) - gap_closed(best ported external) > -margin_gap"
    claim = "At eps = 2, our system closes at least as much of the gap as the best ported external agent on the same backbone."
    m = margins(panel)
    if m["margin_gap"] is None:
        raise Missing("margin_gap is undefined (no GFP upper limit, or its denominator is tiny)")
    name, how = best_ported_external(panel)
    spec = (panel.systems_cfg.get("externals") or {}).get(name, {}).get("ported")
    if spec is None:
        raise Missing(f"external {name!r} has no ported entry in systems.yaml")
    ext = panel.pick(spec, f"external {name} (ported)", prefer="subset")
    ours = panel.role("ours_compose", prefer="subset")
    r = _res("H9", "primary", claim, rule, statistic="gap_closed(ours) - gap_closed(best_ported_external)", margin_gap=m["margin_gap"],
             margin_gap_source=m["source"], best_ported_external=name, best_ported_external_source=how, eval_set="subset (external agents)")  # fmt: skip
    ps, oks = [], []
    for score in ("end_to_end", "questions_only"):
        a = ours.se_e2e if score == "end_to_end" else ours.se_qo
        b = ext.se_e2e if score == "end_to_end" else ext.se_qo
        if a is None or b is None:
            continue
        for u, up in panel.uppers().items():
            B = panel.boot({"ob": _one_bank_vec(panel), "a": a, "b": b, "up": up.se_e2e})
            ga, gb = gap_stat(B, "ob", "a", "up"), gap_stat(B, "ob", "b", "up")
            d = ga["samples"] - gb["samples"]
            point, ci = ga["point"] - gb["point"], ci_tuple(d)
            p = p_greater(d, -m["margin_gap"])
            ok = bool(np.isfinite(ci[0]) and ci[0] > -m["margin_gap"])
            comp = {"name": f"{score} vs {u}", "estimate": point, "ci": ci, "p": p, "rule_met": ok, "n": B["n"],
                    "ours_gap": ga["point"], "external_gap": gb["point"], "is_test": score == "end_to_end"}  # fmt: skip
            r["components"].append(comp)
            if score == "end_to_end":
                ps.append(p)
                oks.append(ok)
                r["n_alerts"], r["n_pos"], r["n_clusters"] = B["n"], B["n_pos"], B["n_clusters"]
                if "mde_brier" not in r:
                    r["mde_brier"] = mde(B["samples"]["a"] - B["samples"]["ob"])
    if not ps:
        raise Missing("no upper limit provided")
    r.update(p=max(ps), rule_met=all(oks), estimate=r["components"][0]["estimate"], ci=r["components"][0]["ci"])
    return r


def h6(panel: Panel, inputs: dict) -> dict:
    rule = "pattern-edge share difference CI lower bound >= 0.15 AND Brier difference CI upper bound < 0"
    claim = "Smart retrieval shows at least 15 points more real pattern edges among the 40, and improves results."
    a = panel.role("evidence_scorer_pcst")
    b = panel.role("evidence_top40")
    spec = (inputs.get("evidence_edges") or {})
    path = resolve_path(spec.get("path"), panel.split)
    if path is None or not path.exists():
        raise Missing("pattern-edge shares per evidence view are not provided (inputs.yaml evidence_edges.path: view, alert_id, pattern_edge_share)")
    e = pl.read_parquet(path)  # time-guard: not transactions
    ids = panel.alerts["alert_id"].to_list()

    def share(view: str) -> np.ndarray:
        d = dict(e.filter(pl.col("view") == view).select("alert_id", "pattern_edge_share").iter_rows())
        return np.array([d.get(i, np.nan) for i in ids], dtype="f8")

    views = (spec.get("views") or {"scorer_pcst": "scorer_pcst", "top40": "top40_by_amount"})
    sa, sb = share(views["scorer_pcst"]), share(views["top40"])
    B = panel.boot({"a": a.se_e2e, "b": b.se_e2e, "sa": sa, "sb": sb})
    ds = B["samples"]["sa"] - B["samples"]["sb"]
    db = diff_stat(B, "a", "b")
    ps, pb = p_greater(ds, 0.15), p_less(db["samples"], 0.0)
    ok = bool(ci_tuple(ds)[0] >= 0.15 and db["ci"][1] < 0)
    r = _res("H6", "secondary", claim, rule, statistic="pattern_edge_share(scorer_pcst) - pattern_edge_share(top40); B(scorer_pcst) - B(top40)")
    r.update(n_alerts=B["n"], n_pos=B["n_pos"], n_clusters=B["n_clusters"], p=max(ps, pb), rule_met=ok,
             estimate=float(B["point"]["sa"] - B["point"]["sb"]), ci=ci_tuple(ds))  # fmt: skip
    r["components"] = [{"name": "pattern-edge share difference", "estimate": r["estimate"], "ci": r["ci"], "p": ps, "rule_met": bool(r["ci"][0] >= 0.15)},
                       {"name": "Brier difference", "estimate": db["point"], "ci": db["ci"], "p": pb, "rule_met": bool(db["ci"][1] < 0)}]  # fmt: skip
    return r


def h7(panel: Panel) -> dict:
    rule = "95% CI upper bound of B(ours_compose) - B(llm_decides) < 0"
    claim = "A calibrated formula making the final call beats the LLM deciding."
    B = panel.boot({"a": panel.role("ours_compose").se_e2e, "b": panel.role("llm_decides").se_e2e})
    d = diff_stat(B, "a", "b")
    p, ok = p_less(d["samples"], 0.0), bool(d["ci"][1] < 0)
    r = _res("H7", "secondary", claim, rule, statistic="B(ours_compose) - B(llm_decides)")
    r.update(n_alerts=B["n"], n_pos=B["n_pos"], n_clusters=B["n_clusters"], estimate=d["point"], ci=d["ci"], p=p, rule_met=ok,
             mde_brier=mde(d["samples"]))  # fmt: skip
    r["components"].append({"name": r["statistic"], "estimate": d["point"], "ci": d["ci"], "p": p, "rule_met": ok})
    return r


def h10(panel: Panel, inputs: dict) -> dict:
    rule = "95% CI upper bound of B(ours) - B(dp_fedavg) < equivalence_margin"
    claim = "Private questions match or beat DP federated training on Brier at a comparable privacy level."
    m = margins(panel)
    fed = panel.role("dp_fedavg")
    B = panel.boot({"a": panel.role("ours_compose").se_e2e, "b": fed.se_e2e})
    d = diff_stat(B, "a", "b")
    p, ok = p_less(d["samples"], m["equivalence_margin"]), st.non_inferior(d["ci"], m["equivalence_margin"])
    r = _res("H10", "secondary", claim, rule, statistic="B(ours) - B(dp_fedavg)", margin=m["equivalence_margin"], margin_source=m["source"],
             caveat="training-time eps and per-question eps are different units; both labels are shown where they are plotted")  # fmt: skip
    label = ((inputs.get("scores") or {}).get("dp_fedavg") or {}).get("eps_label")
    r.update(n_alerts=B["n"], n_pos=B["n_pos"], n_clusters=B["n_clusters"], estimate=d["point"], ci=d["ci"], p=p, rule_met=ok,
             dp_fedavg_eps_label=label, mde_brier=mde(d["samples"]))  # fmt: skip
    r["components"].append({"name": r["statistic"], "estimate": d["point"], "ci": d["ci"], "p": p, "rule_met": ok})
    return r


def h8(panel: Panel) -> dict:
    claim = "Two questions at once, then one more, keeps 80% of the benefit in 60% of the time."
    rule = "exploratory, uncorrected: benefit ratio >= 0.80 AND simulated-time ratio <= 0.60 (point estimates, CIs shown)"
    seq, bat = panel.role("async_sequential"), panel.role("async_batched")
    B = panel.boot({"ob": _one_bank_vec(panel), "seq": seq.se_e2e, "bat": bat.se_e2e})
    s = B["samples"]
    ben_seq, ben_bat = s["ob"] - s["seq"], s["ob"] - s["bat"]
    ratio = np.where(ben_seq > 1e-12, ben_bat / np.where(ben_seq == 0, 1, ben_seq), np.nan)
    d_seq = B["point"]["ob"] - B["point"]["seq"]
    point = (B["point"]["ob"] - B["point"]["bat"]) / d_seq if d_seq > 1e-12 else float("nan")
    t_seq, t_bat = seq.cost.get("sim_days"), bat.cost.get("sim_days")
    tratio = (t_bat / t_seq) if t_seq and t_bat is not None else float("nan")
    r = _res("H8", "exploratory", claim, rule, statistic="benefit(batched)/benefit(sequential); sim_days(batched)/sim_days(sequential)")
    r.update(n_alerts=B["n"], n_pos=B["n_pos"], n_clusters=B["n_clusters"], estimate=point, ci=ci_tuple(ratio), status="exploratory")
    r["components"] = [{"name": "benefit ratio", "estimate": point, "ci": ci_tuple(ratio), "p": None, "rule_met": bool(point >= 0.8)},
                       {"name": "simulated-time ratio", "estimate": tratio, "ci": (float("nan"), float("nan")), "p": None,
                        "rule_met": bool(np.isfinite(tratio) and tratio <= 0.6)}]  # fmt: skip
    r["rule_met"] = all(c["rule_met"] for c in r["components"])
    r["verdict"] = "met (exploratory, uncorrected)" if r["rule_met"] else "not met (exploratory, uncorrected)"
    return r


def h5(panel: Panel, inputs: dict) -> dict:
    claim = "The fast model handles most routine decisions within its accuracy bound, at least 10x faster, with no detection loss."
    rule = "exploratory, uncorrected: error_marginal <= 2 * alpha per decision type; speedup >= 10; no detection loss"
    path = resolve_path((inputs.get("fast_model") or {}).get("path"), panel.split)
    if path is None or not path.exists():
        raise Missing("fast-model decision results are not provided (inputs.yaml fast_model.path; scripts/score_fast_baseline.py)")
    d = json.loads(path.read_text(encoding="utf-8"))
    types = d.get("decision_types", d if isinstance(d, list) else [])
    r = _res("H5", "exploratory", claim, rule, statistic="error_marginal vs 2*alpha; speedup; Brier change", status="exploratory")
    ok_all = True
    for t in types:
        a, e = t.get("alpha"), t.get("error_marginal")
        bound = None if a is None else 2 * a
        ok = bool(e is not None and bound is not None and e <= bound)
        ok_all &= ok
        r["components"].append({"name": f"{t.get('name')}: error_marginal vs 2*alpha", "estimate": e, "ci": (float("nan"), bound), "p": None,
                                "rule_met": ok, "kept": t.get("kept"), "speedup": t.get("speedup")})  # fmt: skip
    sp = [t.get("speedup") for t in types if t.get("speedup") is not None]
    speed_ok = bool(sp) and min(sp) >= 10
    r["rule_met"] = bool(ok_all and speed_ok and types)
    r["verdict"] = "met (exploratory, uncorrected)" if r["rule_met"] else "not met (exploratory, uncorrected)"
    r["notes"].append("error given accepted is descriptive only (not bounded by the cascade)")
    return r


HYPOTHESES = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H9": h9, "H6": h6, "H7": h7, "H10": h10, "H5": h5, "H8": h8}
_CLAIMS = {  # for not_run rows when the function fails before it can build its own row
    "H1": ("primary", "At eps = 2 per alert, private questions close at least 30% of the gap (Brier).", "95% CI lower bound > 0 AND point estimate >= 0.30"),
    "H2": ("primary", "Questions get at least 70% of the benefit of sharing raw records, while sharing none.", "95% CI lower bound >= 0.70"),
    "H3": ("primary", "LLM composing its own questions is at least as good as the best simple planner AND the enumerator, and better than itself with a fixed list.", "all three hold"),
    "H4": ("primary", "Back-testing matters: without it results get worse; with it, under 10% of accepted questions are useless on later data.", "superior holds AND fdr_audit upper 95% bound < 0.10"),
    "H9": ("primary", "At eps = 2, our system closes at least as much of the gap as the best ported external agent on the same backbone.", "95% CI lower bound > -margin_gap"),
    "H6": ("secondary", "Smart retrieval shows at least 15 points more real pattern edges among the 40, and improves results.", "share difference CI lower bound >= 0.15 AND Brier CI upper bound < 0"),
    "H7": ("secondary", "A calibrated formula making the final call beats the LLM deciding.", "95% CI upper bound < 0"),
    "H10": ("secondary", "Private questions match or beat DP federated training on Brier at a comparable privacy level.", "95% CI upper bound < equivalence_margin"),
    "H5": ("exploratory", "The fast model handles most routine decisions within its accuracy bound, at least 10x faster, with no detection loss.", "exploratory"),
    "H8": ("exploratory", "Two questions at once, then one more, keeps 80% of the benefit in 60% of the time.", "exploratory"),
}


def run_hypotheses(panel: Panel, inputs: dict) -> dict:
    """All ten hypotheses, Holm within the primary and the secondary family, exploratory ones uncorrected."""
    res: dict[str, dict] = {}
    for hid, fn in HYPOTHESES.items():
        tier, claim, rule = _CLAIMS[hid]
        try:
            res[hid] = fn(panel, inputs) if hid in ("H5", "H6", "H10") else fn(panel)
        except Missing as e:
            res[hid] = _not_run(hid, tier, claim, rule, str(e))
    for fam_name, fam in (("primary", PRIMARY), ("secondary", SECONDARY)):
        pv = {h: (res[h]["p"] if res[h]["status"] == "tested" and res[h]["p"] is not None else 1.0) for h in fam}
        holm = st.holm(pv, ALPHA)
        for h in fam:
            r = res[h]
            r["family"] = fam_name
            r["family_size"] = len(fam)
            if r["status"] != "tested":
                r["p_adj"], r["holm_reject"] = 1.0, False
                continue
            r["p_adj"], r["holm_reject"] = holm[h]["p_adj"], bool(holm[h]["reject"])
            if r.get("audit_missing"):
                r["verdict"] = "not_run"
                r["status"] = "not_run"
                r["reason"] = r["fdr_audit"]["reason"] + " (the superiority part was tested; see components)"
                r["holm_reject"] = False
                continue
            supported = bool(r["rule_met"] and r["holm_reject"])
            r["verdict"] = "supported" if supported else "not_supported"
            r["verdict_text"] = _verdict_text(r)
    for h in EXPLORATORY:
        r = res[h]
        r["family"] = "exploratory"
        r["label"] = "exploratory, uncorrected"
        if r["status"] == "not_run":
            r["verdict_text"] = f"not_run: {r['reason']}"
        else:
            r["verdict_text"] = f"{r['verdict']}: " + "; ".join(f"{c['name']} {_fmt(c['estimate'])}" for c in r["components"][:3])
    for h in PRIMARY + SECONDARY:
        if res[h]["status"] == "not_run":
            res[h]["verdict_text"] = f"not_run: {res[h]['reason']}"
    return res


def _verdict_text(r: dict) -> str:
    comps = "; ".join(f"{c['name']}: {_fmt(c['estimate'])} {_fci(c['ci'])}" + ("" if c.get("rule_met") is None else (" ok" if c["rule_met"] else " FAILS"))
                      for c in r["components"] if c.get("p") is not None or c.get("rule_met") is not None)  # fmt: skip
    holm = f"Holm p_adj {r['p_adj']:.4f} ({r['family']} family of {r['family_size']})"
    if r["verdict"] == "supported":
        return f"supported: rule '{r['rule']}' holds ({comps}); {holm}"
    why = "rule not met" if not r["rule_met"] else "rule met but not significant after Holm"
    return f"not_supported ({why}): {comps}; {holm}"


# --------------------------------------------------------------------------------------------------------------------
# values fixed on validation
# --------------------------------------------------------------------------------------------------------------------
def fixed_from_run(panel: Panel, hyp: dict) -> dict:
    """What a validation run fixes for the test run (prereg `fix_on_val`): written to fixed_on_val.json."""
    out = {"source": "validation dry run", "n_val_alerts": int(len(panel.y)), **panel.fixed_derived}
    cl = panel.cluster_info
    if cl["largest_share"] > 0.2 and cl.get("hub_cap") is None:
        out["hub_cap"] = None
        out["hub_cap_note"] = "largest cluster share > 0.2: choose a hub cap on validation (ftm.eval.bootstrap.cluster_ids hub_cap)"
    else:
        out["hub_cap"] = cl.get("hub_cap")
    out["largest_cluster_share"] = cl["largest_share"]
    for k in ("best_simple_planner", "equivalence_margin", "margin_gap", "best_ported_external"):
        out.setdefault(k, panel.fixed.get(k))
    return out


def counts_by_group(rows, key: str) -> dict:
    return dict(Counter(r[key] for r in rows))
