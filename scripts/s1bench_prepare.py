"""S1BENCH data preparation (CPU): raw pinned downloads -> prepared JSONL the scorers read.  Owner: T3 (Akshat).

    python scripts/s1bench_prepare.py download            # snapshot_download every pinned dataset (retries; DNS here is flaky)
    python scripts/s1bench_prepare.py external            # the six public tasks -> prepared/<task>/{eval,calib}.jsonl + meta.json
    python scripts/s1bench_prepare.py s1mb                # S1MB parquet -> prepared/s1mb/<benchmark>.jsonl (one line per case x decision)
    python scripts/s1bench_prepare.py published           # the published predictions/metrics (s1mb-result) of the models in models.yaml

Nothing here reads OUR data.  S1MB targets are kept in the same line as the inference input only because the scorer needs both; the
model-facing code (`ftm.fastmodel.s1bench_s1mb.render`) receives the input fields only and a test checks that it never sees `target`.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CFG = yaml.safe_load((ROOT / "configs" / "s1bench" / "datasets.yaml").read_text())
CACHE = Path(CFG["cache_root"])
PREP = Path(CFG["prepared_root"])


def _retry(fn, tries: int = 40):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # DNS in this environment drops intermittently
            print("retry", k, str(e)[:100], flush=True)
            time.sleep(8)
    raise SystemExit("download failed")


def cmd_download(a) -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import huggingface_hub as h

    for name, t in CFG["tasks"].items():
        pats = [t["eval"]["file"], t["calib"]["file"], "README.md"]
        _retry(lambda t=t, pats=pats: h.snapshot_download(t["repo"], repo_type="dataset", revision=t["revision"], allow_patterns=pats,
                                                          local_dir=CACHE / "hf" / t["repo"].replace("/", "__")))  # fmt: skip
        print("ok", name)
    s = CFG["s1mb"]
    _retry(lambda: h.snapshot_download(s["repo"], repo_type="dataset", revision=s["revision"], allow_patterns=["data/*", "*.json", "*.md", "_sources/*.json"],
                                       ignore_patterns=["_open_jev/*", "_build/*"], local_dir=CACHE / "hf" / s["repo"].replace("/", "__")))  # fmt: skip
    print("ok s1mb")


def _names(parquet: Path, col: str) -> list[str]:
    import pyarrow.parquet as pq

    f = json.loads(pq.read_schema(parquet).metadata[b"huggingface"])["info"]["features"][col]
    return (f.get("feature") or f)["names"]


def _read(path: Path, text: str, label: str):
    import polars as pl

    return (pl.read_csv(path) if path.suffix == ".csv" else pl.read_parquet(path)).select(text, label).to_dicts()


def cmd_external(a) -> None:
    random.seed(0)
    for name, t in CFG["tasks"].items():
        root = CACHE / "hf" / t["repo"].replace("/", "__")
        out = PREP / name
        out.mkdir(parents=True, exist_ok=True)
        kind = t["kind"]
        ef = root / t["eval"]["file"]
        if kind in ("multiclass", "multiclass_oos"):
            col = t["eval"]["label"]
            try:
                labels = [n.replace("_", " ") for n in _names(ef, col)]
            except Exception:
                labels = None
        elif kind == "multilabel_yesno":
            labels = [n for n in _names(ef, t["eval"]["label"])]
        else:
            labels = ["yes", "no"]
        rows = {}
        for part in ("eval", "calib"):
            d = t[part]
            data = _read(root / d["file"], d["text"], d["label"])
            if kind == "multiclass" and labels is None:  # string labels (HWU64): the label set is the sorted test labels
                labels = sorted({r[d["label"]] for r in data}) if part == "eval" else labels
            rows[part] = data
        lab_index = {name: i for i, name in enumerate(labels)} if labels and isinstance(rows["eval"][0][t["eval"]["label"]], str) else None
        meta = {"task": name, "kind": kind, "labels": labels, "question": t["question"], "repo": t["repo"], "revision": t["revision"],
                "licence": t["licence"], "oos_label": None}  # fmt: skip
        if kind == "multiclass_oos":
            labels = ["out of scope" if n == "oos" else n for n in labels]
            meta["labels"] = labels
            meta["oos_label"] = labels.index("out of scope") if "out of scope" in labels else None
        for part in ("eval", "calib"):
            d = t[part]
            data = rows[part]
            if part == "calib":  # a fixed 500-item subsample (the cap), never the eval split
                data = random.Random(0).sample(data, min(CFG["calib_cap"], len(data)))
            with open(out / f"{part}.jsonl", "w") as f:
                for i, r in enumerate(data):
                    y = r[d["label"]]
                    if kind == "binary_yesno":
                        lab = 0 if int(y) == 1 else 1  # index into ("yes", "no")
                    elif kind == "multilabel_yesno":
                        lab = sorted(int(v) for v in y)  # the list of unfair types present
                    else:
                        lab = lab_index[y] if lab_index else int(y)
                    f.write(json.dumps({"id": f"{name}:{part}:{i}", "text": r[d["text"]], "label": lab}) + "\n")
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(name, {p: len(rows[p]) for p in rows}, "labels", len(labels or []))


def _norm(config: str) -> str:
    return config.lower().replace("__", "--").replace("_", "-")


def _text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def cmd_s1mb(a) -> None:
    import polars as pl

    root = CACHE / "hf" / CFG["s1mb"]["repo"].replace("/", "__")
    out = PREP / "s1mb"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    # Only training-manifest.json defines the active evaluation membership (quarantined Parquet stays out).
    active = {e["dataset"] for e in json.loads((root / "training-manifest.json").read_text())["evaluation"]}
    for cfgdir in sorted((root / "data").iterdir()):
        if cfgdir.name not in active:
            continue
        files = list(cfgdir.rglob("*.parquet"))
        if not files:
            continue
        df = pl.read_parquet(files[0])
        by_type: dict[str, list] = {}
        for r in df.iter_rows(named=True):
            targets = {t["decision_id"]: t for t in r["targets"]}
            state = json.loads(r["input"]["state_json"])
            for d in r["input"]["decisions"]:
                tg = targets[d["id"]]
                line = {
                    "case_id": r["case_id"], "group_id": r["group_id"], "input_hash": r["input_hash"], "question_id": d["id"], "task": d["type"],
                    "state": state,
                    "system_prompt": d["system_prompt"],
                    "instructions": json.loads(d["instructions_json"]),
                    "options": [{"id": c["id"], "description": json.loads(c["description_json"]), "value": c["value"]} for c in d["criteria"]],
                    "target": dict(zip(tg["ids"], tg["probabilities"], strict=True)),
                }  # fmt: skip
                by_type.setdefault(d["type"], []).append(line)
        for typ, lines in by_type.items():
            bid = f"{_norm(cfgdir.name)}-{typ}-test-v1"
            with open(out / f"{bid}.jsonl", "w") as f:
                for ln in lines:
                    f.write(json.dumps(ln, ensure_ascii=False) + "\n")
            manifest[bid] = {"config": cfgdir.name, "task": typ, "judgments": len(lines)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "benchmarks,", sum(m["judgments"] for m in manifest.values()), "judgments")


def cmd_published(a) -> None:
    """Predictions + stored metrics of the published models -> prepared/published/<model>/<benchmark>.json (predictions as {case|qid: probs})."""
    models = yaml.safe_load((ROOT / "configs" / "s1bench" / "models.yaml").read_text())["published_rows"]
    src = CACHE / "hf" / "hotchpotch__s1mb-result"
    for m in models:
        d = src / m.replace("/", "__")
        out = PREP / "published" / m.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("*.json.xz")):
            j = json.load(lzma.open(f))
            preds = {f"{p['case_id']}|{p['question_id']}": p.get("probabilities") for p in j["predictions"]}
            (out / (j["benchmark"]["id"] + ".json")).write_text(json.dumps(
                {"benchmark": j["benchmark"]["id"], "status": j["status"], "counts": j["counts"], "metrics": j["metrics"], "predictions": preds,
                 "abstentions": [f"{p['case_id']}|{p['question_id']}" for p in j["predictions"] if p.get("abstention")], "elapsed_seconds": j.get("elapsed_seconds")}))  # fmt: skip
        print(m, len(list(out.glob("*.json"))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "external", "s1mb", "published"])
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
