"""S1BENCH Part B: fine-tune an open System One model on OUR TRAIN decision rows (never val, never test).  Owner: T3 (Akshat).

Backend implemented: `gliner25-decide` (fastino/GLiNER2.5-Decide, 340M; documented trainer `gliner2.training.GLiNER2Trainer` with LoRA).  The other small
open models are not trainable from their published packages: laya's recipe is a Kaggle notebook (github NandhaKishorM/laya) and bekko's training code is a separate
repo; they would need their own entry point, so they are reported as zero-shot only (docs/t3_s1bench.md section 3).

    python scripts/s1bench_finetune.py gliner25-decide --cap 1000 --epochs 1 --batch-size 8 --lora --out outputs/s1bench/ft/gliner25-ft    # CPU (venv-open) or GPU
    python scripts/s1bench_partb.py open --model gliner25-decide --adapter outputs/s1bench/ft/gliner25-ft --out outputs/s1bench/partb/gliner25-ft/scores.jsonl

The training example is built exactly like the zero-shot input (`ftm.fastmodel.s1bench_adapters.gliner_input`): text = "Question: <instruction>" + the decision
context, one classification task "decision" over the option keys with the shared option descriptions.  `--cap` keeps at most N train rows per decision (seeded).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.fastmodel.decisions import DECISIONS  # noqa: E402
from ftm.fastmodel.s1bench_adapters import TypedQuestion, gliner_input  # noqa: E402

MODELS = yaml.safe_load((ROOT / "configs" / "s1bench" / "models.yaml").read_text())
DEFS = yaml.safe_load((ROOT / "configs" / "s1bench" / "decisions.yaml").read_text())
TRAIN = ROOT / "data" / "decisions" / "hi_medium_s1v1" / "decisions_train.jsonl"


def train_rows(path: Path, cap: int | None, seed: int) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["decision"] in DECISIONS]
    if any(r.get("split") != "train" for r in rows):
        raise SystemExit("fine-tuning reads TRAIN rows only")
    if cap:
        rng = random.Random(seed)
        by: dict[str, list[dict]] = {}
        for r in rows:
            by.setdefault(r["decision"], []).append(r)
        rows = [r for d in sorted(by) for r in (by[d] if len(by[d]) <= cap else rng.sample(by[d], cap))]
        rng.shuffle(rows)
    return rows


def example_for(row: dict) -> dict:
    """One gliner2 training record: the zero-shot input plus the gold option key."""
    d = DEFS["decisions"][row["decision"]]
    q = TypedQuestion({DEFS["state_key"]: row["context"]}, d["instruction"], tuple((o["id"], o["description"]) for o in d["options"]))
    text, labels = gliner_input(q)
    keys = list(labels)
    idx = [o["id"] for o in d["options"]].index(row["label"])
    gold_key = keys[idx] if not q.is_yes_no else ("true" if row["label"] == "yes" else "false")
    return {"input": text, "output": {"classifications": [{"task": "decision", "labels": keys, "true_label": [gold_key], "label_descriptions": labels}]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["gliner25-decide"])
    ap.add_argument("--train", type=Path, default=TRAIN)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true", help="build the examples and print counts, no training")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = train_rows(a.train, a.cap, a.seed)
    recs = [example_for(r) for r in rows]
    print(json.dumps({"rows": len(recs), "by_decision": {d: sum(r["decision"] == d for r in rows) for d in DECISIONS}, "example": recs[0]["output"]}))
    if a.dry_run:
        return
    from gliner2 import AutoExtractor, TrainingDataset
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    m = MODELS["open_models"][a.model]
    model = AutoExtractor.from_pretrained(m["repo"], revision=m["revision"]).to(a.device)
    ds = TrainingDataset.from_records(recs)
    cfg = TrainingConfig(output_dir=str(a.out), experiment_name=a.model, num_epochs=a.epochs, max_steps=a.max_steps, batch_size=a.batch_size,
                         eval_strategy="no", use_lora=a.lora, seed=a.seed, num_workers=0, save_best=False, fp16=False, bf16=False)  # fmt: skip
    trainer = GLiNER2Trainer(model, cfg, train_data=ds)
    out = trainer.train()
    (a.out / "train_summary.json").write_text(json.dumps({"rows": len(recs), "config": {"cap": a.cap, "epochs": a.epochs, "batch_size": a.batch_size, "lora": a.lora,
                                                                                         "seed": a.seed}, "result": {k: v for k, v in (out or {}).items() if isinstance(v, int | float | str)}}, indent=1))  # fmt: skip
    print("saved", a.out)


if __name__ == "__main__":
    main()
