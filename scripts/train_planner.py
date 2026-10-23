"""QLoRA SFT of the planner (distilled teacher steps) and of the 4B fast model (small-decision dataset).

Owner: T3 (Akshat).  Spec: tracks.md 5.4 (steps 1-4) and 5.6; plan.md 6.11 and 6.14.

    # 1. distil teacher traces (run_llm_agent.py --keep-prompts on TRAIN alerts) into SFT examples
    python scripts/train_planner.py distill --runs outputs/runs/<teacher1> ... --data hi_small \
        --backtest outputs/backtest/lr_table.jsonl --backtest-mode realistic --fusion-params outputs/fusion/val_fit.json \
        --teacher-model <served model name> --teacher-revision <40-char hash> --out outputs/sft/planner_train.jsonl
    # 2. check config + data without a GPU, then train one seed (block GPU_CALENDAR.md first)
    python scripts/train_planner.py train --config configs/llm/sft_planner.yaml --dataset outputs/sft/planner_train.jsonl --dry-run
    python scripts/train_planner.py train --config configs/llm/sft_planner.yaml --dataset outputs/sft/planner_train.jsonl --seed 0
    # 3. fast model: temperature on val `fit`, conformal sets (fast + 8B) on val `cal`, report on `eval`
    python scripts/train_planner.py calibrate --dataset outputs/fast/val_decisions.jsonl --fast-url http://... \
        --large-url http://... --out outputs/fast/calibration.json

The trainer (Unsloth + TRL) is imported lazily, so `--dry-run`, `distill` and the tests need no GPU packages.
NOT YET RUN ON A GPU: the Unsloth/TRL calls follow their documented APIs and must be checked on the first real run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf  # noqa: E402

from ftm.llm.budget import estimate_messages  # noqa: E402

REQUIRED = ("task", "base_model", "revision", "max_seq_length", "lora", "train", "seeds", "output_dir")


def load_config(path: Path) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}")
    if cfg["task"] not in ("planner", "fast_model"):
        raise SystemExit(f"{path}: task must be planner or fast_model")
    if len(cfg["seeds"]) < 3:
        raise SystemExit(f"{path}: plan rule is at least 3 seeds for anything trained")
    return cfg


def load_dataset(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        if not msgs or msgs[-1].get("role") != "assistant" or not msgs[-1].get("content"):
            raise SystemExit(f"{path}:{i + 1}: each example needs messages ending in a non-empty assistant turn")
    if not rows:
        raise SystemExit(f"{path}: no examples")
    return rows


def data_report(rows: list[dict], cfg: dict) -> dict:
    lens = sorted(estimate_messages(r["messages"]) for r in rows)
    too_long = sum(n > cfg["max_seq_length"] for n in lens)
    return {"examples": len(rows), "est_tokens_median": statistics.median(lens), "est_tokens_max": lens[-1],
            "over_max_seq_length": too_long, "alerts": len({r.get("alert_id") for r in rows})}  # fmt: skip


def train(cfg: dict, rows: list[dict], seed: int) -> Path:  # pragma: no cover - needs a GPU
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    model, tok = FastLanguageModel.from_pretrained(model_name=cfg["base_model"], revision=cfg["revision"],
                                                   max_seq_length=cfg["max_seq_length"],
                                                   load_in_4bit=cfg.get("load_in_4bit", True))  # fmt: skip
    lo = cfg["lora"]
    model = FastLanguageModel.get_peft_model(model, r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                                             target_modules=lo["target_modules"], bias="none",
                                             use_gradient_checkpointing="unsloth", random_state=seed)  # fmt: skip
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=cfg.get("enable_thinking", False))
             for r in rows]  # fmt: skip
    out = ROOT / cfg["output_dir"] / f"seed={seed}"
    t = cfg["train"]
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=Dataset.from_dict({"text": texts}),
        args=SFTConfig(dataset_text_field="text", max_length=cfg["max_seq_length"], num_train_epochs=t["epochs"],
                       learning_rate=t["learning_rate"], per_device_train_batch_size=t["per_device_batch_size"],
                       gradient_accumulation_steps=t["grad_accum"], warmup_ratio=t["warmup_ratio"],
                       weight_decay=t["weight_decay"], optim=t["optim"], seed=seed, logging_steps=10,
                       output_dir=str(out / "checkpoints"), save_strategy="steps", save_steps=200, report_to=[]),
    )  # fmt: skip
    if t.get("responses_only", True):
        trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                          response_part="<|im_start|>assistant\n")  # fmt: skip
    trainer.train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    return out


class TableAcceptance:
    """RealisticAcceptance over a back-test table. Pending: T2's own adapter (PR #24) will replace this; until then the
    caller states the mode the table was made in and anything but realistic is refused."""

    def __init__(self, table, mode: str):
        if mode != "realistic":
            raise SystemExit(f"distillation needs a realistic-mode back-test table, not {mode!r} (exact mode ignores the noise)")
        self.table = table

    def __call__(self, canonical: str, eps: float) -> bool:
        entry = self.table.get(canonical, eps)
        return entry is not None and entry.accepted


def cmd_distill(a) -> int:
    import polars as pl

    from ftm.llm.fusion_adapter import FusedLR
    from ftm.llm.investigation import LRTable
    from ftm.planners.llm_distill import Teacher, distill_examples

    labels = {}
    for split in a.splits:
        df = pl.read_parquet(ROOT / "data" / "alerts" / a.data / f"{split}.parquet", columns=["alert_id", "label", "split"])  # time-guard: not transactions
        labels.update({r["alert_id"]: r for r in df.to_dicts()})
    table = LRTable.from_rows(json.loads(x) for x in a.backtest.read_text(encoding="utf-8").splitlines() if x.strip())
    traces = [json.loads(x) for run in a.runs for x in (run / "traces.jsonl").read_text(encoding="utf-8").splitlines()]

    runs_model = {json.loads((run / "summary.json").read_text(encoding="utf-8")).get("model") for run in a.runs}
    if runs_model != {a.teacher_model}:
        raise SystemExit(f"runs were made with {sorted(map(str, runs_model))}, not --teacher-model {a.teacher_model!r}")
    fusion = FusedLR.from_params(json.loads(a.fusion_params.read_text(encoding="utf-8")), table)  # fitted on val (T4)
    examples, counts = distill_examples(traces, labels, TableAcceptance(table, a.backtest_mode), fusion=fusion,
                                        teacher=Teacher(a.teacher_model, a.teacher_revision), splits=tuple(a.splits))  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    summary = {"out": str(a.out), "runs": [str(r) for r in a.runs], "backtest": str(a.backtest), "backtest_mode": a.backtest_mode,
               "fusion_params": str(a.fusion_params), **counts}  # fmt: skip
    (a.out.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_calibrate(a) -> int:
    from ftm.fastmodel.decisions import DecisionExample
    from ftm.fastmodel.model import JsonDecisionModel, LogprobFastModel
    from ftm.fastmodel.splits import calibrate_cascade
    from ftm.llm.client import HttpChatClient

    rows = [json.loads(x) for x in a.dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
    examples = [DecisionExample(r["decision"], r["context"], r["label"], r.get("alert_id"), r.get("split")) for r in rows]
    fast = LogprobFastModel(HttpChatClient(a.fast_url, model=a.fast_model))
    large = JsonDecisionModel(HttpChatClient(a.large_url, model=a.large_model))
    temps, _, report = calibrate_cascade(examples, [r["val_part"] for r in rows], fast, large, alpha=a.alpha)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"temperatures": temps, **report}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["flagged"]:
        print(f"WARNING: full set (always defer) or no fit rows for: {report['flagged']}")
    return 0


def cmd_train(a) -> int:
    cfg = load_config(a.config)
    rows = load_dataset(a.dataset)
    report = {"config": str(a.config), "task": cfg["task"], "base_model": cfg["base_model"], **data_report(rows, cfg)}
    print(json.dumps(report, indent=2))
    if report["over_max_seq_length"]:
        print(f"WARNING: {report['over_max_seq_length']} examples may exceed max_seq_length (estimate)")
    if a.dry_run:
        return 0
    if a.seed not in cfg["seeds"]:
        raise SystemExit(f"seed {a.seed} is not in the config seeds {cfg['seeds']}")
    print("Reminder: QLoRA takes the whole GPU. Is the window blocked in GPU_CALENDAR.md?")
    print(f"saved {train(cfg, rows, a.seed)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("distill")
    d.add_argument("--runs", type=Path, nargs="+", required=True)
    d.add_argument("--data", default="hi_small")
    d.add_argument("--splits", nargs="+", default=["train"])
    d.add_argument("--backtest", type=Path, required=True)
    d.add_argument("--backtest-mode", required=True, help="mode the table was made in; only 'realistic' is accepted")
    d.add_argument("--fusion-params", type=Path, required=True, help="Fusion.params() JSON of the model fitted on val")
    d.add_argument("--teacher-model", required=True)
    d.add_argument("--teacher-revision", required=True, help="weights revision (commit hash) of the served teacher")
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--dataset", type=Path, required=True, help="JSONL rows: decision, context, label, val_part (fit/cal/eval)")
    c.add_argument("--fast-url", required=True)
    c.add_argument("--fast-model", default=None)
    c.add_argument("--large-url", required=True)
    c.add_argument("--large-model", default=None)
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--dataset", type=Path, required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return {"distill": cmd_distill, "calibrate": cmd_calibrate, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
