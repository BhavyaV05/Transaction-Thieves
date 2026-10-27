"""QLoRA SFT of the planner (distilled teacher steps) and of the 4B fast model (small-decision dataset).

Owner: T3 (Akshat).  Spec: tracks.md 5.4 (steps 1-4) and 5.6; plan.md 6.11 and 6.14.

    # 1. distil teacher traces (run_llm_agent.py --keep-prompts on TRAIN alerts) into SFT examples
    python scripts/train_planner.py distill --runs outputs/t3_teacher/runs/c0?? --data hi_medium \
        --backtest data/backtest/hi_medium/7df6bad5337d/lr_table.jsonl [--fusion-params outputs/fusion/val_fit.json] \
        --teacher-model gemma4-12b --teacher-revision <40-char hash> --out outputs/sft/planner_train.jsonl
    # (no --fusion-params: the naive summed fusion the teacher runs and the final run use)
    # 2. check config + data without a GPU, then train one seed (block GPU_CALENDAR.md first)
    python scripts/train_planner.py train --config configs/llm/sft_planner.yaml --dataset outputs/sft/planner_train.jsonl --dry-run
    python scripts/train_planner.py train --config configs/llm/sft_planner.yaml --dataset outputs/sft/planner_train.jsonl --seed 0
    # 3. fast model: temperature on val `fit`, conformal sets (fast + 8B) on val `cal`, report on `eval`
    python scripts/train_planner.py calibrate --dataset outputs/fast/val_decisions.jsonl --fast-url http://... \
        --large-url http://... --out outputs/fast/calibration.json

The trainer (transformers + PEFT + TRL + bitsandbytes) is imported lazily, so `--dry-run`, `distill` and the tests need no GPU packages.
Environment: ~/projects/train-venv (versions in docs/t3_finetune_fastmodel.md).
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


OPTIM = {"adamw_8bit": "adamw_bnb_8bit"}  # config name (Unsloth's) -> transformers' name


def render_prompt_completion(tok, messages: list[dict], enable_thinking: bool) -> tuple[str, str]:
    """(prompt text, completion text) of one chat example, rendered exactly as inference sees it: the prompt ends with the
    generation prompt (Qwen3 with thinking off: `<|im_start|>assistant\n<think>\n\n</think>\n\n`) and the completion is the
    assistant JSON (or the one label token) plus `<|im_end|>`. Loss on the completion only (responses_only)."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    return prompt, messages[-1]["content"] + "<|im_end|>"


def train(cfg: dict, rows: list[dict], seed: int) -> Path:  # pragma: no cover - needs a GPU
    """QLoRA SFT with plain transformers + PEFT + TRL + bitsandbytes (NF4, double quant, bf16 compute). Unsloth is not used:
    its release that resolves against torch 2.8 / sm_120 here is mid-2025 and does not run with transformers 5
    (docs/t3_finetune_fastmodel.md, environment). Same algorithm: 4-bit frozen base, LoRA on all linear layers."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["revision"])
    bnb = BitsAndBytesConfig(load_in_4bit=cfg.get("load_in_4bit", True), bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)  # fmt: skip
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg["revision"], quantization_config=bnb,
                                                 dtype=torch.bfloat16, device_map={"": 0})  # fmt: skip
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lo = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"], bias="none",
                                             target_modules=lo["target_modules"], task_type="CAUSAL_LM"))  # fmt: skip
    pairs = [render_prompt_completion(tok, r["messages"], cfg.get("enable_thinking", False)) for r in rows]
    ds = Dataset.from_dict({"prompt": [p for p, _ in pairs], "completion": [c for _, c in pairs]})
    out = ROOT / cfg["output_dir"] / f"seed={seed}"
    t = cfg["train"]
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds,
        args=SFTConfig(max_length=cfg["max_seq_length"], num_train_epochs=t["epochs"], learning_rate=t["learning_rate"],
                       per_device_train_batch_size=t["per_device_batch_size"], gradient_accumulation_steps=t["grad_accum"],
                       warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"], optim=OPTIM.get(t["optim"], t["optim"]),
                       completion_only_loss=bool(t.get("responses_only", True)), bf16=True, gradient_checkpointing=True,
                       seed=seed, data_seed=seed, logging_steps=10, output_dir=str(out / "checkpoints"), save_strategy="no",
                       report_to=[]),
    )  # fmt: skip
    trainer.train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    (out / "train_log.json").write_text(json.dumps({"seed": seed, "examples": len(rows), "log_history": trainer.state.log_history}, indent=1),
                                        encoding="utf-8")
    return out


def cmd_distill(a) -> int:
    import polars as pl

    from ftm.llm.backtest_table import TableAcceptance, load_backtest
    from ftm.llm.fusion_adapter import FusedLR
    from ftm.planners.llm_distill import Teacher, distill_examples

    labels = {}
    for split in a.splits:
        df = pl.read_parquet(ROOT / "data" / "alerts" / a.data / f"{split}.parquet", columns=["alert_id", "label", "split"])  # time-guard: not transactions
        labels.update({r["alert_id"]: r for r in df.to_dicts()})
    try:  # the mode is read from T2's cache entries beside the table; --backtest-mode is only a cross-check
        bt = load_backtest(a.backtest, mode=a.backtest_mode, require="realistic")
    except ValueError as e:
        raise SystemExit(str(e)) from e
    table = bt.table
    traces = [json.loads(x) for run in a.runs for x in (run / "traces.jsonl").read_text(encoding="utf-8").splitlines()]

    runs_model = {json.loads((run / "summary.json").read_text(encoding="utf-8")).get("model") for run in a.runs}
    if runs_model != {a.teacher_model}:
        raise SystemExit(f"runs were made with {sorted(map(str, runs_model))}, not --teacher-model {a.teacher_model!r}")
    if a.fusion_params is not None:
        fusion = FusedLR.from_params(json.loads(a.fusion_params.read_text(encoding="utf-8")), table)  # fitted on val (T4)
    else:  # the naive summed fusion every run uses until the team adopts the fitted one (configs/final/jobs.yaml)
        from ftm.llm.investigation import NaiveLRFusion

        fusion = NaiveLRFusion(table)
    examples, counts = distill_examples(traces, labels, TableAcceptance(bt), fusion=fusion,
                                        teacher=Teacher(a.teacher_model, a.teacher_revision), splits=tuple(a.splits),
                                        allow_naive=a.fusion_params is None)  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    summary = {"out": str(a.out), "runs": [str(r) for r in a.runs], "backtest": str(a.backtest), "backtest_mode": bt.mode,
               "backtest_namespace": bt.namespace, "backtest_accepted_by_eps": bt.summary()["accepted_by_eps"],
               "fusion_params": str(a.fusion_params) if a.fusion_params else "naive", **counts}  # fmt: skip
    (a.out.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_fast_data(a) -> int:
    """T2's small-decision TRAIN rows -> chat examples (`to_sft_example`: the assistant turn is one label token)."""
    from collections import Counter

    from ftm.fastmodel.decisions import DecisionExample, to_sft_example

    n, by = 0, Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.decisions.open(encoding="utf-8") as f, a.out.open("w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "train":
                raise SystemExit(f"{a.decisions}: row with split {r.get('split')!r}; the fast model trains on train rows only")
            g.write(json.dumps(to_sft_example(DecisionExample(r["decision"], r["context"], r["label"], r.get("alert_id"), "train"))) + "\n")
            n, by[r["decision"]] = n + 1, by[r["decision"]] + 1
    summary = {"out": str(a.out), "decisions": str(a.decisions), "examples": n, "by_decision": dict(by)}
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
    adapter = ROOT / cfg["output_dir"] / f"seed={a.seed}" / "adapter"
    if a.skip_existing and (adapter / "adapter_config.json").exists():
        print(f"seed {a.seed}: {adapter} exists, skipped")
        return 0
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
    d.add_argument("--backtest", type=Path, required=True, help="assembled back-test dir (or its lr_table.jsonl)")
    d.add_argument("--backtest-mode", default=None,
                   help="cross-check, or the stated mode of a table copied away from its cache; read from the cache otherwise")
    d.add_argument("--fusion-params", type=Path, default=None,
                   help="Fusion.params() JSON of the model fitted on val; default: the naive summed fusion (the one the teacher runs used)")
    d.add_argument("--teacher-model", required=True)
    d.add_argument("--teacher-revision", required=True, help="weights revision (commit hash) of the served teacher")
    d.add_argument("--out", type=Path, required=True)
    fd = sub.add_parser("fast-data")
    fd.add_argument("--decisions", type=Path, default=ROOT / "data" / "decisions" / "hi_medium" / "decisions_train.jsonl")
    fd.add_argument("--out", type=Path, default=ROOT / "outputs" / "sft" / "fast_train.jsonl")
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
    t.add_argument("--skip-existing", action="store_true", help="a seed whose adapter is already saved is not trained again (resume)")
    a = ap.parse_args(argv)
    return {"distill": cmd_distill, "fast-data": cmd_fast_data, "calibrate": cmd_calibrate, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
