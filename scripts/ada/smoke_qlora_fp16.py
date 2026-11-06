"""Smoke test for the cluster `ada` (RTX 2080 Ti, sm_75, 11 GB): does the planner QLoRA recipe (Qwen3-8B, NF4, LoRA r16 on all
projections, max_seq_length 7168, batch 1, gradient checkpointing) train in fp16 in 11 GB?  Owner: T3 (Akshat).

No bf16 on Turing, so the compute dtype is float16 with a GradScaler (LoRA weights in fp32).  Random token ids of the full length
(the memory and speed do not depend on the content); loss on the last 256 positions, like a response-only loss.  Prints peak memory and
seconds per optimizer step as one JSON line.  Not a training run: no data, no checkpoint, no result about the model.

    python scripts/ada/smoke_qlora_fp16.py --model Qwen/Qwen3-8B --seq 7168 --steps 3
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seq", type=int, default=7168)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, dtype=torch.float16, device_map={"": 0}, attn_implementation=a.attn)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4)
    scaler = torch.amp.GradScaler("cuda")
    ids = torch.randint(1000, 100000, (1, a.seq), device="cuda")
    labels = ids.clone()
    labels[:, : -256] = -100
    out = {"model": a.model, "seq": a.seq, "attn": a.attn, "gpu": torch.cuda.get_device_name(0), "load_s": round(load_s, 1), "steps": []}
    try:
        for _i in range(a.steps):
            t1 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            out["steps"].append({"loss": float(loss), "s": round(time.time() - t1, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        out["ok"] = True
    except torch.cuda.OutOfMemoryError as e:
        out["ok"], out["error"] = False, "CUDA OOM: " + str(e)[:200]
    out["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
