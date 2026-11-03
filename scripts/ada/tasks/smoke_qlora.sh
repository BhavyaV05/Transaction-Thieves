#!/bin/bash
# Qwen3-8B QLoRA (NF4, r16, all projections, max_seq_length 7168, batch 1, gradient checkpointing) in fp16 on one RTX 2080 Ti (11 GB): 3 real steps.  1 GPU.
source $(dirname $0)/common.sh; D=$RES/smoke_qlora; mkdir -p $D; cd $HOME/tt/ada_scripts
export HF_HUB_ENABLE_HF_TRANSFER=0
${VENV:-$HOME/venv312}/bin/hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 > $D/download.log 2>&1
for seq in 7168 4096 2048; do   # the real length first; shorter ones only to find where it stops fitting
  $PY smoke_qlora_fp16.py --model Qwen/Qwen3-8B --seq $seq --steps 3 2>&1 | tee $D/seq$seq.log | grep -E "RESULT|Error|error" ; grep -q '"ok": true' $D/seq$seq.log && break
done
cat $D/seq*.log | grep RESULT > $D/result.txt
