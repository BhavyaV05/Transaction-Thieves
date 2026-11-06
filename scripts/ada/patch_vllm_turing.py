"""Patch the installed vLLM 0.31.0 Triton unified-attention launch so Gemma 4's head_size-512 global-attention layers fit the 64 KB shared memory of an sm_75 GPU.  Owner: T3 (Akshat).

    python scripts/ada/patch_vllm_turing.py <venv>

Without it `vllm serve google/gemma-4-12B-it-qat-w4a16-ct` loads on a 2080 Ti and dies in the first attention call (cudagraph profiling) with
`triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 98304, Hardware limit: 65536` (prefill tile 32 x head 512 x fp16, default pipeline stages).
The patch changes launch parameters only (KV tile size and Triton num_stages for head_size >= 512 on compute capability < 8.0); the math is unchanged.
Idempotent; knobs for experiments: env VLLM_T75_TILE, VLLM_T75_STAGES (read at call time).
"""

import glob
import sys

path = glob.glob(f"{sys.argv[1]}/lib/python3.*/site-packages/vllm/v1/attention/ops/triton_unified_attention.py")[0]
s = open(path).read()
if "T75_PATCH" in s:
    print("already patched", path)
    sys.exit(0)
old = "    launch_kwargs: dict[str, int] = {}\n"
new = (
    "    # T75_PATCH (ada, T3): sm_75 has 64 KB shared memory; head_size 512 needs a smaller KV tile and fewer pipeline stages\n"
    "    if head_size >= 512 and torch.cuda.get_device_capability()[0] < 8:\n"
    "        import os as _os\n\n"
    "        tile_size = int(_os.environ.get('VLLM_T75_TILE', 16))\n"
    "        launch_num_stages = int(_os.environ.get('VLLM_T75_STAGES', 2))\n"
    "    launch_kwargs: dict[str, int] = {}\n"
)
assert s.count(old) == 1
open(path, "w").write(s.replace(old, new))
print("patched", path)
