"""Group A core: Multi-GNN (Egressy et al., AAAI 2024), official IBM code. Reproduce on its split, then our split, one-bank and see-everything.

Owner: T1 (Abhishek). See tracks.md section 3.

The IBM code lives OUTSIDE the repo (Apache-2.0, https://github.com/IBM/Multi-GNN, commit
252b0252afca109d1d216c411c59ff70753b25fc) in ~/Documents/t1_external/Multi-GNN. Only this thin launcher and
the compatibility patch `multi_gnn_t1.patch` (apply with `git apply` inside that clone) are in the repo.
Status: the reproduction on the paper's own split is wired (scripts/train_gnn.py --repro); the
our-split, one-bank and see-everything runs are not done.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

MULTI_GNN_COMMIT = "252b0252afca109d1d216c411c59ff70753b25fc"
# Multi-PNA with edge updates (EU), reverse message passing, ego IDs and port numbering = the paper's best HI-Small model.
PNA_EU_ARGS = ["--model", "pna", "--emlps", "--reverse_mp", "--ego", "--ports"]


def repro_command(data_name: str = "Small_HI", seed: int = 1, batch_size: int = 2048, epochs: int = 100) -> list[str]:
    """Command line of the reproduction run (run it from the Multi-GNN clone)."""
    return ["python3.12", "main.py", "--data", data_name, *PNA_EU_ARGS, "--seed", str(seed), "--testing", "--save_model",
            "--unique_name", "--n_epochs", str(epochs), "--batch_size", str(batch_size)]  # fmt: skip


def launch(clone: Path, metrics_file: Path, budget_s: int, **kw) -> subprocess.Popen:
    """Start the run in the background with the patch's time budget and per-epoch metrics file."""
    env = {**os.environ, "T1_BUDGET_S": str(budget_s), "T1_METRICS": str(metrics_file),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # fmt: skip
    return subprocess.Popen(repro_command(**kw), cwd=clone, env=env)
