"""Dense half of the hybrid regulatory retriever (plan.md 6.7): bge-small on CPU through sentence-transformers.

Owner: T4.  The model and its revision are pinned (the HF snapshot hash), the weights come from the HF cache (`FTM_HF_HOME`
if set, else the default cache). `sentence-transformers` is imported lazily: without it `retrieval: bm25` is unaffected and
`retrieval: hybrid` raises a clear error. Tests pass a stub `embedder(texts) -> vectors` instead.
"""

from __future__ import annotations

import os
from pathlib import Path

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def hf_cache_dir() -> str | None:
    """FTM_HF_HOME is an HF_HOME-style directory (docs/hardware.md): the hub cache sits in `hub/` when that exists."""
    home = os.environ.get("FTM_HF_HOME")
    if not home:
        return None
    hub = Path(home) / "hub"
    return str(hub if hub.exists() else home)


def bge_embedder(model: str = BGE_MODEL, revision: str = BGE_REVISION, device: str = "cpu"):
    """`embedder(texts) -> list[vector]` with L2-normalised bge-small embeddings (cosine = dot)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("retrieval: hybrid needs sentence-transformers (uv pip install sentence-transformers)") from e
    st = SentenceTransformer(model, revision=revision, cache_folder=hf_cache_dir(), device=device)

    def embed(texts: list[str]) -> list[list[float]]:
        return st.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).tolist()

    return embed
