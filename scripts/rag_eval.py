"""Regulatory RAG evaluation (T4). CPU only, seconds.

    python scripts/rag_eval.py                        # retrieval recall@k: BM25 vs the v0 keyword match
    python scripts/rag_eval.py --cases-out data/runs/citation_cases.jsonl   # also write the 100 citation cases

Writes data/runs/rag_eval.json. Citation accuracy of a system is scored with
ftm.compliance.rag_eval.citation_accuracy on its cited rule ids for these cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftm.compliance import rag_eval  # noqa: E402
from ftm.compliance.gate import Gate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/runs/rag_eval.json")
    ap.add_argument("--cases-out", default=None)
    a = ap.parse_args()
    res = {}
    for mode in ("bm25", "keyword"):
        g = Gate.from_config(retrieval=mode)
        res[mode] = [rag_eval.retrieval_eval(lambda q, k, g=g: g.lookup_rules(q, k=k), k=k) for k in (3, 5, 10)]
    cases = rag_eval.citation_cases()
    res["citation_cases"] = {"n": len(cases), "expected_rules": sorted({c["expected_rule_id"] for c in cases})}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    if a.cases_out:
        Path(a.cases_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.cases_out).write_text("".join(json.dumps(c) + "\n" for c in cases))
    for mode in ("bm25", "keyword"):
        print(mode, " ".join(f"recall@{r['k']}={r['recall_at_k']:.3f}" for r in res[mode]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
