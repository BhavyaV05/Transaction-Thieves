import ast
from pathlib import Path

import pytest


@pytest.mark.skip(reason="T2, week 2: static check that no code path in ftm/banks returns raw rows")
def test_no_code_path_returns_raw_rows():
    ...


def test_every_tool_applies_time_guard():
    """T1: transaction Parquet is read only through ftm/data/io.py (which applies the guards).

    Nothing under ftm/ outside ftm/data/ may scan/read Parquet or import duckdb. ftm/data/ build
    steps read their own freshly written output; io.py must call both guards.
    A line that reads other Parquet (run logs, back-test caches) is allowed if it ends with the
    marker `# time-guard: not transactions`.
    """
    root = Path(__file__).resolve().parents[1] / "ftm"
    banned_calls = {"scan_parquet", "read_parquet", "read_table", "ParquetFile"}
    offenders = []
    for path in root.rglob("*.py"):
        if path.relative_to(root).parts[0] == "data":
            continue
        src = path.read_text()
        exempt = {
            i for i, line in enumerate(src.splitlines(), 1) if "# time-guard: not transactions" in line
        }
        for node in ast.walk(ast.parse(src)):
            if getattr(node, "lineno", None) in exempt:
                continue
            if isinstance(node, ast.Attribute) and node.attr in banned_calls:
                offenders.append(f"{path.relative_to(root)}:{node.lineno} {node.attr}")
            if isinstance(node, ast.Name) and node.id in banned_calls:
                offenders.append(f"{path.relative_to(root)}:{node.lineno} {node.id}")
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] in {"duckdb", "pyarrow"} for n in names):
                offenders.append(f"{path.relative_to(root)}:{node.lineno} import {names}")
    assert not offenders, f"read transactions via ftm.data.io only: {offenders}"
    io_src = (root / "data" / "io.py").read_text()
    assert "enforce_as_of" in io_src and "require_final_run" in io_src


@pytest.mark.skip(reason="T2, week 4: budgets and the gate cannot be bypassed through the harness")
def test_harness_cannot_bypass_gate_or_accountant():
    ...
