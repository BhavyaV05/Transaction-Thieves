import ast
from pathlib import Path

import pytest


def test_no_code_path_returns_raw_rows():
    """T2: nothing in ftm/banks can hand out ledger rows. The partner-bank package never calls a row-extracting
    Polars method, never returns the ledger, and its public methods are annotated with scalar-valued responses only
    (`bind_alert` builds slot sets from the HOME bank's own view and is exempt)."""
    root = Path(__file__).resolve().parents[1] / "ftm" / "banks"
    row_calls = {"rows", "iter_rows", "to_dicts", "to_pandas", "to_numpy", "row", "to_list", "to_arrow", "write_parquet"}
    frames = {"DataFrame", "LazyFrame", "Series"}
    offenders = []
    for path in root.glob("*.py"):
        if path.name == "binding.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in row_calls:
                offenders.append(f"{path.name}:{node.lineno} .{node.attr}")
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Attribute) and node.value.attr == "_ledger":
                offenders.append(f"{path.name}:{node.lineno} returns the ledger")
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and node.returns is not None:
                if frames & {n.id for n in ast.walk(node.returns) if isinstance(n, ast.Name)} | {
                    n.attr for n in ast.walk(node.returns) if isinstance(n, ast.Attribute)
                }:
                    offenders.append(f"{path.name}:{node.lineno} {node.name} is annotated to return a frame")
    assert not offenders, offenders


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


def test_harness_cannot_bypass_gate_or_accountant():
    """T2: every road to a partner bank goes through the gate and the accountant. Forbidden or over-budget asks never
    reach a bank, and the budget holds under random asking. (More cases: tests/test_harness.py.)"""
    import random
    from datetime import datetime

    from ftm.compliance.gate import BudgetRefused, Gate, GateBlocked
    from ftm.dsl.parser import gate_validator
    from ftm.harness import Harness, MockBank
    from ftm.privacy.accountant import PrivacyAccountant
    from ftm.types import AlertContext, Limits

    class Spy(MockBank):
        calls = 0

        def respond(self, *a, **k):
            Spy.calls += 1
            return super().respond(*a, **k)

    acc = PrivacyAccountant(eps_per_alert=2.0)
    gate = Gate.from_config(validator=gate_validator, accountant=acc)
    tools = Harness({b: Spy(b, 0, p_decline=0.0) for b in (2, 3)}, gate, acc).toolbox()
    ctx = AlertContext("A1", "1", datetime(2022, 9, 5).isoformat(), "val", 0.1)
    tools.begin(ctx, Limits())
    ok = "n_counterparties(ALERT.receiver, dir=in, window=before(72))"
    for bad in [(2, "nope(ALERT.receiver)", 1.0), (1, ok, 1.0), (9, ok, 1.0), (2, ok, 0.7),
                (2, "n_tx(ALERT.receiver, dir=in, window=before(72), account=A1b2c3d4)", 1.0)]:  # fmt: skip
        with pytest.raises(GateBlocked):
            tools.ask_bank(*bad)
    assert Spy.calls == 0
    rng = random.Random(0)
    for _ in range(200):
        try:
            tools.ask_bank(rng.choice([2, 3]), ok, rng.choice([0.25, 0.5, 1.0, 2.0]))
        except (GateBlocked, BudgetRefused):
            pass
    assert tools.eps_spent <= 2.0 + 1e-9 and acc.alert_spent["A1"] <= 2.0 + 1e-9
