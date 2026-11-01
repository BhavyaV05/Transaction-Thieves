"""Rich rendering of a `Demo`: the alert panel, the step-by-step investigation, the decision, and the side-by-side comparison.

Owner: T3 (Akshat).

`export_html` renders the same walkthrough into one self-contained static HTML page (inline CSS, no external assets)."""

from __future__ import annotations

import io
import time
from collections.abc import Callable

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ftm.demo.model import AlertView, ArmRun, Demo, DemoStep, lr_effect, zone

ZONE_STYLE = {"auto-clear": "green", "human review": "yellow", "escalate": "bold red"}
NR = Text("not recorded", style="dim italic")


def bar(p: float | None, width: int = 30, threshold: float | None = None) -> Text:
    """0..1 bar; '|' marks the threshold."""
    if p is None:
        return Text("n/a", style="dim")
    n = int(round(max(0.0, min(1.0, p)) * width))
    chars = ["#"] * n + ["."] * (width - n)
    if threshold is not None:
        t = min(width - 1, int(threshold * width))
        chars[t] = "|"
    t = Text("[" + "".join(chars) + "]")
    t.stylize("cyan", 1, 1 + n)
    t.stylize("dim", 1 + n, 1 + width)
    return t


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def pct(p: float | None) -> str:
    return "n/a" if p is None else f"{p:.3f}"


def _row(grid: Table, key: str, val) -> None:
    grid.add_row(Text(key, style="bold"), val if isinstance(val, (Text, Group, Table)) else Text(str(val)))


def alert_panel(a: AlertView, threshold: float | None = None, evidence_rows: int = 12) -> Panel:
    s = a.summary
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    amount = f"{s['amount']:,.2f} {s.get('currency', '')}".strip() if s.get("amount") is not None else None
    _row(grid, "Alert", f"{a.alert_id} ({a.split})   time {s.get('time', '?')}")
    _row(grid, "Transfer", f"{amount or '?'} via {s.get('format', '?')}, direction {s.get('direction', '?')} for the home bank")
    _row(grid, "Parties", f"{s.get('sender', '?')} (bank {s.get('sender_bank', '?')})  ->  {s.get('receiver', '?')} (bank {s.get('receiver_bank', '?')})")
    _row(grid, "Home bank", f"{a.home_bank}   (the only bank whose records we may read in full; accounts are aliased)")
    shape = s.get("shape") or {}
    if shape:
        _row(grid, "Shape", ", ".join(f"{k}={v}" for k, v in shape.items() if k not in ("receiver", "typology")))
    row = Text()
    row.append(f"{a.p0:.3f}  ")
    row.append_text(bar(a.p0, 30, threshold))
    row.append("   the home bank's own detector (one-bank view, calibrated)")
    _row(grid, "p0", row)
    parts = [grid, Text("")]
    ev_name = a.evidence_view or "week-2 stub (top-K by amount)"
    parts.append(Text(f"Evidence trail the planner sees (home bank only, aliased, time-ordered; view: {ev_name})", style="bold"))
    lines = [ln for ln in (a.evidence or "").splitlines() if ln.strip()]
    if not lines:
        parts.append(NR)
    else:
        hidden = max(0, len(lines) - evidence_rows) if evidence_rows else 0
        if hidden:
            parts.append(Text(f"... {hidden} earlier rows not shown ({len(lines)} in the trail)", style="dim"))
        for ln in lines[hidden:]:
            parts.append(Text(ln, style="bold yellow" if "ALERT" in ln.upper() else None, no_wrap=False, overflow="fold"))
    for n in a.notes:
        parts.append(Text(n, style="dim italic"))
    return Panel(Group(*parts), title="(a) The alert as the home bank sees it", border_style="cyan", box=box.ROUNDED)


def step_panel(st: DemoStep, arm: ArmRun, threshold: float | None, width: int = 24) -> Panel:
    title = f"Step {st.n}"
    if st.kind == "stop":
        body = Group(*_thought(st), Text(f"The agent stops here. Posterior stays {pct(st.post_after)}."))
        return Panel(body, title=f"{title} | stop", border_style="blue", box=box.ROUNDED)
    if st.kind == "rejected":
        g = Table.grid(padding=(0, 2))
        g.add_column(no_wrap=True)
        g.add_column(overflow="fold")
        _row(g, "Thought", st.thought or NR)
        _row(g, "Proposed", st.program or "(nothing usable)")
        _row(g, "Rejected", Text(st.note or "?", style="red"))
        return Panel(g, title=f"{title} | proposal not sent (no eps spent, no bank asked)", border_style="red", box=box.ROUNDED)
    g = Table.grid(padding=(0, 2))
    g.add_column(no_wrap=True)
    g.add_column(overflow="fold")
    _row(g, "Thought", st.thought or NR)
    if st.action:
        _row(g, "Action", f"{st.action}")
    _row(g, "Question", Text(st.program or "?", style="bold"))
    _row(g, "In English", st.gloss or NR)
    route = Text(f"bank {st.bank}")
    route.append("  " + (st.bank_why or "(routing reason not available without the alert data)"), style="dim")
    _row(g, "Routed to", route)
    gate = Text()
    colour = {"allowed": "green", "blocked": "red", "over_budget": "red"}.get(st.gate or "", "white")
    gate.append((st.gate or "?").upper().replace("_", " "), style=f"bold {colour}")
    if st.rule_id:
        gate.append(f"  rule {st.rule_id}", style="bold")
        if st.rule_text:
            gate.append(f': "{_clip(st.rule_text, 150)}"', style="dim")
        if st.rule_derived:
            gate.append("  (re-derived from the gate config; the run did not log it)", style="dim italic")
    elif st.gate == "allowed":
        gate.append("  cited rule: ", style="dim")
        gate.append_text(NR)
    if st.gate_reason:
        gate.append(f"  why: {st.gate_reason}", style="red")
    _row(g, "Compliance gate", gate)
    eps = Text()
    if st.eps is not None:
        eps.append(f"asked at eps {st.eps:g}; spent {st.eps_spent:g}" if st.eps_spent is not None else f"asked at eps {st.eps:g}")
    if st.eps_left is not None and arm.eps_budget:
        eps.append(f"; left {st.eps_left:g} of {arm.eps_budget:g}  ")
        eps.append_text(bar(max(st.eps_left, 0) / arm.eps_budget, 12))
    _row(g, "Privacy accountant", eps if eps.plain else NR)
    ans = Text()
    if st.outcome == "reply":
        ans.append(f"bank {st.bank} answered ", style=None)
        ans.append(f"{st.answer_value!r}", style="bold")
        ans.append(f" ({st.answer_kind or 'value'}, noisy" + (f", {st.latency_days:.1f} days" if st.latency_days is not None else "") + ")")
    elif st.outcome in ("decline", "timeout"):
        ans.append(f"bank {st.bank} {'declined to answer' if st.outcome == 'decline' else 'did not answer in time'}", style="yellow")
    elif st.outcome in ("blocked", "over_budget"):
        ans.append("no question went out", style="red")
    else:
        ans.append_text(NR)
    _row(g, "Bank's answer", ans)
    lr = Text()
    if st.outcome == "reply":
        if st.lr is not None:
            lr.append((f"bin {st.bin_label}  " if st.bin_label else "") + f"LR x{st.lr:.2f}  ({lr_effect(st.lr)})")
            if st.used is False:
                lr.append("  (not used by fusion)", style="dim")
        else:
            lr.append("no accepted back-tested LR for this question: it does not move the posterior", style="dim")
    else:
        lr.append("-", style="dim")
    _row(g, "Likelihood ratio", lr)
    post = Text()
    post.append(f"{pct(st.post_before)} -> ", style=None)
    post.append(pct(st.post_after), style="bold")
    post.append("  ")
    post.append_text(bar(st.post_after, width, threshold))
    _row(g, "Posterior", post)
    border = "green" if st.gate == "allowed" else "red"
    return Panel(g, title=f"{title} | ask", border_style=border, box=box.ROUNDED)


def _thought(st: DemoStep) -> list:
    return [Text.assemble(("Thought  ", "bold"), st.thought or NR)]


def agreement(p: float, label: int, demo: Demo) -> tuple[str, str, str]:
    """(sentence, colour, short) comparing the three-way decision with the true label."""
    z = zone(p, demo.threshold, demo.clear_below)
    if z == "human review":
        return "sent to a human reviewer (neither wrong nor settled)", "yellow", "review"
    ok = (z == "escalate") == bool(label)
    return ("decision agrees" if ok else "decision disagrees"), ("green" if ok else "red"), ("yes" if ok else "no")


def decision_panel(arm: ArmRun, demo: Demo, reveal: bool, label: int | None, title: str | None = None) -> Panel:
    z = zone(arm.final_p, demo.threshold, demo.clear_below)
    g = Table.grid(padding=(0, 2))
    g.add_column(no_wrap=True)
    g.add_column(overflow="fold")
    _row(g, "Final probability", Text.assemble((pct(arm.final_p), "bold"), "  ", bar(arm.final_p, 30, demo.threshold)))
    _row(g, "Threshold", f"escalate at >= {demo.threshold:g}, auto-clear below {demo.clear_below:g}, human review in between (illustrative)")
    _row(g, "Decision", Text(z.upper(), style=ZONE_STYLE[z]))
    spent = "n/a" if arm.eps_spent is None else f"{arm.eps_spent:g}" + (f" of {arm.eps_budget:g}" if arm.eps_budget else "")
    _row(g, "Cost", f"{arm.n_asked} question{'' if arm.n_asked == 1 else 's'} answered, eps spent {spent}" + (f", {arm.llm_calls} LLM calls" if arm.llm_calls is not None else ""))
    if arm.stop_reason:
        _row(g, "Stopped because", arm.stop_reason)
    if reveal:
        if label is None:
            _row(g, "True label", NR)
        else:
            verdict = agreement(arm.final_p, label, demo)
            _row(g, "True label", Text.assemble(("laundering" if label else "not laundering", "bold"), "   ",
                                                (verdict[0], verdict[1])))
    return Panel(g, title=title or "(c) The decision", border_style=ZONE_STYLE[z].split()[-1], box=box.ROUNDED)


def _arm_title(arm: ArmRun) -> str:
    return f"{arm.label}" + (" [scripted mock LLM]" if arm.mock else f" [{arm.model}]" if arm.model else "")


def comparison_table(demo: Demo, reveal: bool) -> Table:
    t = Table(title="Comparison on this alert", box=box.SIMPLE_HEAVY)
    for c in ("system", "questions", "eps spent", "p0", "final p", "decision") + (("true label", "correct") if reveal else ()):
        t.add_column(c, justify="left" if c in ("system", "decision") else "right")
    for arm in demo.arms:
        z = zone(arm.final_p, demo.threshold, demo.clear_below)
        row = [_arm_title(arm), str(arm.n_asked), "-" if arm.eps_spent is None else f"{arm.eps_spent:g}", pct(arm.p0), pct(arm.final_p),
               Text(z, style=ZONE_STYLE[z])]
        if reveal:
            lab = demo.alert.label
            row += ["?" if lab is None else ("laundering" if lab else "clean"),
                    "?" if lab is None else agreement(arm.final_p, lab, demo)[2]]
        t.add_row(*row)
    return t


def render(demo: Demo, console: Console, *, pause: float = 0.0, reveal_label: bool = False, evidence_rows: int = 12,
           sleep: Callable[[float], None] = time.sleep) -> None:
    """Print the whole walkthrough. `pause` seconds between steps (live / replay on a terminal); 0 = all at once."""
    wait = (lambda: sleep(pause)) if pause > 0 else (lambda: None)
    console.print(Text("Privacy-preserving cross-bank AML alert triage: one alert, end to end", style="bold underline"))
    console.print(alert_panel(demo.alert, demo.threshold, evidence_rows))
    wait()
    investigated = [a for a in demo.arms if a.name != "p0_only"]
    if len(investigated) <= 1:
        for arm in demo.arms:
            if arm.name == "p0_only":
                continue
            console.print(Text(f"\n(b) The investigation: {_arm_title(arm)}", style="bold"))
            for st in arm.steps:
                console.print(step_panel(st, arm, demo.threshold))
                wait()
            for n in arm.notes:
                console.print(Text(n, style="dim italic"))
            console.print(decision_panel(arm, demo, reveal_label, demo.alert.label))
        base = next((a for a in demo.arms if a.name == "p0_only"), None)
        if base is not None:
            console.print(decision_panel(base, demo, reveal_label, demo.alert.label, title="For reference: p0 only (no questions)"))
    else:
        console.print(Text("\n(b) The investigations, side by side", style="bold"))
        depth = max(len(a.steps) for a in investigated)
        for i in range(depth):  # one table row per step index, so the arms stay aligned and each row can be paused on
            cols = Table(box=None, expand=True, show_header=(i == 0), padding=(0, 1))
            for arm in investigated:
                cols.add_column(_arm_title(arm), ratio=1, overflow="fold")
            cols.add_row(*[step_panel(a.steps[i], a, demo.threshold) if i < len(a.steps) else Text("") for a in investigated])
            console.print(cols)
            wait()
        for arm in investigated:
            for n in arm.notes:
                console.print(Text(f"{arm.label}: {n}", style="dim italic"))
        console.print(Text("\n(c) Decisions", style="bold"))
        console.print(comparison_table(demo, reveal_label))
        for arm in demo.arms:
            if arm.name != "p0_only":
                console.print(decision_panel(arm, demo, reveal_label, demo.alert.label, title=f"(c) The decision: {_arm_title(arm)}"))
    if len(investigated) <= 1 and len(demo.arms) > 1:
        console.print(comparison_table(demo, reveal_label))


HTML_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
{stylesheet}
body {{ background: {bg}; color: {fg}; margin: 0; padding: 24px; }}
pre {{ font-family: Menlo, Consolas, "DejaVu Sans Mono", monospace; font-size: 13px; line-height: 1.35; white-space: pre; }}
code {{ font-family: inherit; }}
.wrap {{ max-width: 100%; overflow-x: auto; }}
</style></head>
<body><div class="wrap"><pre><code>{code}</code></pre></div></body></html>
"""


def export_html(demo: Demo, path: str, *, reveal_label: bool = False, width: int = 120, evidence_rows: int = 12) -> str:
    """Write the walkthrough as ONE self-contained page (inline styles, no external assets). Returns the HTML."""
    console = Console(file=io.StringIO(), record=True, width=width, force_terminal=True, color_system="truecolor", legacy_windows=False)
    render(demo, console, pause=0.0, reveal_label=reveal_label, evidence_rows=evidence_rows)
    from rich.terminal_theme import MONOKAI

    html = console.export_html(theme=MONOKAI, code_format="{code}", inline_styles=True)
    page = HTML_PAGE.format(title=f"Demo: alert {demo.alert.alert_id}", stylesheet="", bg=MONOKAI.background_color.hex,
                            fg=MONOKAI.foreground_color.hex, code=html)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return page
