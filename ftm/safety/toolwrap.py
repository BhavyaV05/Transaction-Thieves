"""A ToolBox wrapper base for the safety experiments (E15): every tool call passes `_check`, every bank reply `_reply`.

Owner: T4 (Shreyas).  Not part of the frozen harness contract: a wrapper is a ToolBox that holds another ToolBox, so it
can sit between a system and the harness without changing either. `__getattr__` forwards everything else (begin, end,
wrap, the session facts), so a runner that holds the session itself still works.
"""

from __future__ import annotations

from ftm.types import Reply


class ToolWrap:
    """Override `_check(tool, args)` (raise to refuse the call) and `_reply(reply)` (return the reply the system sees).
    `ask_banks` (the wave extension) exists on the wrapper only if it exists on the inner toolbox, so a system that
    probes for it sees the same capabilities with or without the wrapper."""

    def __init__(self, inner):
        self._inner = inner
        if hasattr(inner, "ask_banks"):
            self.ask_banks = self._ask_banks

    def __getattr__(self, name):
        if name.startswith("__") or name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    # hooks ---------------------------------------------------------------------------------------------------------
    def _check(self, tool: str, args: dict) -> None:
        return None

    def _reply(self, reply):
        return reply

    # the tools -----------------------------------------------------------------------------------------------------
    def ask_bank(self, bank, program, eps, *args, **kw):
        self._check("ask_bank", {"bank": bank, "program": program, "eps": eps})
        r = self._inner.ask_bank(bank, program, eps, *args, **kw)
        return self._reply(r) if isinstance(r, Reply) else r

    def _ask_banks(self, requests: list[tuple]) -> list:
        """A wave: each request is checked on its own; a refused one comes back as its exception, as the harness does."""
        out: list = [None] * len(requests)
        allowed: list[tuple[int, tuple]] = []
        for i, req in enumerate(requests):
            try:
                self._check("ask_bank", {"bank": req[0], "program": req[1], "eps": req[2]})
            except PermissionError as exc:
                out[i] = exc
            else:
                allowed.append((i, req))
        if allowed:
            answers = self._inner.ask_banks([r for _, r in allowed])
            for (i, _), r in zip(allowed, answers, strict=True):
                out[i] = self._reply(r) if isinstance(r, Reply) else r
        return out

    def local_graph(self, query, as_of):
        self._check("local_graph", {"query": query, "as_of": as_of})
        return self._inner.local_graph(query, as_of)

    def lookup_rules(self, query):
        self._check("lookup_rules", {"query": query})
        return self._inner.lookup_rules(query)

    def search_library(self, shape):
        self._check("search_library", {"shape": shape})
        return self._inner.search_library(shape)
