"""Alert-aware query for `lookup_rules` (audit finding 15): the planner used one fixed sentence for every alert, so every alert got
the same top-5. `rules_query(alert)` builds the query from what the alert says: the parties' jurisdictions, the payment format and
currency, an amount band and a typology guess when the alert carries one. The words are the ones the rule and red-flag texts use
(jurisdiction, cross-border, structuring, ...), so BM25 and the dense half both have something to match.

Owner: T4. Pure function of the alert summary; no label, no id, no account name reaches the query.
"""

from __future__ import annotations

BASE = "sharing information with another financial institution about suspected money laundering"
FIXED_RULES_QUERY = BASE  # the planner's original one query for every alert (audit finding 15); `--rules fixed`
JURISDICTION_WORDS = {"US": "United States FinCEN 314(b)", "SG": "Singapore COSMIC MAS"}
# the red-flag corpus' own vocabulary for the typologies (configs/compliance/red_flags.yaml)
TYPOLOGY_WORDS = {"fan-in": "many senders to one account, funds passed on quickly", "fan-out": "one account sending to many recipients",
                  "cycle": "funds returned to the originating bank, circular movement", "scatter-gather": "layering through intermediaries",
                  "gather-scatter": "layering through intermediaries", "stack": "layered transfers", "bipartite": "layered transfers between two groups",
                  "random": "unusual transfers", "structuring": "just below reporting thresholds, structuring"}  # fmt: skip
BANDS = ((1_000, "small amount"), (10_000, "amount under ten thousand"), (100_000, "large amount above ten thousand, reporting threshold"))


def amount_band(amount) -> str:
    try:
        a = float(amount)
    except (TypeError, ValueError):
        return ""
    for hi, words in BANDS:
        if a < hi:
            return words
    return "very large amount"


def rules_query(alert, gate=None) -> str:
    """Query text for `tools.lookup_rules` for this alert. `gate` (optional) maps banks to jurisdictions; without it the default
    (US) is assumed for every bank. Always contains the base sentence, so the legal-basis rules keep matching."""
    s = alert.summary
    jur_of = gate.jurisdiction if gate is not None else (lambda b: "US")
    home = jur_of(alert.home_bank)
    partners = sorted({jur_of(b) for b in s.get("partner_banks", [])}) or [home]
    parts = [BASE, JURISDICTION_WORDS.get(home, home)]
    cross = [j for j in partners if j != home]
    if cross:
        parts.append("cross-border request to " + " ".join(JURISDICTION_WORDS.get(j, j) for j in cross))
    for key in ("format", "currency"):
        if s.get(key):
            parts.append(str(s[key]))
    band = amount_band(s.get("amount"))
    if band:
        parts.append(band)
    guess = str(s.get("typology_guess") or s.get("typology") or "").lower()
    if guess:
        parts.append(TYPOLOGY_WORDS.get(guess, guess))
    return " ".join(parts)
