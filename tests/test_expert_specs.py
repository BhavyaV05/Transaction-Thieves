"""T4: expert question specs agree with the red-flag corpus, the gate and the answer kinds."""

from collections import Counter

from ftm.compliance import expert_specs as es
from ftm.compliance import gate as g

KINDS = {"bool", "count", "kary", "set_count"}
TYPOLOGIES = {"fan_in", "fan_out", "cycle", "scatter_gather", "gather_scatter", "bipartite", "stack", "random"}


def test_about_sixty_questions_with_unique_ids_and_families():
    qs = es.load_questions()
    assert 50 <= len(qs) <= 70
    assert len({q.qid for q in qs}) == len(qs)
    fams = es.load_families()
    assert len({f["family"] for f in fams}) == len(fams)


def test_fields_are_valid():
    flags = g.load_red_flags()
    for q in es.load_questions():
        assert q.kind in KINDS, q.qid
        assert set(q.typologies) <= TYPOLOGIES, q.qid
        assert q.red_flags and set(q.red_flags) <= set(flags), q.qid
        assert q.sensitivity >= 0 and q.expect


def test_questions_name_no_account_date_or_amount_literal():
    for q in es.load_questions():
        assert "SUBJ" in q.text or "ALERT." in q.text, q.qid
        assert not g.identifying_literals(q.text), q.qid
        assert "{" not in q.text, q.qid  # every template slot was filled


def test_set_count_questions_use_home_bank_sets():
    for q in es.load_questions():
        if q.kind == "set_count":
            assert "ALERT.payers" in q.text or "ALERT.payees" in q.text, q.qid


def test_coverage_of_typologies_and_observable_red_flags():
    qs = es.load_questions()
    fam_typ = {q.family: q.typologies for q in qs}
    per_typ = Counter(t for ts in fam_typ.values() for t in ts)
    for t in TYPOLOGIES - {"random"}:
        assert per_typ[t] >= 2, t
    cited = {f for q in qs for f in q.red_flags}
    observable = {f.flag_id for f in g.load_red_flags().values() if f.observable == "yes"}
    assert len(cited & observable) >= 15
    assert all(g.load_red_flags()[f].observable != "no" for f in cited)  # no question rests on an unobservable flag
    assert Counter(q.kind for q in qs).keys() == KINDS  # every answer kind is exercised


def test_family_order_is_the_expert_ranking():
    fams = [f["family"] for f in es.load_families()]
    assert fams[:2] == ["pass_through", "payees_hit_home_payers"]  # the proposal's worked example comes first
    assert fams[-1] == "entity_accounts"  # restricted, never asked, last
