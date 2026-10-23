"""T4: injection test set generator (hand-off #14)."""

from collections import Counter

from ftm.safety import injection as inj


def test_counts_labels_and_goals():
    items = inj.generate()
    c = Counter(it.kind for it in items)
    assert c == {"direct": 100, "authority": 100, "smuggling": 100, "exfil": 100, "benign": 400}
    for it in items:
        assert it.injection_present == (it.kind != "benign")
        assert it.goal == inj.GOALS.get(it.kind)
        assert 0 < len(it.note) <= inj.NOTE_MAX
    assert len({it.item_id for it in items}) == len(items)


def test_seeded_and_reproducible():
    a = [it.as_dict() for it in inj.generate(seed=3)]
    b = [it.as_dict() for it in inj.generate(seed=3)]
    c = [it.as_dict() for it in inj.generate(seed=4)]
    assert a == b and a != c


def test_attack_text_is_planted_whole():
    for it in inj.generate():
        if it.kind == "smuggling":
            assert inj.SMUGGLING[it.template] in it.note  # smuggled syntax kept exact
        elif it.kind != "benign":
            assert inj.ATTACKS[it.kind][it.template].lower().rstrip(".") in it.note.lower()


def test_every_template_and_placement_used():
    items = inj.generate()
    for kind in inj.TYPES:
        mine = [it for it in items if it.kind == kind]
        assert {it.template for it in mine} == set(range(len(inj.ATTACKS[kind])))
        assert {it.placement for it in mine} == set(inj.PLACEMENTS)


def test_benign_notes_contain_no_attack_text():
    attacks = [t.lower() for pool in inj.ATTACKS.values() for t in pool]
    for it in inj.generate():
        if it.kind == "benign":
            assert not any(a in it.note.lower() for a in attacks)


def test_jsonl_roundtrip(tmp_path):
    items = inj.generate(n_per_type=5, n_benign=5)
    p = inj.write_jsonl(items, tmp_path / "set.jsonl")
    assert inj.read_jsonl(p) == items
