import pytest

from ftm import guards


def test_test_split_locked_by_default(monkeypatch):
    monkeypatch.delenv(guards.FINAL_RUN_ENV, raising=False)
    with pytest.raises(guards.TestSplitLocked):
        guards.require_final_run()


def test_test_split_unlocked_only_with_flag(monkeypatch):
    monkeypatch.setenv(guards.FINAL_RUN_ENV, "1")
    guards.require_final_run()
    monkeypatch.setenv(guards.FINAL_RUN_ENV, "true")  # only the exact value "1" counts
    with pytest.raises(guards.TestSplitLocked):
        guards.require_final_run()


def test_assert_not_future():
    guards.assert_not_future(5, 5)
    guards.assert_not_future(None, 5)
    with pytest.raises(guards.TimeGuardError):
        guards.assert_not_future(6, 5)


def test_enforce_as_of_filters_future_rows():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"timestamp": [1, 2, 3, 4], "x": list("abcd")})
    out = guards.enforce_as_of(df, 3)
    assert out["timestamp"].to_list() == [1, 2, 3]
    with pytest.raises(guards.TimeGuardError):
        guards.enforce_as_of(df, None)
