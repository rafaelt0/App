from utils import db


def test_cache_clear_prefix_keeps_unrelated_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", str(tmp_path / "cache.sqlite"))

    db.cache_set("fund_PETR4", {"price": 38})
    db.cache_set("fund_VALE3", {"price": 62})
    db.cache_set("other_setting", {"enabled": True})

    removed = db.cache_clear_prefix("fund_")

    assert removed == 2
    assert db.cache_get("fund_PETR4") is None
    assert db.cache_get("fund_VALE3") is None
    assert db.cache_get("other_setting") == {"enabled": True}


def test_cache_clear_prefix_rejects_empty_prefix():
    try:
        db.cache_clear_prefix("")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty cache prefix should raise ValueError")


def test_cache_clear_prefix_treats_like_wildcards_literally(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", str(tmp_path / "cache.sqlite"))

    db.cache_set("fund_%_special", {"value": 1})
    db.cache_set("fund_other", {"value": 2})
    db.cache_clear_prefix("fund_%")

    assert db.cache_get("fund_%_special") is None
    assert db.cache_get("fund_other") == {"value": 2}
