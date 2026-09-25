from pathlib import Path


def test_identity_is_random_session_only_and_ignores_incoming_uid(monkeypatch):
    import streamlit as st

    from utils.identity import get_browser_uid

    session = {"_browser_uid": "legacy-bearer-token"}
    query = {"uid": "attacker-controlled-bearer"}
    monkeypatch.setattr(st, "session_state", session)
    monkeypatch.setattr(st, "query_params", query)

    first = get_browser_uid()
    assert first not in {"legacy-bearer-token", "attacker-controlled-bearer"}
    assert "_browser_uid" not in session
    assert "uid" not in query
    assert get_browser_uid() == first

    monkeypatch.setattr(st, "session_state", {})
    second = get_browser_uid()
    assert second != first


def test_app_navigation_does_not_leak_identity_in_urls():
    files = [Path("Main_Page.py"), *Path("pages").glob("*.py")]
    for path in files:
        source = path.read_text()
        assert "uid=" not in source
        assert 'query_params["uid"]' not in source
    assert "portfolio_tickers=" not in Path("Main_Page.py").read_text()
