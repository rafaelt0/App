import asyncio
import threading

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest

from backend import app as api_module, state
from utils import db


HEADERS = {"Origin": "http://localhost:3000", "X-B3Lab-Request": "1"}


@pytest.fixture
def isolated_api(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_DB", str(tmp_path / "sessions.sqlite"))
    monkeypatch.setattr(state, "_states", {})
    monkeypatch.setattr(api_module, "_capacity", asyncio.Semaphore(2))
    app = FastAPI()
    app.middleware("http")(api_module.owned_requests)
    app.add_api_route("/api/session", api_module.read_session, methods=["GET"])
    app.add_api_route("/api/session/selection", api_module.select_stocks, methods=["PUT"])
    app.add_api_route("/api/watchlist", api_module.read_watchlist, methods=["GET"])
    app.add_api_route("/api/watchlist/{ticker}", api_module.add_favorite, methods=["PUT"])
    return app


def test_anonymous_sessions_do_not_accept_url_or_invented_cookie_identity(isolated_api):
    with TestClient(isolated_api) as owner, TestClient(isolated_api) as visitor:
        bootstrap = owner.get("/api/session")
        cookie = bootstrap.headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Max-Age" not in cookie
        assert owner.put("/api/watchlist/WEGE3", headers=HEADERS).json()["watchlist"] == ["WEGE3"]
        visitor.cookies.set("b3lab_session", "attacker-controlled-token")
        assert visitor.get("/api/session?uid=legacy").json()["watchlist"] == []
        assert visitor.get("/api/watchlist").json()["watchlist"] == []
        assert owner.get("/api/watchlist").json()["watchlist"] == ["WEGE3"]


def test_selection_revision_and_csrf_prevent_unrequested_mutations(isolated_api):
    with TestClient(isolated_api) as client:
        client.get("/api/session")
        selected = client.put("/api/session/selection", headers=HEADERS, json={"tickers": ["WEGE3"], "revision": 0})
        assert selected.status_code == 200
        assert selected.json()["revision"] == 1
        assert client.put("/api/session/selection", headers=HEADERS, json={"tickers": ["VALE3"], "revision": 0}).status_code == 409
        for headers in ({"Origin": "https://evil.example", "X-B3Lab-Request": "1"}, {"Origin": "http://localhost:3000"}):
            assert client.put("/api/session/selection", headers=headers, json={"tickers": [], "revision": 1}).status_code == 403
        assert client.get("/api/session").json()["selection"] == ["WEGE3"]


@pytest.mark.parametrize("pressure", ["capacity", "idle"])
def test_inflight_owner_cannot_be_evicted_and_publish_after_selection_changes(isolated_api, monkeypatch, pressure):
    if pressure == "capacity":
        monkeypatch.setattr(state, "_MAX_ACTIVE_SESSIONS", 1)
    entered, finish = threading.Event(), threading.Event()

    @isolated_api.post("/api/held")
    def held(request: Request):
        ctx = state.session(request)
        generation = state.begin(ctx, "home", ctx.revision)
        revision = ctx.revision
        if pressure == "idle":
            ctx.last_seen = 0
        entered.set()
        assert finish.wait(10)
        state.publish(ctx, "home", revision, generation, {"id": "held"})
        return {"published": True}

    with TestClient(isolated_api) as owner, TestClient(isolated_api) as another:
        owner.get("/api/session")
        response = {}
        thread = threading.Thread(target=lambda: response.update(result=owner.post("/api/held", headers=HEADERS)))
        thread.start()
        try:
            assert entered.wait(5)
            assert another.get("/api/session").status_code == (503 if pressure == "capacity" else 200)
            changed = owner.put("/api/session/selection", headers=HEADERS, json={"tickers": ["WEGE3"], "revision": 0})
            assert changed.status_code == 200
        finally:
            finish.set()
            thread.join(10)
        assert not thread.is_alive()
        assert response["result"].status_code == 409
        assert owner.get("/api/session").json()["selection"] == ["WEGE3"]


def test_news_refresh_shares_the_two_analysis_admission_limit(isolated_api):
    entered = threading.Barrier(3)
    finish = threading.Event()

    @isolated_api.post("/api/news/refresh")
    def refresh():
        entered.wait(timeout=10)
        assert finish.wait(10)
        return {"completed": True}

    with TestClient(isolated_api) as client:
        client.get("/api/session")
        responses = []
        workers = [threading.Thread(target=lambda: responses.append(client.post("/api/news/refresh", headers=HEADERS))) for _ in range(2)]
        for worker in workers:
            worker.start()
        try:
            entered.wait(timeout=10)
            assert client.post("/api/news/refresh", headers=HEADERS).status_code == 503
        finally:
            finish.set()
            for worker in workers:
                worker.join(10)
        assert [response.status_code for response in responses] == [200, 200]
