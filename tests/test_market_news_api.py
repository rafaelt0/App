import datetime as dt
import email.utils
import urllib.parse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import news


@pytest.fixture
def client():
    news._market_headlines.clear()
    app = FastAPI()
    app.include_router(news.router)
    with TestClient(app) as client:
        yield client
    news._market_headlines.clear()


def story(title, hours=0, link="https://publisher.example/article"):
    published = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    return {"title": title, "published": published, "link": link, "provider": "Publisher", "summary": ""}


def test_market_headlines_work_without_portfolio_and_rank_recent_items(client, monkeypatch):
    items = [story("Bolsa sobe", 0), story("Banco Central decide juros", 1),
             story("Lucro anunciado", 2), story("Lucro antigo", 49), story("Lucro futuro", -1)]
    monkeypatch.setattr(news, "_fetch_news_query", lambda query: items)
    response = client.post("/api/news/market")
    assert response.status_code == 200
    data = response.json()
    assert data["window_hours"] == 48
    assert [article["title"] for article in data["articles"]] == ["Banco Central decide juros", "Lucro anunciado", "Bolsa sobe"]
    assert [article["importance_label"] for article in data["articles"]] == ["Alta", "Alta", "Média"]


def test_market_feed_parses_and_deduplicates_rss(client, monkeypatch):
    published = email.utils.format_datetime(dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5))
    item = f"<item><title>Lucro cresce - Publisher</title><source>Publisher</source><link>https://publisher.example/story</link><pubDate>{published}</pubDate></item>"
    xml = f"<rss><channel>{item}{item}</channel></rss>".encode()
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            assert size == 2 * 1024 * 1024 + 1
            return xml

    def fetch(request, timeout):
        requests.append(request.full_url)
        assert timeout == 8
        return Response()

    monkeypatch.setattr(news.urllib.request, "urlopen", fetch)
    data = client.post("/api/news/market").json()
    assert len(data["articles"]) == 1
    assert data["articles"][0]["title"] == "Lucro cresce"
    assert data["articles"][0]["provider"] == "Publisher"
    assert data["articles"][0]["link"] == "https://publisher.example/story"
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(requests[0]).query)
    assert "Ibovespa" in query["q"][0] and "when:2d" in query["q"][0]
    assert query["hl"] == ["pt-BR"]


def test_market_feed_is_bounded_cached_and_sanitizes_links(client, monkeypatch):
    calls = []

    def fetch(query):
        calls.append(query)
        return [story(f"Lucro {index}", index, "javascript:alert(1)") for index in range(20)]

    monkeypatch.setattr(news, "_fetch_news_query", fetch)
    first = client.post("/api/news/market").json()
    second = client.post("/api/news/market").json()
    assert first == second
    assert len(calls) == 1
    assert len(first["articles"]) == 12
    assert first["articles"][0]["title"] == "Lucro 0"
    assert all(article["link"] is None for article in first["articles"])
    assert client.get("/api/news/market").status_code == 405


def test_market_feed_failure_can_retry_and_differs_from_empty(client, monkeypatch):
    def unavailable(query):
        raise TimeoutError("Source unavailable")

    monkeypatch.setattr(news, "_fetch_news_query", unavailable)
    response = client.post("/api/news/market")
    assert response.status_code == 503
    assert "indisponíveis" in response.json()["detail"]
    monkeypatch.setattr(news, "_fetch_news_query", lambda query: [])
    response = client.post("/api/news/market")
    assert response.status_code == 200
    assert response.json()["articles"] == []
