import datetime as dt
import requests

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import news
from backend.state import PortfolioSnapshot, SessionState, replace_selection


@pytest.fixture
def context():
    tickers = ["PETR4", "VALE3", "WEGE3"]
    portfolio = PortfolioSnapshot("portfolio-one", 0, tickers, {"PETR4": .2, "VALE3": .3, "WEGE3": .5},
                                  "manual", pd.DataFrame(), pd.DataFrame(), 365, {})
    return SessionState("owner-one", selection=tickers, portfolio=portfolio)


@pytest.fixture
def client(context):
    app = FastAPI()
    @app.middleware("http")
    async def owner(request, call_next):
        request.state.session = context
        return await call_next(request)
    app.include_router(news.router)
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def isolated_caches(monkeypatch):
    caches = (news._fetch_brazilian_news, news.get_brazilian_news, news._cached_article_text, news._cached_sentiment)
    for cached in caches:
        cached.clear()
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    yield
    for cached in caches:
        cached.clear()


def story(title, url, age=0, summary=""):
    published = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=age)
    return {"title": title, "link": url, "date": published.strftime("%d/%m/%Y %H:%M UTC"),
            "published": published, "provider": "Publisher", "summary": summary}


def test_shared_story_classified_once_and_weighted_per_asset(client, monkeypatch):
    shared = story("lucro lucro", "https://publisher.example/shared?utm_source=one")
    duplicate = {**shared, "link": "HTTPS://PUBLISHER.EXAMPLE/shared#top"}
    negative = story("prejuizo prejuizo", "https://publisher.example/negative", age=1)
    feeds = {"PETR4": [shared, negative], "VALE3": [duplicate], "WEGE3": []}
    monkeypatch.setattr(news, "get_brazilian_news", lambda ticker: {"ok": True, "items": feeds[ticker]})
    extracted = []
    def extract(url):
        extracted.append(url)
        return ""
    monkeypatch.setattr(news, "_cached_article_text", extract)
    response = client.post("/api/news/analyze", json={"revision": 0})
    assert response.status_code == 200
    data = response.json()
    assert len(extracted) == 2  # Shared URL is not extracted independently for each ticker.
    assert data["articles"][0]["tickers"] == ["PETR4", "VALE3"]
    assert data["summary"]["observation_count"] == 3
    assert data["summary"]["counts"] == {"positive": 1, "neutral": 0, "negative": 1}
    # PETR4's mean is zero; VALE3's mean is +1. Story count must not weight PETR4 twice.
    assert data["summary"]["raw_score"] == pytest.approx(.6)
    assert data["summary"]["coverage"] == pytest.approx(.5)
    assert data["summary"]["score100"] == 80
    rows = data["summary"]["tone"]
    assert rows[0]["mean"] == 0
    assert rows[1]["contribution"] == pytest.approx(30)
    assert rows[2]["mean"] is None and rows[2]["contribution"] is None
    assert sum(row["contribution"] or 0 for row in rows) == pytest.approx(30)


def test_feed_failure_empty_and_no_model_work_are_distinct(client, monkeypatch):
    monkeypatch.setattr(news, "get_brazilian_news", lambda ticker: {"ok": ticker != "PETR4", "items": []})
    def forbidden(*args):
        pytest.fail("Empty sample must not initialize model or extract publisher content")
    monkeypatch.setattr(news.requests, "post", forbidden)
    monkeypatch.setattr(news, "_cached_article_text", forbidden)
    data = client.post("/api/news/analyze", json={"revision": 0}).json()
    assert data["summary"]["score100"] is None
    assert data["summary"]["coverage"] == 0
    assert data["summary"]["tone"][0]["state"] == "Feed indisponível"
    assert data["summary"]["tone"][1]["state"] == "Sem notícias nos últimos 7 dias"
    assert data["model"]["engines_used"] == []


def test_news_is_post_only_and_stale_context_never_fetches(client, context, monkeypatch):
    def forbidden(*args):
        pytest.fail("GET or stale/missing portfolio must not run providers")
    monkeypatch.setattr(news, "get_brazilian_news", forbidden)
    assert client.get("/api/news/analyze").status_code == 405
    assert client.get("/api/news/refresh").status_code == 405
    assert client.post("/api/news/analyze", json={"revision": 0, "portfolio_id": "other"}).status_code == 409
    replace_selection(context, ["PETR4"], 0)
    assert client.post("/api/news/analyze", json={"revision": 0}).status_code == 409


def test_selection_change_during_fetch_rejects_publication(client, context, monkeypatch):
    def feed(ticker):
        replace_selection(context, ["PETR4"], context.revision)
        return {"ok": True, "items": []}
    monkeypatch.setattr(news, "get_brazilian_news", feed)
    response = client.post("/api/news/analyze", json={"revision": 0})
    assert response.status_code == 409
    assert "news" not in context.artifacts


def test_cached_rss_rolling_cutoff_and_per_asset_ten_before_shared_merge(context, monkeypatch):
    entries = [story("lucro lucro", f"https://publisher.example/{index}", age=index) for index in range(12)]
    entries += [story("prejuizo prejuizo", "https://publisher.example/expired", age=169)]
    context.portfolio.tickers = ["PETR4"]
    context.portfolio.weights = {"PETR4": 1}
    monkeypatch.setattr(news, "get_brazilian_news", lambda ticker: {"ok": True, "items": entries})
    monkeypatch.setattr(news, "_cached_article_text", lambda url: "")
    data = news.analyze_portfolio(context, context.portfolio)
    assert {article["link"] for article in data["articles"]} == {f"https://publisher.example/{index}" for index in range(10)}
    assert data["summary"]["counts"] == {"positive": 10, "neutral": 0, "negative": 0}


def test_zero_weight_stories_do_not_fabricate_neutral_portfolio_note(context, monkeypatch):
    context.portfolio.weights = {"PETR4": 0, "VALE3": .5, "WEGE3": .5}
    monkeypatch.setattr(news, "get_brazilian_news", lambda ticker: {"ok": True, "items": [story("lucro", "https://publisher.example/one")] if ticker == "PETR4" else []})
    monkeypatch.setattr(news, "_cached_article_text", lambda url: "")
    data = news.analyze_portfolio(context, context.portfolio)
    assert data["summary"]["counts"]["positive"] == 1
    assert data["summary"]["raw_score"] is None and data["summary"]["score100"] is None
    assert data["articles"][0]["intensity"] == "Evidência limitada"
    assert data["articles"][0]["evidence_count"] == 1
    assert data["articles"][0]["scores"] == []


def jev_response(positive=.399, negative=.2, neutral=.401):
    response = requests.Response()
    response.status_code = 200
    import json
    response._content = json.dumps({"answers": {"sentiment": {
        "type": "choice", "probabilities": {
            "POSITIVE": positive, "NEGATIVE": negative, "NEUTRAL": neutral,
        },
    }}}).encode()
    return response


def test_jev_request_threshold_before_rounding_and_cache(context, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "server-secret")
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs)
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert kwargs["headers"] == {"Authorization": "Bearer server-secret"}
        assert kwargs["json"]["model"] == "jev-latest"
        assert kwargs["json"]["state"] == {"article": "Lucro cresce Resumo"}
        assert kwargs["json"]["questions"]["sentiment"]["type"] == "choice"
        assert kwargs["timeout"] == (5, 20)
        assert kwargs["allow_redirects"] is False
        return jev_response()
    monkeypatch.setattr(news.requests, "post", post)
    result = news.analyze_sentiment("Lucro cresce", "Resumo", context)
    assert result["is_jev"] is True
    assert result["raw_text_length"] == 3
    assert result["score"] == .20
    assert result["sentiment"] == "Neutro"
    assert sum(row["score"] for row in result["scores"]) == pytest.approx(1)
    assert news.analyze_sentiment("Lucro cresce", "Resumo", context) == result
    assert len(calls) == 1


@pytest.mark.parametrize("status", [401, 429, 500, 529, None])
def test_api_failure_backoff_and_recovery(context, monkeypatch, status):
    monkeypatch.setenv("TYPESAFE_API_KEY", "server-secret")
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            return jev_response(.8, .1, .1)
        if status is None:
            raise requests.Timeout("unavailable")
        response = requests.Response()
        response.status_code = status
        return response
    monkeypatch.setattr(news.requests, "post", post)
    result = news.analyze_sentiment("não lucro. lucro lucro", "", context)
    assert result["is_jev"] is False
    assert result["engine"] == "PLN Léxico (fallback)"
    assert result["pos_terms"] == ["lucro", "lucro"]
    assert len(result["neg_terms"]) == 1
    assert result["score"] == .33
    assert result["evidence_count"] == 3
    assert result["scores"] == []
    assert news.analyze_sentiment("não lucro. lucro lucro", "", context) == result
    assert len(calls) == 1
    context.model_retry_at = 0
    assert news.analyze_sentiment("não lucro. lucro lucro", "", context)["is_jev"] is True
    assert len(calls) == 2


@pytest.mark.parametrize("probabilities", [(.8, .8, .3), (-.1, .5, .6), (float("nan"), .1, .9), (True, 0, 0)])
def test_invalid_jev_probabilities_use_identified_fallback(context, monkeypatch, probabilities):
    monkeypatch.setenv("TYPESAFE_API_KEY", "server-secret")
    monkeypatch.setattr(news.requests, "post", lambda *a, **kw: jev_response(*probabilities))
    result = news.analyze_sentiment("lucro", "", context)
    assert result["engine"] == "PLN Léxico (fallback)"
    assert result["scores"] == []


def test_missing_key_never_calls_api(context, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing credentials must not call Jev")
    monkeypatch.setattr(news.requests, "post", forbidden)
    assert news.analyze_sentiment("lucro", "", context)["engine"] == "PLN Léxico (fallback)"


def test_real_feed_wrapper_distinguishes_parse_failure_and_empty(monkeypatch):
    payloads = [b"<invalid", b"<rss><channel/></rss>", b"x" * (2 * 1024 * 1024 + 1)]
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return payloads.pop(0)[:size]
    def urlopen(request, timeout):
        assert timeout == 8
        return Response()
    monkeypatch.setattr(news.urllib.request, "urlopen", urlopen)
    assert news.get_brazilian_news("PETR4") == {"ok": False, "items": []}
    assert news.get_brazilian_news("VALE3") == {"ok": True, "items": []}
    assert news.get_brazilian_news("WEGE3") == {"ok": False, "items": []}


def test_rss_vs_extracted_body_provenance_and_importance_remain_independent(context, monkeypatch):
    entry = story("Empresa patrocina esporte", "https://publisher.example/one", summary="Evento de marketing")
    context.portfolio.tickers = ["PETR4"]
    context.portfolio.weights = {"PETR4": 1}
    monkeypatch.setattr(news, "get_brazilian_news", lambda ticker: {"ok": True, "items": [entry]})
    monkeypatch.setattr(news, "_cached_article_text", lambda url: "lucro lucro")
    article = news.analyze_portfolio(context, context.portfolio)["articles"][0]
    assert article["importance_rank"] == 1
    assert article["sentiment"] == "Otimista"
    assert article["summary"] == "Evento de marketing"
    assert article["text_source"] == "Texto do artigo"


def test_refresh_refetches_rss_and_text_and_retries_failed_inference(client, context, monkeypatch):
    context.selection = ["PETR4"]
    context.portfolio.tickers = ["PETR4"]
    context.portfolio.weights = {"PETR4": 1}
    entry = story("lucro lucro", "https://publisher.example/one")
    news_items = [entry]
    fetched = []
    extracted = []
    # Exercise the real decorated feed caches, with a deterministic external RSS transport.
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            import email.utils
            published = email.utils.format_datetime(entry["published"])
            xml = "".join(f'<item><title>{item["title"]}</title><link>{item["link"]}</link><pubDate>{published}</pubDate></item>' for item in news_items)
            return f"<rss><channel>{xml}</channel></rss>".encode()
    def urlopen(request, timeout):
        fetched.append(request.full_url)
        return Response()
    def extract(url):
        extracted.append(url)
        return ""
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs["json"]["state"]["article"])
        if len(calls) == 1:
            raise requests.Timeout("transient failure")
        return jev_response(.1, .8, .1)
    monkeypatch.setenv("TYPESAFE_API_KEY", "server-secret")
    monkeypatch.setattr(news.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(news, "extract_article_text", extract)
    monkeypatch.setattr(news.requests, "post", post)
    initial = client.post("/api/news/analyze", json={"revision": 0}).json()
    assert initial["articles"][0]["is_jev"] is False
    news_items.append(story("prejuizo prejuizo", "https://publisher.example/two"))
    context.model_retry_at = 12345
    refreshed = client.post("/api/news/refresh", json={"revision": 0}).json()
    assert len(fetched) == 2
    assert extracted.count("https://publisher.example/one") == 2
    assert len(calls) == 3  # Failed judgments are retried after refresh.
    engines = {article["title"]: article["engine"] for article in refreshed["articles"]}
    assert engines == {"lucro lucro": "Jev", "prejuizo prejuizo": "Jev"}
    assert refreshed["summary"]["counts"] == {"positive": 0, "neutral": 0, "negative": 2}
    assert refreshed["model"]["engines_used"] == ["Jev"]
    assert context.model_retry_at == 0


def test_single_term_lexical_items_are_shown_but_excluded_from_the_note(client, monkeypatch):
    feeds = {
        "PETR4": [story("prejuizo", "https://publisher.example/weak")],
        "VALE3": [story("lucro lucro", "https://publisher.example/strong")],
        "WEGE3": [],
    }
    monkeypatch.setattr(news, "get_brazilian_news", lambda ticker: {"ok": True, "items": feeds[ticker]})
    monkeypatch.setattr(news, "_cached_article_text", lambda url: "")
    data = client.post("/api/news/analyze", json={"revision": 0}).json()
    weak = next(article for article in data["articles"] if article["title"] == "prejuizo")
    assert weak["intensity"] == "Evidência limitada"
    assert weak["sentiment"] == "Pessimista"
    summary = data["summary"]
    # The lone weak negative word would have pulled the note to 60 at PETR4's .2 weight.
    assert summary["counts"] == {"positive": 1, "neutral": 0, "negative": 1}
    assert summary["coverage"] == pytest.approx(.3)
    assert summary["raw_score"] == pytest.approx(1)
    assert summary["score100"] == 100
    assert summary["observation_count"] == 2  # Every article-ticker pair is displayed, scored or not.
    assert summary["tone"][0]["state"] == "Evidência limitada"
    assert summary["tone"][0]["contribution"] is None


def test_only_limited_evidence_reports_no_note_instead_of_neutral(client, monkeypatch):
    monkeypatch.setattr(news, "get_brazilian_news",
                        lambda ticker: {"ok": True, "items": [story("lucro", f"https://publisher.example/{ticker}")]})
    monkeypatch.setattr(news, "_cached_article_text", lambda url: "")
    summary = client.post("/api/news/analyze", json={"revision": 0}).json()["summary"]
    assert summary["score100"] is None
    assert summary["raw_score"] is None
    assert summary["coverage"] == 0
    assert summary["label"] == "EVIDÊNCIA LIMITADA"
    assert summary["counts"]["positive"] == 3
