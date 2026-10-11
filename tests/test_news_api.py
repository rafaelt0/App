import datetime as dt
import sys
from types import SimpleNamespace

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
    monkeypatch.setattr(news, "_model", None)
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
    monkeypatch.setattr(news, "load_finbert_pipeline", lambda ctx: None)
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
    monkeypatch.setattr(news, "load_finbert_pipeline", forbidden)
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
    monkeypatch.setattr(news, "load_finbert_pipeline", lambda ctx: None)
    data = news.analyze_portfolio(context, context.portfolio)
    assert {article["link"] for article in data["articles"]} == {f"https://publisher.example/{index}" for index in range(10)}
    assert data["summary"]["counts"] == {"positive": 10, "neutral": 0, "negative": 0}


def test_zero_weight_stories_do_not_fabricate_neutral_portfolio_note(context, monkeypatch):
    context.portfolio.weights = {"PETR4": 0, "VALE3": .5, "WEGE3": .5}
    monkeypatch.setattr(news, "get_brazilian_news", lambda ticker: {"ok": True, "items": [story("lucro", "https://publisher.example/one")] if ticker == "PETR4" else []})
    monkeypatch.setattr(news, "_cached_article_text", lambda url: "")
    monkeypatch.setattr(news, "load_finbert_pipeline", lambda ctx: None)
    data = news.analyze_portfolio(context, context.portfolio)
    assert data["summary"]["counts"]["positive"] == 1
    assert data["summary"]["raw_score"] is None and data["summary"]["score100"] is None
    assert data["articles"][0]["intensity"] == "Evidência limitada"
    assert data["articles"][0]["evidence_count"] == 1
    assert data["articles"][0]["scores"] == []


class Tokenizer:
    def encode(self, text, add_special_tokens, truncation, max_length):
        assert not add_special_tokens and truncation and max_length == 510
        return text.split()[:max_length]
    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(tokens)


class Classifier:
    tokenizer = Tokenizer()
    def __call__(self, texts):
        assert len(texts[0].split()) == 510
        return [[{"label": "POSITIVE", "score": .399}, {"label": "NEGATIVE", "score": .2}, {"label": "NEUTRAL", "score": .8}]]


def test_finbert_independent_evidence_threshold_before_rounding_and_token_limit():
    result = news.analise_sentimento_finbert("palavra " * 700, "", Classifier())
    assert result["is_finbert"] is True
    assert result["raw_text_length"] == 510
    assert result["score"] == .20
    assert result["sentiment"] == "Neutro"  # .199 is below .20 despite rounded score.
    assert sum(row["score"] for row in result["scores"]) == pytest.approx(1.399)
    assert "evidence_count" not in result


def test_inference_error_keeps_lexical_negation_repetition_and_provenance():
    class FailedClassifier:
        tokenizer = Tokenizer()
        def __call__(self, texts):
            raise RuntimeError("inference unavailable")
    result = news.analise_sentimento_finbert("não lucro. lucro lucro", "", FailedClassifier())
    assert result["is_finbert"] is False
    assert result["engine"] == "PLN Léxico (fallback)"
    assert result["pos_terms"] == ["lucro", "lucro"]
    assert len(result["neg_terms"]) == 1
    assert result["score"] == .33
    assert result["evidence_count"] == 3
    assert result["scores"] == []


def test_failed_model_load_backoff_and_successful_pipeline_reuse(context, monkeypatch):
    attempts = []
    class Unavailable:
        @staticmethod
        def from_pretrained(name):
            attempts.append(name)
            raise OSError("offline model artifacts")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Unavailable, BertForSequenceClassification=Unavailable, pipeline=lambda *args, **kwargs: None))
    assert news.load_finbert_pipeline(context) is None
    assert news.load_finbert_pipeline(context) is None
    assert attempts == [news.MODEL_NAME]
    context.model_retry_at = 0  # Explicit refresh resets the failed-load backoff.
    assert news.load_finbert_pipeline(context) is None
    assert attempts == [news.MODEL_NAME, news.MODEL_NAME]
    loaded = object()
    monkeypatch.setattr(news, "_model", loaded)
    assert news.load_finbert_pipeline(context) is loaded


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
    monkeypatch.setattr(news, "load_finbert_pipeline", lambda ctx: None)
    article = news.analyze_portfolio(context, context.portfolio)["articles"][0]
    assert article["importance_rank"] == 1
    assert article["sentiment"] == "Otimista"
    assert article["summary"] == "Evento de marketing"
    assert article["text_source"] == "Texto do artigo"


def test_refresh_refetches_rss_and_text_but_preserves_cached_inference_fallback(client, context, monkeypatch):
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
    class RecoveringModel:
        tokenizer = Tokenizer()
        calls = 0
        def __call__(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient inference failure")
            return [[{"label": "POSITIVE", "score": .2}, {"label": "NEGATIVE", "score": .8}, {"label": "NEUTRAL", "score": .3}]]
    model = RecoveringModel()
    monkeypatch.setattr(news.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(news, "extract_article_text", extract)
    monkeypatch.setattr(news, "load_finbert_pipeline", lambda ctx: model)
    initial = client.post("/api/news/analyze", json={"revision": 0}).json()
    assert initial["articles"][0]["is_finbert"] is False
    news_items.append(story("prejuizo prejuizo", "https://publisher.example/two"))
    context.model_retry_at = 12345
    refreshed = client.post("/api/news/refresh", json={"revision": 0}).json()
    assert len(fetched) == 2
    assert extracted.count("https://publisher.example/one") == 2
    assert model.calls == 2  # First article's inference fallback remains cached for an hour.
    engines = {article["title"]: article["engine"] for article in refreshed["articles"]}
    assert engines == {"lucro lucro": "PLN Léxico (fallback)", "prejuizo prejuizo": "FinBERT-PT-BR"}
    assert refreshed["summary"]["counts"] == {"positive": 1, "neutral": 0, "negative": 1}
    assert refreshed["model"]["engines_used"] == ["FinBERT-PT-BR", "PLN Léxico (fallback)"]
    assert context.model_retry_at == 0
