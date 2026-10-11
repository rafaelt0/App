"""Portfolio-bound news analysis; provider work is explicitly POST-only."""
import datetime as dt
import hashlib
import logging
import threading
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException, Request

from backend.serialization import StrictModel, clean_json, figure
from backend.state import begin, publish, require_portfolio, session
from utils.cache import cache_data
from utils.charts import CHART_DANGER, CHART_POSITIVE, CHART_SECONDARY, apply_plotly_theme
from utils.news import (
    NEWS_IMPORTANCE_LABELS, _article_url_key, aggregate_ticker_sentiment,
    analise_sentimento_pln, build_news_query, extract_article_text,
    merge_shared_articles, parse_rss_items, rank_news_importance,
    recent_rss_sample, sentiment_intensity, ticker_tone_rows,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/news", tags=["news"])
MODEL_NAME = "lucas-leme/FinBERT-PT-BR"
MODEL_KEY = MODEL_NAME + ":sigmoid-v1"
_model = None
_model_lock = threading.RLock()


class NewsRequest(StrictModel):
    revision: int
    portfolio_id: str | None = None


@cache_data(ttl=600)
def _fetch_brazilian_news(ticker):
    query = urllib.parse.quote(build_news_query(ticker))
    url = f"https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=8) as response:
        xml = response.read(2 * 1024 * 1024 + 1)
    if len(xml) > 2 * 1024 * 1024:
        raise ValueError("RSS excede 2 MB; feed indisponível, nenhum item pontuado")
    return parse_rss_items(xml)


@cache_data(ttl=60)
def get_brazilian_news(ticker):
    try:
        return {"ok": True, "items": _fetch_brazilian_news(ticker)}
    except Exception:
        logger.warning("News RSS fetch/parse unavailable for %s", ticker, exc_info=True)
        return {"ok": False, "items": []}


@cache_data(ttl=60)
def _cached_article_text(url):
    try:
        return extract_article_text(url)
    except Exception:
        logger.debug("Article extraction unavailable: %s", url, exc_info=True)
        return ""


def load_finbert_pipeline(ctx):
    """Cache only a successfully loaded model; failed-load backoff belongs to the owner."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model
        if time.time() < ctx.model_retry_at:
            return None
        try:
            from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
            _model = pipeline("text-classification", model=model, tokenizer=tokenizer,
                              top_k=None, function_to_apply="sigmoid")
            ctx.model_retry_at = 0
            return _model
        except Exception:
            logger.warning("FinBERT unavailable; using lexicon fallback", exc_info=True)
            ctx.model_retry_at = time.time() + 300
            return None


def _lexicon(title, summary):
    result = analise_sentimento_pln(title, summary)
    # Do not present synthetic lexicon scores as model probability evidence.
    return {**result, "scores": [], "is_finbert": False, "engine": "PLN Léxico (fallback)"}


def analise_sentimento_finbert(title, summary, nlp):
    if nlp is None:
        return _lexicon(title, summary)
    try:
        with _model_lock:
            text = title + (" " + summary if summary else "")
            text = nlp.tokenizer.decode(nlp.tokenizer.encode(
                text, add_special_tokens=False, truncation=True, max_length=510),
                skip_special_tokens=True)
            scores = nlp([text])[0]
        values = {item["label"]: item["score"] for item in scores}
        score = values.get("POSITIVE", 0.0) - values.get("NEGATIVE", 0.0)
        sentiment = "Otimista" if score >= .20 else "Pessimista" if score <= -.20 else "Neutro"
        return {"sentiment": sentiment, "score": round(score, 2), "scores": scores,
                "is_finbert": True, "engine": "FinBERT-PT-BR", "raw_text_length": len(text.split())}
    except Exception:
        logger.warning("FinBERT inference failed; article uses lexicon fallback", exc_info=True)
        return _lexicon(title, summary)


@cache_data(ttl=3600)
def _cached_sentiment(title, summary, model_key, _nlp):
    # _nlp is deliberately excluded from hashing; model_key versions semantics.
    return analise_sentimento_finbert(title, summary, _nlp)


def _safe_link(link):
    return str(link).strip() if _article_url_key(link) else None


def summarize(articles, tickers, weights, feed_status):
    observations = [{"ticker": ticker, "score": item["score"]}
                    for item in articles for ticker in item["tickers"]]
    score, coverage = aggregate_ticker_sentiment(observations, weights)
    note = int((score + 1) / 2 * 100) if coverage > 0 else None
    label = ("SEM DADOS RECENTES" if note is None else
             "FORTEMENTE OTIMISTA" if note >= 80 else "OTIMISTA" if note >= 60 else
             "NEUTRO / EQUILIBRADO" if note >= 40 else "PESSIMISTA" if note >= 20 else "FORTEMENTE PESSIMISTA")
    tone = ticker_tone_rows(articles, tickers, weights, feed_status)
    distribution = [{key: row[key] for key in ("ticker", "positive", "neutral", "negative")} for row in tone]
    chart = go.Figure()
    for key, name, color in (("negative", "Pessimista", CHART_DANGER),
                             ("neutral", "Neutro", CHART_SECONDARY),
                             ("positive", "Otimista", CHART_POSITIVE)):
        chart.add_trace(go.Bar(name=name, y=tickers, x=[row[key] for row in distribution],
                               orientation="h", marker={"color": color}))
    chart.update_layout(barmode="stack", title="Notícias por ativo", height=max(280, 190 + len(tickers) * 40),
                        margin={"l": 48, "r": 12, "t": 46, "b": 118},
                        xaxis={"dtick": 1}, legend={"orientation": "h", "y": -.27, "x": 0})
    apply_plotly_theme(chart)
    status = "neutral" if note is None else "positive" if note >= 60 else "warning" if note >= 40 else "negative"
    return {"raw_score": score if coverage > 0 else None, "score100": note, "label": label,
            "coverage": coverage, "status": status,
            "counts": {key: sum(item["sentiment"] == sentiment for item in articles)
                       for key, sentiment in (("positive", "Otimista"), ("neutral", "Neutro"), ("negative", "Pessimista"))},
            "observation_count": len(observations), "tone": tone, "distribution": distribution,
            "chart": figure(chart)}


def analyze_portfolio(ctx, portfolio):
    fetched = []
    feeds = []
    with ThreadPoolExecutor(max_workers=min(8, len(portfolio.tickers))) as executor:
        results = list(executor.map(get_brazilian_news, portfolio.tickers))
    now = dt.datetime.now(dt.timezone.utc)
    for ticker, feed in zip(portfolio.tickers, results):
        items = recent_rss_sample(feed["items"], now)
        feeds.append({"ticker": ticker, "ok": feed["ok"], "recent_count": len(items)})
        fetched.extend({**item, "ticker": ticker} for item in items)
    merged = merge_shared_articles(fetched)
    nlp = load_finbert_pipeline(ctx) if merged else None
    articles = []
    if merged:
        with ThreadPoolExecutor(max_workers=min(4, len(merged))) as executor:
            texts = list(executor.map(_cached_article_text, (item["link"] for item in merged)))
        for item, text in zip(merged, texts):
            result = _cached_sentiment(item["title"], text or item.get("summary", ""),
                                       MODEL_KEY if nlp is not None else "lexicon:v3", nlp)
            importance = rank_news_importance(item["title"], item.get("summary", ""))
            identity = _article_url_key(item["link"]) or " ".join(item["title"].casefold().split())
            articles.append({"id": hashlib.sha256(identity.encode()).hexdigest(), "tickers": item["tickers"],
                             "title": item["title"], "link": _safe_link(item["link"]), "provider": item["provider"],
                             "published": item["published"].isoformat(), "pub_time": item["date"],
                             "summary": item.get("summary", "") or text[:600],
                             "text_source": "Texto do artigo" if text else "Título + descrição RSS" if item.get("summary") else "Somente título (RSS)",
                             "importance_rank": importance, "importance_label": NEWS_IMPORTANCE_LABELS[importance],
                             "sentiment": result["sentiment"], "score": result["score"], "scores": result["scores"],
                             "is_finbert": result["is_finbert"], "engine": result["engine"],
                             "pos_terms": result.get("pos_terms", []), "neg_terms": result.get("neg_terms", []),
                             "evidence_count": result.get("evidence_count"), "raw_text_length": result["raw_text_length"],
                             "intensity": sentiment_intensity(result["score"], None if result["is_finbert"] else result["evidence_count"])})
    summary = summarize(articles, portfolio.tickers, portfolio.weights, {feed["ticker"]: feed["ok"] for feed in feeds})
    return clean_json({"id": uuid.uuid4().hex, "revision": portfolio.revision, "portfolio_id": portfolio.id,
                       "tickers": portfolio.tickers, "weights": portfolio.weights, "as_of": now.isoformat(),
                       "window_days": 7, "max_per_ticker": 10, "feeds": feeds, "articles": articles, "summary": summary,
                       "model": {"available": nlp is not None, "name": "FinBERT-PT-BR", "engines_used": sorted({a["engine"] for a in articles}),
                                 "retry_after": ctx.model_retry_at if merged and nlp is None else None}})


def _command(request, body, refresh):
    ctx = session(request)
    portfolio = require_portfolio(ctx, returns=False)
    if body.portfolio_id is not None and body.portfolio_id != portfolio.id:
        raise HTTPException(409, "O portfólio mudou; carregue novamente as notícias.")
    generation = begin(ctx, "news", body.revision)
    if refresh:
        # Preserve existing process-shared refresh scope and one-hour sentiment cache.
        _fetch_brazilian_news.clear()
        get_brazilian_news.clear()
        _cached_article_text.clear()
        ctx.model_retry_at = 0
    result = analyze_portfolio(ctx, portfolio)
    publish(ctx, "news", body.revision, generation, result, portfolio_id=portfolio.id)
    return result


@router.post("/analyze")
def analyze(request: Request, body: NewsRequest):
    return _command(request, body, False)


@router.post("/refresh")
def refresh(request: Request, body: NewsRequest):
    return _command(request, body, True)
