import datetime
import email.utils

from utils.news import (
    TICKER_TO_COMPANY,
    aggregate_ticker_sentiment,
    analise_sentimento_pln,
    build_news_query,
    merge_shared_articles,
    parse_rss_items,
    recent_rss_sample,
    sentiment_intensity,
    ticker_tone_rows,
)


def test_news_queries_use_quoted_tickers_and_specific_issuer_names():
    for ticker, company in TICKER_TO_COMPANY.items():
        assert build_news_query(ticker) == f'"{ticker}" OR "{company}"'

    assert build_news_query("VALE3") == '"VALE3" OR "Vale S.A."'
    assert build_news_query("RENT3") == '"RENT3" OR "Localiza Rent a Car"'
    assert build_news_query("UNKNOWN") == '"UNKNOWN"'


def test_rss_filters_recent_parseable_items_deduplicates_and_sorts():
    now = datetime.datetime(2025, 1, 8, 12, tzinfo=datetime.timezone.utc)
    xml = b'''<rss><channel>
      <item><title>Older</title><link>https://x/old</link><pubDate>Tue, 31 Dec 2024 12:00:00 GMT</pubDate></item>
      <item><title>First</title><link>HTTPS://x/a</link><pubDate>Tue, 07 Jan 2025 09:00:00 -0300</pubDate></item>
      <item><title>First duplicate URL</title><link>https://x/a</link><pubDate>Tue, 07 Jan 2025 10:00:00 GMT</pubDate></item>
      <item><title>First</title><link>https://x/other</link><pubDate>Tue, 07 Jan 2025 11:00:00 GMT</pubDate></item>
      <item><title>Unparseable</title><link>https://x/no-date</link><pubDate>yesterday</pubDate></item>
      <item><title>Newest</title><link>https://x/new</link><pubDate>Wed, 08 Jan 2025 11:00:00 +0000</pubDate></item>
      <item><title>At cutoff</title><link>https://x/cutoff</link><pubDate>Wed, 01 Jan 2025 12:00:00 GMT</pubDate></item>
      <item><title>Fourth</title><link>https://x/fourth</link><pubDate>Wed, 08 Jan 2025 10:00:00 GMT</pubDate></item>
    </channel></rss>'''

    items = parse_rss_items(xml, now=now)

    assert [item["link"] for item in items] == [
        "https://x/new", "https://x/fourth", "HTTPS://x/a", "https://x/other", "https://x/cutoff"
    ]
    assert items[0]["published"].tzinfo is not None


def test_rss_duplicate_url_keeps_newest_item_even_when_older_item_appears_first():
    now = datetime.datetime(2025, 1, 8, 12, tzinfo=datetime.timezone.utc)
    xml = b'''<rss><channel>
      <item><title>Older headline</title><link>https://x/story</link><pubDate>Tue, 07 Jan 2025 09:00:00 GMT</pubDate></item>
      <item><title>Updated headline</title><link>HTTPS://x/story</link><pubDate>Tue, 07 Jan 2025 10:00:00 GMT</pubDate></item>
    </channel></rss>'''

    items = parse_rss_items(xml, now=now)

    assert [item["title"] for item in items] == ["Updated headline"]


def test_rss_same_headline_with_distinct_query_ids_is_not_deduplicated():
    now = datetime.datetime(2025, 1, 8, 12, tzinfo=datetime.timezone.utc)
    xml = b'''<rss><channel>
      <item><title>Same headline</title><link>https://x/story?id=1&amp;utm_source=a</link><pubDate>Wed, 08 Jan 2025 11:00:00 GMT</pubDate></item>
      <item><title>Same headline</title><link>https://x/story?id=2</link><pubDate>Wed, 08 Jan 2025 10:00:00 GMT</pubDate></item>
      <item><title>Same headline</title><link>https://x/story?utm_medium=rss&amp;id=1</link><pubDate>Wed, 08 Jan 2025 09:00:00 GMT</pubDate></item>
    </channel></rss>'''

    items = parse_rss_items(xml, now=now)

    assert [item["link"] for item in items] == [
        "https://x/story?id=1&utm_source=a", "https://x/story?id=2"
    ]


def test_shared_articles_are_displayed_once_but_count_for_each_asset():
    rows = [
        {"ticker": "AAA", "title": "Company reports gains", "link": "https://news.example/story?utm_source=aaa", "score": 0.8},
        {"ticker": "BBB", "title": "Company reports gains", "link": "HTTPS://NEWS.EXAMPLE/story#top", "score": 0.8},
    ]

    articles = merge_shared_articles(rows)
    per_ticker_items = [
        {"ticker": ticker, "score": article["score"]}
        for article in articles
        for ticker in article["tickers"]
    ]
    score, coverage = aggregate_ticker_sentiment(
        per_ticker_items, {"AAA": 0.25, "BBB": 0.5, "CCC": 0.25}
    )

    assert len(articles) == 1
    assert articles[0]["tickers"] == ["AAA", "BBB"]
    assert abs(score - 0.8) < 1e-9
    assert coverage == 0.75


def test_per_ticker_sample_keeps_newest_ten_after_freshness_and_dedup_before_merge():
    now = datetime.datetime(2025, 1, 8, 12, tzinfo=datetime.timezone.utc)
    def entry(index, published, link=None):
        url = link or f"https://x/story/{index}"
        return (f"<item><title>Story {index}</title><link>{url}</link>"
                f"<pubDate>{email.utils.format_datetime(published, usegmt=True)}</pubDate></item>")

    # A cached feed can still contain an item just beyond the rolling cutoff.
    old = entry("old", now - datetime.timedelta(days=7, minutes=1))
    entries = [entry(i, now - datetime.timedelta(hours=i + 1)) for i in range(12)]
    duplicate = entry("dup", now - datetime.timedelta(hours=13), "https://x/story/0")
    xml_a = f"<rss><channel>{old}{''.join(reversed(entries))}{duplicate}</channel></rss>".encode()
    xml_b = f"<rss><channel>{entries[0]}{''.join(entry(f'b{i}', now - datetime.timedelta(hours=i + 1)) for i in range(10))}</channel></rss>".encode()
    rows = [
        {**item, "ticker": ticker}
        for ticker, xml in (("AAA", xml_a), ("BBB", xml_b))
        for item in recent_rss_sample(parse_rss_items(xml, now - datetime.timedelta(hours=1)), now)
    ]
    assert len(parse_rss_items(xml_a, now - datetime.timedelta(hours=1))) == 13
    assert [item["link"] for item in rows if item["ticker"] == "AAA"] == [
        f"https://x/story/{i}" for i in range(10)
    ]
    assert len([item for item in rows if item["ticker"] == "BBB"]) == 10
    merged = merge_shared_articles(rows)
    assert len(merged) == 19
    assert merged[0]["tickers"] == ["AAA", "BBB"]


def test_petrobras_despenca_headline_is_pessimistic_with_one_negative_term():
    result = analise_sentimento_pln(
        "Petrobras despenca e arrasta Ibovespa (IBOV), enquanto dólar e euro avançam", ""
    )
    assert result["sentiment"] == "Pessimista"
    assert result["score"] == -1.0
    assert result["neg_terms"] == ["despenca"]
    assert result["pos_terms"] == []
    assert result["evidence_count"] == 1


def test_single_term_lexicon_score_is_labeled_low_evidence_not_high_intensity():
    assert sentiment_intensity(1.0, evidence_count=1) == "Evidência limitada"
    assert sentiment_intensity(1.0, evidence_count=2) == "Alto"


def test_lexicon_flips_simple_negation_and_marks_single_term_evidence():
    no_profit = analise_sentimento_pln("Sem lucro no trimestre", "")
    no_loss = analise_sentimento_pln("Não houve prejuízo no trimestre", "")

    assert (no_profit["sentiment"], no_profit["score"], no_profit["evidence_count"]) == (
        "Pessimista", -1.0, 1
    )
    assert (no_loss["sentiment"], no_loss["score"], no_loss["evidence_count"]) == (
        "Otimista", 1.0, 1
    )


def test_negation_does_not_flip_a_second_term_in_the_same_clause():
    result = analise_sentimento_pln("Sem prejuízo e queda", "")

    assert result["pos_terms"] == ["prejuizo"]
    assert result["neg_terms"] == ["queda"]
    assert result["sentiment"] == "Neutro"


def test_portfolio_aggregation_weights_ticker_mean_and_excludes_uncovered_weight():
    items = [
        {"ticker": "AAA", "score": 1.0},
        {"ticker": "AAA", "score": -1.0},
        {"ticker": "BBB", "score": 0.5},
    ]

    score, coverage = aggregate_ticker_sentiment(items, {"AAA": 0.25, "BBB": 0.25, "CCC": 0.5})

    assert score == 0.25
    assert coverage == 0.5


def test_rss_description_is_plain_text_and_utc():
    now = datetime.datetime(2025, 1, 8, 12, tzinfo=datetime.timezone.utc)
    xml = b'''<rss><channel><item><title>News</title><link>https://example.com/a</link>
    <description>&lt;p&gt;Profit &amp;amp; loss&lt;/p&gt;</description>
    <pubDate>Wed, 08 Jan 2025 11:00:00 GMT</pubDate></item></channel></rss>'''
    item = parse_rss_items(xml, now)[0]
    assert item['summary'] == 'Profit & loss'
    assert item['date'].endswith('UTC')


def test_distinct_queries_do_not_merge_but_tracking_does():
    rows = [{'ticker': ticker, 'title': 'Same', 'link': link} for ticker, link in [
        ('A', 'https://example.com/story?id=1&utm_source=x'),
        ('B', 'https://example.com/story?id=2'),
        ('C', 'https://example.com/story?utm_medium=rss&id=1'),
    ]]
    merged = merge_shared_articles(rows)
    assert len(merged) == 2
    assert merged[0]['tickers'] == ['A', 'C']


def test_lexicon_sentence_boundary_phrase_repetition_and_word_count():
    result = analise_sentimento_pln('Não. lucro; reduz dívida e reduz dívida', '')
    assert result['pos_terms'] == ['lucro', 'reduz divida', 'reduz divida']
    assert result['raw_text_length'] == 7


def test_article_extraction_rejects_private_and_redirects(monkeypatch):
    import socket
    import pytest
    from utils import news

    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [(socket.AF_INET, 0, 0, '', ('127.0.0.1', 80))])
    for url in ('file:///etc/passwd', 'http://user:pass@example.com', 'http://example.com'):
        with pytest.raises(ValueError):
            news.extract_article_text(url)

    monkeypatch.setattr(socket, 'getaddrinfo', lambda host, *args, **kwargs: [(socket.AF_INET, 0, 0, '', ('127.0.0.1' if host == '127.0.0.1' else '8.8.8.8', 80))])
    class Response:
        status = 302
        def getheader(self, name, default=None):
            return 'http://127.0.0.1/private' if name == 'Location' else default
    class Conn:
        sock = type("Sock", (), {"settimeout": lambda self, timeout: None})()
        def close(self): pass
    monkeypatch.setattr(news, '_open_article', lambda *args: (Conn(), Response()))
    with pytest.raises(ValueError):
        news.extract_article_text('http://example.com/story')


def test_article_extraction_html_and_limits(monkeypatch):
    from utils import news
    monkeypatch.setattr(news, '_public_address', lambda url: (type('URL', (), {'scheme': 'https'})(), '8.8.8.8', 443))
    class Conn:
        sock = type("Sock", (), {"settimeout": lambda self, timeout: None})()
        def close(self): pass
    class Response:
        status = 200
        def __init__(self, body, content_type='text/html'):
            self.body, self.content_type = body, content_type
        def getheader(self, name, default=None):
            return self.content_type if name == 'Content-Type' else default
        def read1(self, limit):
            chunk, self.body = self.body[:limit], self.body[limit:]
            return chunk
    paragraph = 'Useful publisher paragraph with sufficient content to analyze. ' * 5
    body = f'<article><nav>junk</nav><p>{paragraph}</p></article>'.encode()
    monkeypatch.setattr(news, '_open_article', lambda *args: (Conn(), Response(body)))
    assert news.extract_article_text('https://example.com') == paragraph.strip()
    monkeypatch.setattr(news, '_open_article', lambda *args: (Conn(), Response(body, 'image/png')))
    assert news.extract_article_text('https://example.com') == ''
    monkeypatch.setattr(news, '_open_article', lambda *args: (Conn(), Response(b'x' * (1024 * 1024 + 1))))
    assert news.extract_article_text('https://example.com') == ''


def test_article_timeout_falls_back_per_item(monkeypatch):
    import socket
    from utils import news
    monkeypatch.setattr(news, '_public_address', lambda url: (None, '8.8.8.8', 443))
    def timed_out(*args):
        raise socket.timeout('blocked')
    monkeypatch.setattr(news, '_open_article', timed_out)
    # The page isolates extraction errors per URL and retains the RSS description.
    try:
        news.extract_article_text('https://example.com')
    except socket.timeout:
        assert analise_sentimento_pln('lucro', 'queda')['evidence_count'] == 2
    else:
        assert False, 'timeout must reach the caller for RSS fallback'


def test_tone_rows_shared_article_coverage_and_raw_weights():
    articles = merge_shared_articles([
        {"ticker": "AAA", "title": "Shared", "link": "https://x/shared?utm_source=a", "score": 1, "sentiment": "Otimista"},
        {"ticker": "BBB", "title": "Shared", "link": "https://x/shared", "score": 1, "sentiment": "Otimista"},
        {"ticker": "AAA", "title": "Exclusive", "link": "https://x/exclusive", "score": -1, "sentiment": "Pessimista"},
    ])
    assert len(articles) == 2
    assert len([a for a in articles if "BBB" in a["tickers"]]) == 1  # ticker filter keeps shared card
    observations = [{"ticker": t, "score": a["score"]} for a in articles for t in a["tickers"]]
    for weights in ({"AAA": .2, "BBB": .3, "CCC": .5}, {"AAA": 2, "BBB": 3, "CCC": 5}):
        rows = ticker_tone_rows(articles, list(weights), weights, {"AAA": True, "BBB": True, "CCC": True})
        score, coverage = aggregate_ticker_sentiment(observations, weights)
        assert (score, coverage, int((score + 1) * 50)) == (.6, .5, 80)
        assert [(r["count"], r["positive"], r["neutral"], r["negative"], r["mean"])
                for r in rows] == [(2, 1, 0, 1, 0), (1, 1, 0, 0, 1), (0, 0, 0, 0, None)]
        assert [r["contribution"] for r in rows] == [0, 30, None]
        assert sum(r["contribution"] for r in rows if r["contribution"] is not None) == 50 * score
        assert [r["covered_share"] for r in rows] == [.4, .6, None]


def test_tone_rows_distinguish_failed_and_empty_feeds_without_neutral_score():
    weights = {"AAA": .4, "BBB": .6}
    for status, expected in [
        ({"AAA": True, "BBB": True}, ["Sem notícias nos últimos 7 dias"] * 2),
        ({"AAA": False, "BBB": False}, ["Feed indisponível"] * 2),
        ({"AAA": False, "BBB": True}, ["Feed indisponível", "Sem notícias nos últimos 7 dias"]),
    ]:
        rows = ticker_tone_rows([], list(weights), weights, status)
        score, coverage = aggregate_ticker_sentiment([], weights)
        assert [r["state"] for r in rows] == expected
        assert all(r["count"] == 0 and r["mean"] is None and r["contribution"] is None for r in rows)
        assert (score, coverage) == (0, 0)
        display = str(int((score + 1) * 50)) if coverage else "—"
        assert display == "—"

    article = [{"tickers": ["BBB"], "score": 1, "sentiment": "Otimista"}]
    rows = ticker_tone_rows(article, list(weights), weights, {"AAA": False, "BBB": True})
    assert rows[0]["state"] == "Feed indisponível"
    assert rows[1]["contribution"] == 50
    assert aggregate_ticker_sentiment([{"ticker": "BBB", "score": 1}], weights) == (1, .6)
