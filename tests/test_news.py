import datetime

from utils.news import (
    TICKER_TO_COMPANY,
    aggregate_ticker_sentiment,
    build_news_query,
    parse_rss_items,
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
    </channel></rss>'''

    items = parse_rss_items(xml, now=now)

    assert [item["title"] for item in items] == ["Newest", "First"]
    assert items[0]["published"].tzinfo is not None


def test_rss_duplicate_url_keeps_newest_item_even_when_older_item_appears_first():
    now = datetime.datetime(2025, 1, 8, 12, tzinfo=datetime.timezone.utc)
    xml = b'''<rss><channel>
      <item><title>Older headline</title><link>https://x/story</link><pubDate>Tue, 07 Jan 2025 09:00:00 GMT</pubDate></item>
      <item><title>Updated headline</title><link>HTTPS://x/story</link><pubDate>Tue, 07 Jan 2025 10:00:00 GMT</pubDate></item>
    </channel></rss>'''

    items = parse_rss_items(xml, now=now)

    assert [item["title"] for item in items] == ["Updated headline"]


def test_portfolio_aggregation_weights_ticker_mean_and_excludes_uncovered_weight():
    items = [
        {"ticker": "AAA", "score": 1.0},
        {"ticker": "AAA", "score": -1.0},
        {"ticker": "BBB", "score": 0.5},
    ]

    score, coverage = aggregate_ticker_sentiment(items, {"AAA": 0.25, "BBB": 0.25, "CCC": 0.5})

    assert score == 0.25
    assert coverage == 0.5
