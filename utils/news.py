import datetime
import email.utils
import re
import unicodedata
import xml.etree.ElementTree as ET


def parse_rss_items(xml_data, now=None, limit=3):
    """Parse recent RSS entries; return newest unique articles with aware dates."""
    if limit <= 0:
        return []
    now = now or datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=7)
    root = ET.fromstring(xml_data)
    candidates = []

    def text(item, tag):
        node = item.find(tag)
        return node.text.strip() if node is not None and node.text else ""

    for item in root.findall(".//item"):
        title, link, raw_date = text(item, "title"), text(item, "link"), text(item, "pubDate")
        source = text(item, "source")
        if source and title.endswith(f" - {source}"):
            title = title[:-len(f" - {source}")]
        try:
            published = email.utils.parsedate_to_datetime(raw_date)
            if published is None:
                continue
            if published.tzinfo is None:
                published = published.replace(tzinfo=datetime.timezone.utc)
            published = published.astimezone(datetime.timezone.utc)
        except (TypeError, ValueError, OverflowError):
            continue
        if not title or published > now or published < cutoff:
            continue
        candidates.append({"title": title, "link": link, "date": published.strftime("%d/%m/%Y %H:%M"),
                          "published": published, "provider": source, "summary": ""})

    def normalized(value):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()

    unique = []
    seen_urls = set()
    seen_titles = set()
    for item in sorted(candidates, key=lambda article: article["published"], reverse=True):
        url_key, title_key = normalized(item["link"]), normalized(item["title"])
        if (url_key and url_key in seen_urls) or title_key in seen_titles:
            continue
        if url_key:
            seen_urls.add(url_key)
        seen_titles.add(title_key)
        unique.append(item)
        if len(unique) == limit:
            break
    return unique


def aggregate_ticker_sentiment(items, weights):
    """Weight each covered ticker's mean news score once; report covered portfolio weight."""
    by_ticker = {}
    for item in items:
        by_ticker.setdefault(item["ticker"], []).append(float(item["score"]))
    covered = {ticker: sum(scores) / len(scores) for ticker, scores in by_ticker.items() if scores}
    total_weight = sum(float(weights.get(ticker, 0)) for ticker in covered)
    score = (sum(mean * float(weights.get(ticker, 0)) for ticker, mean in covered.items()) / total_weight
             if total_weight else 0.0)
    return score, total_weight
