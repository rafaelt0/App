import datetime
import email.utils
import re
import unicodedata
import xml.etree.ElementTree as ET


# Use issuer names rather than ambiguous shorthand in Google News searches.
TICKER_TO_COMPANY = {
    "PETR3": "Petrobras", "PETR4": "Petrobras",
    "VALE3": "Vale S.A.",
    "ITUB3": "Itaú Unibanco", "ITUB4": "Itaú Unibanco",
    "BBDC3": "Bradesco", "BBDC4": "Bradesco",
    "BBAS3": "Banco do Brasil",
    "WEGE3": "WEG S.A.",
    "MGLU3": "Magazine Luiza",
    "ABEV3": "Ambev",
    "ELET3": "Eletrobras", "ELET6": "Eletrobras",
    "RENT3": "Localiza Rent a Car",
    "LREN3": "Lojas Renner",
    "PRIO3": "PRIO S.A.",
    "HAPV3": "Hapvida",
    "SANB11": "Santander Brasil",
    "VVAR3": "Via Varejo", "BHIA3": "Casas Bahia",
    "GGBR4": "Gerdau",
    "ITSA4": "Itaúsa",
    "SUZB3": "Suzano S.A.",
    "JBSS3": "JBS",
    "UGPA3": "Ultrapar",
    "RADL3": "Raia Drogasil",
    "EQTL3": "Equatorial Energia",
    "CSAN3": "Cosan",
    "CPFE3": "CPFL Energia",
    "SBSP3": "Sabesp",
    "TAEE11": "Taesa",
    "KLBN11": "Klabin",
}


def build_news_query(ticker):
    """Search exact tickers and issuer names; unknown tickers stay ticker-only."""
    terms = [f'"{ticker}"']
    company = TICKER_TO_COMPANY.get(ticker)
    if company:
        terms.append(f'"{company}"')
    return " OR ".join(terms)


NEWS_IMPORTANCE_LABELS = {1: "Baixa", 2: "Média", 3: "Alta"}

_MATERIAL_NEWS_TERMS = (
    "lucro", "prejuizo", "resultado financeiro", "balanco", "dividend",
    "guidance", "projec", "aquisic", "fusao", "venda de ativo",
    "emissao de acoes", "recompra", "recuperacao judicial", "falencia",
    "default", "divida bilionaria", "multa", "processa", "investiga",
    "cvm", "cade", "banco central", "greve", "paralisac", "acidente",
    "vazamento", "rompimento", "interrupcao", "demissao em massa",
    "contrato bilionario", "capex",
)
_CUSTOMER_TERMS = ("cliente", "consumidor", "usuario")
_COMPLAINT_TERMS = (
    "reclama", "critica", "detona", "insatisf", "fala mal", "queixa",
    "reclame aqui", "mau atendimento",
)
_LOW_IMPORTANCE_TERMS = (
    "patrocin", "campanha", "marketing", "promocao de marca",
    "evento esportivo", "campeonato", "torneio", "clube de futebol",
    "copa do mundo", "olimpiada",
)


def rank_news_importance(title, summary=""):
    """Return 1 (low), 2 (medium), or 3 (high) from the headline and any RSS snippet."""
    text = unicodedata.normalize("NFKD", f"{title or ''} {summary or ''}").casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))

    # ponytail: keyword heuristic; replace with a labeled classifier if rankings prove unreliable.
    if any(term in text for term in _MATERIAL_NEWS_TERMS):
        return 3
    if any(term in text for term in _LOW_IMPORTANCE_TERMS) or (
        any(term in text for term in _CUSTOMER_TERMS)
        and any(term in text for term in _COMPLAINT_TERMS)
    ):
        return 1
    return 2


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
