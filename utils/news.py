import datetime
import http.client
import ipaddress
import socket
import time

from bs4 import BeautifulSoup
import email.utils
import re
import unicodedata
import urllib.parse
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


_TRACKING_QUERY_PARAMS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid"}


def _article_url_key(url):
    try:
        parsed = urllib.parse.urlsplit(str(url).strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        if parsed.username is not None or parsed.password is not None:
            return ""
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname.lower()
        port = parsed.port
    except ValueError:
        return ""

    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port and port != (443 if scheme == "https" else 80):
        netloc += f":{port}"
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_PARAMS
    ]
    return urllib.parse.urlunsplit((
        scheme, netloc, parsed.path.rstrip("/") or "/",
        urllib.parse.urlencode(sorted(query)), "",
    ))


def parse_rss_items(xml_data, now=None):
    """Parse all recent unique entries returned by the feed, newest first."""
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
        description = BeautifulSoup(text(item, "description"), "html.parser").get_text(" ", strip=True)
        candidates.append({"title": title, "link": link, "date": published.strftime("%d/%m/%Y %H:%M UTC"),
                          "published": published, "provider": source, "summary": description[:1200]})

    def normalized(value):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()

    unique = []
    seen_urls = set()
    seen_titles_without_url = set()
    for item in sorted(candidates, key=lambda article: article["published"], reverse=True):
        url_key = _article_url_key(item["link"])
        title_key = normalized(item["title"])
        if url_key:
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
        else:
            if title_key in seen_titles_without_url:
                continue
            seen_titles_without_url.add(title_key)
        unique.append(item)
    return unique


def recent_rss_sample(items, now):
    """Keep up to 10 newest unique entries per feed after the rolling cutoff."""
    cutoff = now - datetime.timedelta(days=7)
    return [item for item in items if cutoff <= item["published"] <= now][:10]


def analise_sentimento_pln(title, summary):
    """Estimate Portuguese financial-news sentiment from a small lexicon."""
    text = unicodedata.normalize("NFD", f"{title} {summary}".lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    positive = {
        "alta", "lucro", "lucros", "recorde", "recordes", "crescimento", "crescimentos",
        "descoberta", "descobertas", "acordo", "acordos", "aprova", "aprovou", "aprovado",
        "aprovados", "aprovacao", "expande", "expandiu", "expansao", "lider", "lidera", "lideranca",
        "recuperacao", "recuperou", "forte", "fortes", "positivo", "positiva", "positivos",
        "ganho", "ganhos", "ganhou", "compra", "compras", "dividendos", "dividendo", "jcp",
        "eficiencia", "eficiente", "modernizacao", "parceria", "parcerias", "valorizacao",
        "valorizou", "descarbonizacao", "melhora", "melhorou", "subiu", "subiram", "superou",
        "superaram", "estabilizou", "estabilidade", "confortavel",
    }
    negative = {
        "queda", "quedas", "prejuizo", "prejuizos", "greve", "greves", "suspende", "suspendeu",
        "inadimplencia", "atraso", "atrasos", "pressao", "pressoes", "perda", "perdas", "reducao",
        "rebaixado", "rebaixada", "rebaixamento", "concorrencia", "fraca", "fraco", "sofre",
        "sofreu", "paralisacao", "divida", "dividas", "pessimista", "crise", "risco", "riscos",
        "cair", "caiu", "recua", "recuou", "despenca", "despencam", "despencou",
        "despencaram", "despencando", "defasagem", "alavancagem",
    }
    positive_phrases = (
        "reduz divida", "reduz dividas", "reduz custo", "reduz custos", "reduz inadimplencia",
        "reducao de divida", "reducao de dividas", "reducao de custos", "reducao de custo",
        "queda da inadimplencia", "queda de inadimplencia", "reduzindo divida", "reduz alavancagem",
        "reduzir divida", "reducao de alavancagem", "renegociacao de dividas",
    )
    negative_phrases = (
        "aumento de custos", "aumento de despesas", "alta da inadimplencia", "aumento de divida",
        "aumento de dividas", "aumento da inadimplencia",
    )
    negations = {"nao", "nunca", "jamais", "sem"}
    matched_positive, matched_negative = [], []
    word_count = len(re.findall(r"[a-z0-9_]+", text))
    # Sentence boundaries reset negation; phrase matches consume tokens once.
    phrases = sorted(((p, True) for p in positive_phrases), key=lambda x: -len(x[0])) + sorted(
        ((p, False) for p in negative_phrases), key=lambda x: -len(x[0]))
    for sentence in re.split(r"[.!?;:]+", text):
        words = re.findall(r"[a-z0-9_]+", sentence)
        index = 0
        while index < len(words):
            match = next(((p, sign, p.split()) for p, sign in phrases
                          if words[index:index + len(p.split())] == p.split()), None)
            if match:
                phrase, is_positive, tokens = match
                step = len(tokens)
                evidence = phrase
            else:
                evidence, step = words[index], 1
                is_positive = evidence in positive
            if match or evidence in positive or evidence in negative:
                negated = False
                for previous in reversed(words[max(0, index - 3):index]):
                    if previous in negations:
                        negated = True
                        break
                    if previous in positive or previous in negative:
                        break
                target = matched_negative if is_positive == negated else matched_positive
                target.append(evidence)
            index += step

    positive_count, negative_count = len(matched_positive), len(matched_negative)
    evidence_count = positive_count + negative_count
    score = (positive_count - negative_count) / evidence_count if evidence_count else 0.0
    sentiment = "Otimista" if score >= 0.2 else "Pessimista" if score <= -0.2 else "Neutro"
    return {
        "sentiment": sentiment,
        "score": round(score, 2),
        "pos_terms": matched_positive,
        "neg_terms": matched_negative,
        "raw_text_length": word_count,
        "evidence_count": evidence_count,
    }


def sentiment_intensity(score, evidence_count=None):
    """Avoid implying strong confidence from one or zero lexical matches."""
    if evidence_count is not None and evidence_count < 2:
        return "Evidência limitada"
    try:
        intensity = abs(float(score))
    except (TypeError, ValueError):
        return "Baixo"
    if intensity >= 0.8:
        return "Alto"
    if intensity >= 0.6:
        return "Médio-Alto"
    if intensity >= 0.3:
        return "Médio"
    if intensity >= 0.1:
        return "Baixo-Médio"
    return "Baixo"


def merge_shared_articles(items):
    """Show a shared story once while retaining every ticker it covers."""
    merged, by_url, by_title = [], {}, {}
    for item in items:
        url_key = _article_url_key(item.get("link", ""))
        title_key = " ".join(
            unicodedata.normalize("NFKC", str(item.get("title", ""))).casefold().split()
        )
        index = by_url.get(url_key) if url_key else None
        if index is None and title_key and not url_key:
            index = by_title.get(title_key)
        if index is None:
            index = len(merged)
            merged.append({**item, "tickers": [item["ticker"]]})
        elif item["ticker"] not in merged[index]["tickers"]:
            merged[index]["tickers"].append(item["ticker"])
        if url_key:
            by_url[url_key] = index
        if title_key and not url_key:
            by_title[title_key] = index
    return merged


def aggregate_ticker_sentiment(items, weights):
    """Weight each covered ticker's mean news score once; report covered portfolio weight."""
    by_ticker = {}
    for item in items:
        by_ticker.setdefault(item["ticker"], []).append(float(item["score"]))
    covered = {ticker: sum(scores) / len(scores) for ticker, scores in by_ticker.items() if scores}
    total_weight = sum(float(weights.get(ticker, 0)) for ticker in covered)
    score = (sum(mean * float(weights.get(ticker, 0)) for ticker, mean in covered.items()) / total_weight
             if total_weight else 0.0)
    portfolio_weight = sum(float(weight) for weight in weights.values())
    return score, total_weight / portfolio_weight if portfolio_weight else 0.0


def ticker_tone_rows(articles, tickers, weights, feed_status):
    """Per-asset observations and contributions in covered-score points."""
    by_ticker = {ticker: [] for ticker in tickers}
    for article in articles:
        for ticker in article["tickers"]:
            if ticker in by_ticker:
                by_ticker[ticker].append(article)
    covered_weight = sum(float(weights[t]) for t, items in by_ticker.items() if items)
    rows = []
    for ticker, items in by_ticker.items():
        mean = sum(item["score"] for item in items) / len(items) if items else None
        rows.append({
            "ticker": ticker,
            "state": "Feed indisponível" if not feed_status.get(ticker) else
                     "Sem notícias nos últimos 7 dias" if not items else "Com notícias",
            "count": len(items),
            "positive": sum(item["sentiment"] == "Otimista" for item in items),
            "neutral": sum(item["sentiment"] == "Neutro" for item in items),
            "negative": sum(item["sentiment"] == "Pessimista" for item in items),
            "mean": mean,
            "covered_share": float(weights[ticker]) / covered_weight if covered_weight and items else None,
            "contribution": 50 * float(weights[ticker]) / covered_weight * mean if covered_weight and items else None,
        })
    return rows


def _public_address(url):
    """Resolve and pin a public address before connecting (including redirects)."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or "@" in parsed.netloc:
        raise ValueError("Invalid article URL")
    try:
        try:
            literal = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            literal = None
        if literal is not None and not literal.is_global:
            raise ValueError("Non-public article address")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        ips = [address[4][0] for address in addresses]
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            raise ValueError("Non-public article address")
    except (socket.gaierror, ValueError) as exc:
        raise ValueError("Unsafe article address") from exc
    return parsed, ips[0], port


def _open_article(parsed, ip, port):
    """Use the verified IP for TCP; preserve the hostname for Host and TLS/SNI."""
    class PinnedHTTPS(http.client.HTTPSConnection):
        def connect(self):
            self.sock = socket.create_connection((ip, port), timeout=self.timeout)
            self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)

    class PinnedHTTP(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.create_connection((ip, port), timeout=self.timeout)

    cls = PinnedHTTPS if parsed.scheme == "https" else PinnedHTTP
    conn = cls(parsed.hostname, port, timeout=3)
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    conn.request("GET", path, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
    return conn, conn.getresponse()


def extract_article_text(url):
    """Best-effort publisher paragraphs; never fetch private destinations or unbounded HTML."""
    deadline = time.monotonic() + 12
    for _ in range(4):
        if time.monotonic() >= deadline:
            return ""
        parsed, ip, port = _public_address(url)
        conn = None
        try:
            conn, response = _open_article(parsed, ip, port)
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    return ""
                url = urllib.parse.urljoin(url, location)
                continue
            if response.status != 200 or "text/html" not in response.getheader("Content-Type", "").lower():
                return ""
            chunks, size = [], 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ""
                # A peer trickling data cannot extend the total fetch budget.
                conn.sock.settimeout(min(3, remaining))
                chunk = response.read1(min(65536, 1024 * 1024 + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > 1024 * 1024:
                    return ""
            body = b"".join(chunks)
            soup = BeautifulSoup(body, "html.parser")
            root = soup.find("article") or soup.find("main")
            if not root:
                return ""
            for node in root.select("script, style, nav, footer, aside, header"):
                node.decompose()
            paragraphs = [p.get_text(" ", strip=True) for p in root.find_all("p")]
            text = " ".join(p for p in paragraphs if len(p) >= 40)[:6000]
            return text if len(text) >= 200 else ""
        finally:
            if conn is not None:
                conn.close()
    return ""
