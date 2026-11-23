import csv
import datetime
import functools
import http.client
import ipaddress
from pathlib import Path
import socket
import time

from bs4 import BeautifulSoup
import email.utils
import re
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET

from utils.assets import asset_metadata
from utils.treasury_data import TREASURY_TITLES


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
    # FIIs, US Stocks, REITs, Bonds & Fixed Income
    "HGLG11": "CSHG Logística FII",
    "KNRI11": "Kinea Renda Imobiliária",
    "MXRF11": "Maxi Renda FII",
    "XPML11": "XP Malls FII",
    "VISC11": "Vinci Shopping Centers",
    "BTLG11": "BTG Pactual Logística",
    "KNCR11": "Kinea Rendimentos Imobiliários",
    "KNIP11": "Kinea Índices de Preços",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet Google",
    "GOOG": "Alphabet Google",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase",
    "V": "Visa Inc",
    "MA": "Mastercard",
    "LLY": "Eli Lilly",
    "AVGO": "Broadcom",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
    "UNH": "UnitedHealth",
    "O": "Realty Income REIT",
    "PLD": "Prologis REIT",
    "AMT": "American Tower REIT",
    "EQIX": "Equinix REIT",
    "SPG": "Simon Property Group",
    "WELL": "Welltower REIT",
    "PSA": "Public Storage REIT",
    "DLR": "Digital Realty REIT",
    "VICI": "VICI Properties REIT",
    "VNQ": "Vanguard Real Estate ETF",
    "TLT": "Treasury Bond ETF",
    "IEF": "Treasury Bond ETF",
    "SHY": "Treasury Bond ETF",
    "SGOV": "Treasury Bill ETF",
    "BND": "Vanguard Total Bond ETF",
    "AGG": "US Aggregate Bond ETF",
    "IMAB11": "IMA-B Tesouro IPCA",
    "B5P211": "IMA-B5 Tesouro IPCA",
    "IRFM11": "IRF-M Tesouro Prefixado",
    "LFTS11": "Tesouro Selic ETF",
    "CDI": "Taxa CDI Selic",
    "SELIC": "Taxa Selic Copom",
    "TESOURO-SELIC": "Tesouro Selic",
    "TESOURO-IPCA": "Tesouro IPCA",
    "TESOURO-IPCA-2035": "Tesouro IPCA",
    "TESOURO-IPCA-2045": "Tesouro IPCA",
    "TESOURO-PRE": "Tesouro Prefixado",
    "POUPANCA": "Caderneta de Poupança",
    "SPY": "S&P 500 ETF",
    "VOO": "Vanguard S&P 500",
    "QQQ": "Nasdaq 100 ETF",
    "BOVA11": "Ibovespa BOVA11",
    "IVVB11": "S&P 500 IVVB11",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

# Validated financial and business news outlets recognized by the Brazilian market.
TRUSTED_NEWS_SOURCES = {
    "valor.globo.com": "Valor Econômico",
    "valorinveste.globo.com": "Valor Investe",
    "infomoney.com.br": "InfoMoney",
    "moneytimes.com.br": "Money Times",
    "seudinheiro.com": "Seu Dinheiro",
    "suno.com.br": "Suno",
    "investidor10.com.br": "Investidor10",
    "einvestidor.estadao.com.br": "E-Investidor",
    "estadao.com.br": "Estadão",
    "exame.com": "Exame",
    "braziljournal.com": "Brazil Journal",
    "neofeed.com.br": "NeoFeed",
    "bloomberglinea.com.br": "Bloomberg Línea",
    "forbes.com.br": "Forbes Brasil",
    "investnews.com.br": "InvestNews",
    "economia.uol.com.br": "UOL Economia",
    "br.investing.com": "Investing.com",
    "br.advfn.com": "ADVFN",
    "conteudos.xpi.com.br": "XP Investimentos",
    "borainvestir.b3.com.br": "Bora Investir",
    "investalk.bb.com.br": "InvesTalk",
    "analisa.genialinvestimentos.com.br": "Genial Analisa",
    "timesbrasil.com.br": "Times Brasil",
}


def _fold_text(value):
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(char for char in text if not unicodedata.combining(char))


_TRUSTED_PROVIDER_NAMES = {
    **{_fold_text(name): name for name in TRUSTED_NEWS_SOURCES.values()},
    "suno - investimentos inteligentes": "Suno",
    "investing.com brasil - financas, cambio e investimentos": "Investing.com",
    "investing.com brasil": "Investing.com",
    "bloomberg linea brasil": "Bloomberg Línea",
    "times brasil | cnbc": "Times Brasil",
    "xp": "XP Investimentos",
    "e-investidor": "E-Investidor",
    "estadao e-investidor": "E-Investidor",
}


def is_trusted_news_source(source_url="", provider="", link=""):
    """Accept only validated Brazilian financial and market news sources."""
    for candidate in (source_url, link):
        try:
            host = (urllib.parse.urlsplit(str(candidate or "").strip()).hostname or "").lower().removeprefix("www.")
        except ValueError:
            host = ""
        if host and any(host == domain or host.endswith(f".{domain}") for domain in TRUSTED_NEWS_SOURCES):
            return True
    if not str(source_url or "").strip():
        return _fold_text(provider).strip() in _TRUSTED_PROVIDER_NAMES
    return False


_GENERIC_ISSUER_WORDS = {
    "s/a", "s.a", "s.a.", "sa", "on", "pn", "pna", "pnb", "unt",
    "cia", "companhia", "bco", "banco", "brasil", "brasileira",
    "part", "participacoes", "holding", "holdings", "grupo",
    "empreend", "empreendimentos", "investimentos", "invest",
    "energia", "energ", "alimentos", "comercio", "com", "ind", "indl",
    "servicos", "engenharia", "construcao", "industria", "industrias",
    "nacional", "geral", "sul", "norte", "leste", "oeste", "estado", "est",
    "realt", "realty", "prop", "tec", "tech", "educ", "educa", "log",
    "seguri", "seguridade", "met", "armas", "modas", "par",
}

_ISSUER_ALIASES = {
    "PETR": ("petrobras",),
    "VALE": ("vale",),
    "ITUB": ("itau", "itau unibanco"),
    "BBDC": ("bradesco",),
    "BBAS": ("banco do brasil",),
    "BPAC": ("btg pactual", "btg"),
    "SANB": ("santander",),
    "WEGE": ("weg",),
    "MGLU": ("magazine luiza", "magalu"),
    "ABEV": ("ambev",),
    "ELET": ("eletrobras", "axia", "axia3", "axia6"),
    "AXIA": ("axia", "eletrobras", "elet3", "elet6"),
    "RENT": ("localiza",),
    "LREN": ("lojas renner", "renner"),
    "PRIO": ("prio", "petrorio"),
    "HAPV": ("hapvida",),
    "VVAR": ("via varejo", "casas bahia", "bhia3"),
    "BHIA": ("casas bahia",),
    "GGBR": ("gerdau",),
    "GOAU": ("gerdau", "metalurgica gerdau"),
    "ITSA": ("itausa",),
    "SUZB": ("suzano",),
    "JBSS": ("jbs", "jbss32"),
    "UGPA": ("ultrapar",),
    "RADL": ("raia drogasil", "rd saude"),
    "EQTL": ("equatorial",),
    "CSAN": ("cosan",),
    "CPFE": ("cpfl",),
    "SBSP": ("sabesp",),
    "TAEE": ("taesa",),
    "KLBN": ("klabin",),
    "RDOR": ("rede d'or", "rede dor", "rede d or"),
    "VIVT": ("telefonica", "vivo"),
    "TIMS": ("tim",),
    "RAIL": ("rumo",),
    "VBBR": ("vibra",),
    "ASAI": ("assai",),
    "B3SA": ("b3",),
    "EMBR": ("embraer", "embj3"),
    "CCRO": ("ccr", "motiva", "motv3"),
    "MOTV": ("motiva", "ccr", "ccro3"),
    "CPLE": ("copel",),
    "CMIG": ("cemig",),
    "CSNA": ("csn", "siderurgica nacional"),
    "USIM": ("usiminas",),
    "BBSE": ("bb seguridade",),
    "CXSE": ("caixa seguridade",),
    "IRBR": ("irb",),
    "PSSA": ("porto seguro",),
    "COGN": ("cogna",),
    "YDUQ": ("yduqs",),
    "TOTS": ("totvs",),
    "LWSA": ("lwsa", "locaweb"),
    "CYRE": ("cyrela",),
    "MRVE": ("mrv",),
    "EGIE": ("engie",),
    "ENEV": ("eneva",),
    "BRAV": ("brava", "brava energia"),
}

_NON_ECONOMIC_HEADLINE_TERMS = (
    "futebol", "paulistao", "brasileirao", "campeonato", "torneio",
    "copa do mundo", "olimpiada", "sub-17", "sub-20", "elenco", "superliga",
    "volei", "basquete", "corrida de rua", "maratona", "show", "festival",
    "teatro", "cinema", "exposicao", "carnaval", "forum longevidade",
    "viver mais e melhor", "qualidade de vida", "saude fisica",
    "envelhecer", "envelhecimento",
)

_FINANCIAL_CONTEXT_TERMS = (
    "acao", "acoes", "papel", "papeis", "ibovespa", "ibov", "b3", "bolsa",
    "adr", "adrs", "bdr", "bdrs", "dividendo", "dividendos", "jcp", "provento",
    "proventos", "payout", "lucro", "lucra", "prejuizo", "balanco", "resultado",
    "resultados", "trimestre", "1t", "2t", "3t", "4t", "receita", "faturamento",
    "ebitda", "margem", "caixa", "divida", "alavancagem", "capex", "investimento",
    "investimentos", "plano de negocios", "guidance", "projecao", "projecoes",
    "preco-alvo", "recomendacao", "compra", "venda", "neutra", "neutro",
    "analista", "analistas", "mercado", "cotacao", "cotacoes", "alta", "queda",
    "sobe", "sobem", "cai", "caem", "recua", "recuam", "avanca", "avancam",
    "dispara", "despenca", "salta", "tomba", "aquisicao", "fusao",
    "joint venture", "debenture", "debentures", "emissao", "oferta", "ipo",
    "follow-on", "recompra", "cvm", "cade", "aneel", "anp", "bndes",
    "banco central", "selic", "juros", "credito", "inadimplencia", "provisao",
    "provisoes", "tarifa", "leilao", "concessao", "privatizacao", "ceo",
    "cfo", "presidente", "conselho", "diretoria", "governanca", "acordo",
    "contrato", "operacao", "operacoes", "producao", "capacidade", "fabrica",
    "unidade", "planta", "refinaria", "mineracao", "minerio", "petroleo",
    "celulose", "aco", "energia", "saneamento", "varejo", "fii", "fundo",
    "fundos", "carteira", "carteiras",
)


@functools.lru_cache(maxsize=1)
def _listed_issuer_index():
    path = Path(__file__).resolve().parent.parent / "acoes-listadas-b3.csv"
    by_ticker, by_root = {}, {}
    try:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ticker = (row.get("Ticker") or "").strip().upper()
                if not ticker:
                    continue
                by_ticker[ticker] = (
                    (row.get("Empresa") or "").strip(),
                    (row.get("RazaoSocial") or "").strip(),
                )
                by_root.setdefault(ticker[:4], []).append(ticker.lower())
    except OSError:
        return {}, {}
    return by_ticker, by_root


def _ticker_aliases(ticker):
    clean = (ticker or "").strip().upper()
    if not clean:
        return ()
    by_ticker, by_root = _listed_issuer_index()
    root = clean[:4]
    aliases = {clean.lower()} if len(clean) > 2 else set()
    if clean in by_ticker:
        aliases.update(by_root.get(root, ()))
    for alias in _ISSUER_ALIASES.get(root, ()) if clean in by_ticker or clean in TICKER_TO_COMPANY else ():
        aliases.add(_fold_text(alias))
    company = TICKER_TO_COMPANY.get(clean) or asset_metadata(clean)["company"].split(" (")[0]
    empresa, _ = by_ticker.get(clean, ("", ""))
    for raw_name in (company, empresa):
        folded = _fold_text(raw_name)
        stripped = " ".join(re.sub(r"\b(?:s/a|s\.a\.?|sa|on|pn|pna|pnb|unt)\b", " ", folded).split())
        if clean not in by_ticker:
            stripped = re.sub(r"\b(?:corporation|corp|inc|ltd|reit)\.?$", "", stripped).strip()
        if stripped and stripped not in _GENERIC_ISSUER_WORDS and len(stripped) >= 3:
            aliases.add(stripped)
        words = [word for word in re.findall(r"[a-z0-9]+", stripped)
                 if word not in _GENERIC_ISSUER_WORDS and len(word) >= 4]
        if len(words) == 1:
            aliases.add(words[0])
    return tuple(sorted(aliases, key=len, reverse=True))


def _contains_term(folded_text, term):
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, folded_text) is not None


_VALE_VERBAL_RE = re.compile(
    r"\b(?:nao\s+|ainda\s+|ja\s+|quanto\s+|mais\s+|menos\s+)?vale\s+"
    r"(?:a\s+pena|mais|menos|quanto|comprar|vender|investir|apostar|entrar|manter|segurar|esperar|conferir|lembrar|destacar|ressaltar|notar|citar|mencionar|encarar|r\$|us\$|\d)"
)


def _contains_alias(folded_text, alias):
    text = _VALE_VERBAL_RE.sub(" ", folded_text) if alias == "vale" else folded_text
    return _contains_term(text, alias)


def is_relevant_company_news(ticker, title, summary=""):
    """Keep only articles whose headline or snippet ties to the issuer and its economic activity."""
    clean = (ticker or "").strip().upper()
    if not clean:
        return False
    folded_title = _fold_text(title)
    if not folded_title:
        return False
    folded_summary = _fold_text(summary)
    combined = f"{folded_title} {folded_summary}".strip()
    def ticker_mentioned(text):
        if len(clean) > 2:
            return _contains_term(text, clean.lower())
        return bool(re.search(rf"\${re.escape(clean.lower())}\b|\({re.escape(clean.lower())}\)", text))

    # Reject comparative mentions such as "valendo 14 Petrobras" or "quer um Bradesco".
    for alias in _ticker_aliases(clean):
        if re.search(rf"\b(?:\d+|um|uma|novo|nova)\s+{re.escape(alias)}\b", folded_title):
            if not ticker_mentioned(folded_title):
                return False

    aliases = _ticker_aliases(clean)
    mentioned_in_title = ticker_mentioned(folded_title) or any(_contains_alias(folded_title, alias) for alias in aliases)
    ticker_in_summary = ticker_mentioned(folded_summary)
    if not mentioned_in_title and not ticker_in_summary:
        return False

    has_financial_context = (
        ticker_mentioned(combined)
        or any(_contains_term(combined, term) for term in _FINANCIAL_CONTEXT_TERMS)
    )
    if any(_contains_term(combined, term) for term in _NON_ECONOMIC_HEADLINE_TERMS):
        if not any(_contains_term(folded_title, term) for term in (
            "acao", "acoes", "lucro", "prejuizo", "dividendo", "dividendos",
            "balanco", "resultado", "cotacao", "preco-alvo", "recomendacao", "cvm",
        )):
            return False

    return has_financial_context


def build_news_query(ticker):
    """Search exact tickers and issuer names; unknown tickers stay ticker-only."""
    terms = [f'"{ticker}"']
    company = TICKER_TO_COMPANY.get(ticker) or asset_metadata(ticker)["company"].split(" (")[0]
    if ticker in TREASURY_TITLES:
        return f'"Tesouro Direto" "{company}"'
    if company != ticker:
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


def parse_rss_items(xml_data, now=None, trusted_only=False):
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
        source_node = item.find("source")
        source = source_node.text.strip() if source_node is not None and source_node.text else ""
        source_url = (source_node.attrib.get("url") or "").strip() if source_node is not None else ""
        if source and title.endswith(f" - {source}"):
            title = title[:-len(f" - {source}")]
        if trusted_only and not is_trusted_news_source(source_url=source_url, provider=source, link=link):
            continue
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
                          "published": published, "provider": source, "source_url": source_url,
                          "summary": description[:1200]})

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


_NEGATIONS = {"nao", "nunca", "jamais", "sem"}
# Clause boundaries stop a direction verb from pairing with a topic in another clause.
_CLAUSE_BREAK = r"[.!?;:,()]+|\b(?:e|mas|porem|enquanto|com|apos|ao|quando|pois)\b"
# Topic polarity: the subject matter itself is good or bad news ("lucro" good, "prejuizo" bad).
_TOPIC_POLARITY = {
    "lucro": 1, "lucros": 1, "receita": 1, "receitas": 1, "dividendo": 1, "dividendos": 1,
    "jcp": 1, "ebitda": 1, "ganho": 1, "ganhos": 1, "faturamento": 1,
    "prejuizo": -1, "prejuizos": -1, "divida": -1, "dividas": -1, "provisao": -1, "provisoes": -1,
    "custo": -1, "custos": -1, "despesa": -1, "despesas": -1, "inadimplencia": -1, "multa": -1,
    "multas": -1, "alavancagem": -1, "defasagem": -1, "risco": -1, "riscos": -1, "crise": -1,
}
# Direction words move a paired topic up or down; alone they describe the stock's own move.
_DIRECTION_POLARITY = {
    **dict.fromkeys((
        "alta", "altas", "sobe", "subir", "subiu", "subiram", "sobem", "cresce", "crescem",
        "cresceu", "crescimento", "crescimentos", "avanca", "avancam", "avancou", "avanco",
        "dispara", "disparam", "disparou", "salta", "saltam", "saltou", "eleva", "elevam", "elevou",
        "aumenta", "aumentam", "aumentou", "aumento", "aumentos", "supera", "superam", "superou",
        "superaram", "recupera", "recuperam", "recuperou", "recuperacao", "melhora", "melhorou",
        "valoriza", "valorizam", "valorizou", "valorizacao", "ganhou", "forte", "fortes",
    ), 1),
    **dict.fromkeys((
        "cai", "caem", "caiu", "cair", "queda", "quedas", "recua", "recuam", "recuou", "despenca",
        "despencam", "despencou", "despencaram", "despencando", "tomba", "tombam", "tombou",
        "encolhe", "encolhem", "encolheu", "corta", "cortam", "cortou", "reduz", "reduzem", "reduziu",
        "reducao", "reducoes", "diminui", "diminuem", "diminuiu", "perda", "perdas", "perdeu",
        "fraco", "fraca", "fracos", "fracas",
    ), -1),
}
# Terms that carry their own polarity and never pair with a topic or direction.
_STANDALONE_POLARITY = {
    **dict.fromkeys((
        "acordo", "acordos", "aprova", "aprovou", "aprovado", "aprovados", "aprovacao", "expande",
        "expandiu", "expansao", "lider", "lidera", "lideranca", "positivo", "positiva", "positivos",
        "compra", "compras", "eficiencia", "eficiente", "modernizacao", "parceria", "parcerias",
        "descoberta", "descobertas", "recorde", "recordes", "estabilizou", "estabilidade",
        "confortavel", "descarbonizacao",
    ), 1),
    **dict.fromkeys((
        "greve", "greves", "suspende", "suspendeu", "atraso", "atrasos", "pressao", "pressoes",
        "pressiona", "pressionam", "pressionou", "rebaixado", "rebaixada", "rebaixamento",
        "concorrencia", "sofre", "sofreu", "paralisacao", "pessimista",
    ), -1),
}
# A news item about a currency, index, or commodity says nothing about a stock's own move.
_MARKET_TERMS = {
    "dolar", "euro", "cambio", "juros", "selic", "ibovespa", "ibov", "bolsa", "mercado",
    "petroleo", "minerio", "inflacao", "indice", "indices",
}
_OUTCOME_PHRASES = (
    *((phrase, 1) for phrase in (
        "reduz divida", "reduz dividas", "reduz custo", "reduz custos", "reduz inadimplencia",
        "reducao de divida", "reducao de dividas", "reducao de custos", "reducao de custo",
        "queda da inadimplencia", "queda de inadimplencia", "reduzindo divida", "reduz alavancagem",
        "reduzir divida", "reducao de alavancagem", "renegociacao de dividas",
    )),
    *((phrase, -1) for phrase in (
        "aumento de custos", "aumento de despesas", "alta da inadimplencia", "aumento de divida",
        "aumento de dividas", "aumento da inadimplencia", "recuperacao judicial",
    )),
)
_EXPECTATION_NOUNS = (
    "esperado", "esperada", "esperados", "esperadas", "estimativa", "estimativas", "consenso",
    "previsto", "previstos", "previsao", "previsoes", "expectativa", "expectativas",
)
# Longest phrases first so "abaixo do esperado" is consumed before any single word.
_PHRASES = sorted(
    [(tuple(phrase.split()), "standalone", sign) for phrase, sign in _OUTCOME_PHRASES]
    + [((side, prep, noun), "direction", sign)
       for side, sign in (("abaixo", -1), ("aquem", -1), ("acima", 1))
       for prep in ("do", "da", "de", "dos", "das")
       for noun in _EXPECTATION_NOUNS],
    key=lambda entry: -len(entry[0]),
)


def _word_entry(word):
    if word in _TOPIC_POLARITY:
        return "topic", _TOPIC_POLARITY[word]
    if word in _DIRECTION_POLARITY:
        return "direction", _DIRECTION_POLARITY[word]
    if word in _STANDALONE_POLARITY:
        return "standalone", _STANDALONE_POLARITY[word]
    return None, 0


def _negated(words, index):
    """A negation within three words flips the term, unless another term sits between them."""
    for previous in reversed(words[max(0, index - 3):index]):
        if previous in _NEGATIONS:
            return True
        if _word_entry(previous)[0]:
            return False
    return False


def _clause_units(words, market):
    """Return (position, polarity, label) units for one clause, in text order."""
    elements = []
    index = 0
    while index < len(words):
        phrase = next((entry for entry in _PHRASES
                       if tuple(words[index:index + len(entry[0])]) == entry[0]), None)
        if phrase:
            tokens, kind, sign = phrase
            step, label = len(tokens), " ".join(tokens)
        else:
            step, label = 1, words[index]
            kind, sign = _word_entry(label)
            if kind is None:
                index += 1
                continue
        elements.append((index, kind, -sign if _negated(words, index) else sign, label))
        index += step

    # Each direction pairs with its nearest unpaired topic; the pair scores topic x direction.
    topics = [element for element in elements if element[1] == "topic"]
    paired, units = set(), []
    for direction in (element for element in elements if element[1] == "direction"):
        free = [topic for topic in topics if topic[0] not in paired]
        if free:
            topic = min(free, key=lambda element: abs(element[0] - direction[0]))
            paired.update((topic[0], direction[0]))
            first, second = sorted((topic, direction))
            units.append((first[0], topic[2] * direction[2], f"{first[3]} {second[3]}"))
    for position, kind, polarity, label in elements:
        if position in paired or (kind == "direction" and market):
            continue
        units.append((position, polarity, label))
    return sorted(units)


def analise_sentimento_pln(title, summary):
    """Estimate Portuguese financial-news polarity from topics, direction words, and standalone terms."""
    # The title and summary are separate sentences, so they never pair across the boundary.
    text = unicodedata.normalize("NFD", f"{title}. {summary}".lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    matched_positive, matched_negative = [], []
    word_count = len(re.findall(r"[a-z0-9_]+", text))
    for clause in re.split(_CLAUSE_BREAK, text):
        words = re.findall(r"[a-z0-9_]+", clause)
        market = any(word in _MARKET_TERMS for word in words)
        for _, polarity, label in _clause_units(words, market):
            (matched_positive if polarity > 0 else matched_negative).append(label)

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


def has_scoring_evidence(article):
    """Lexicon items with fewer than two evidence terms stay visible but never move the aggregate.

    A single term would otherwise set the 0-100 note alone, because the per-ticker mean
    normalizes away any fractional weight given to the item.
    """
    return article.get("intensity") != "Evidência limitada"


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
    scored = {ticker: [item for item in items if has_scoring_evidence(item)] for ticker, items in by_ticker.items()}
    covered_weight = sum(float(weights[t]) for t, usable in scored.items() if usable)
    rows = []
    for ticker, items in by_ticker.items():
        usable = scored[ticker]
        mean = sum(item["score"] for item in usable) / len(usable) if usable else None
        rows.append({
            "ticker": ticker,
            "state": "Feed indisponível" if not feed_status.get(ticker) else
                     "Sem notícias nos últimos 7 dias" if not items else
                     "Evidência limitada" if not usable else "Com notícias",
            "count": len(items),
            "positive": sum(item["sentiment"] == "Otimista" for item in items),
            "neutral": sum(item["sentiment"] == "Neutro" for item in items),
            "negative": sum(item["sentiment"] == "Pessimista" for item in items),
            "mean": mean,
            "covered_share": float(weights[ticker]) / covered_weight if covered_weight and usable else None,
            "contribution": 50 * float(weights[ticker]) / covered_weight * mean if covered_weight and usable else None,
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
