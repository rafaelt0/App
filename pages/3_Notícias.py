import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import re
import unicodedata
import urllib.request
import urllib.parse
from html import escape
import logging

logger = logging.getLogger(__name__)
from utils.charts import apply_plotly_theme
from utils import db as _db
from utils.identity import get_browser_uid
from utils.news import aggregate_ticker_sentiment, parse_rss_items

from utils.ui import (
    empty_state_card,
    load_css,
    loading_overlay,
    render_page_header,
    svg_icon,
)

# CSS customizado
load_css()


# ─── SVG Icon Library ─────────────────────────────────────────────────────────
_svg = svg_icon

ICO_OK      = _svg('<circle cx="12" cy="12" r="9" stroke="#00ff87" stroke-width="1.8"/>'
                   '<path d="M8 12l3 3 5-5" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
ICO_WARN    = _svg('<path d="M12 3L22 21H2L12 3Z" stroke="#ffd600" stroke-width="1.8" stroke-linejoin="round"/>'
                   '<line x1="12" y1="10" x2="12" y2="14" stroke="#ffd600" stroke-width="2" stroke-linecap="round"/>'
                   '<circle cx="12" cy="17.5" r="1" fill="#ffd600"/>')
ICO_CRIT    = _svg('<circle cx="12" cy="12" r="9" stroke="#ff3d5a" stroke-width="1.8"/>'
                   '<line x1="9" y1="9" x2="15" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>'
                   '<line x1="15" y1="9" x2="9" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>')
ICO_NEWS    = _svg('<rect x="3" y="4" width="18" height="16" rx="2" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<line x1="7" y1="8" x2="17" y2="8" stroke="#00d2ff" stroke-width="1.5" stroke-linecap="round"/>'
                   '<line x1="7" y1="12" x2="13" y2="12" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>'
                   '<line x1="7" y1="16" x2="15" y2="16" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>', 16)
ICO_CHART   = _svg('<rect x="3" y="12" width="3" height="9" rx="1" fill="#00ff87"/>'
                   '<rect x="9" y="7"  width="3" height="14" rx="1" fill="#00d2ff"/>'
                   '<rect x="15" y="9" width="3" height="12" rx="1" fill="#ffd600"/>', 16)
ICO_TARGET  = _svg('<circle cx="12" cy="12" r="9" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<circle cx="12" cy="12" r="5" stroke="#ffd600" stroke-width="1.5"/>'
                   '<circle cx="12" cy="12" r="2" fill="#00ff87"/>', 16)
ICO_IDEA    = _svg('<circle cx="12" cy="10" r="6" stroke="#ffd600" stroke-width="1.8"/>'
                   '<path d="M9 16.5h6M10 19h4" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="12" y1="4" x2="12" y2="2" stroke="#ffd600" stroke-width="1.5" stroke-linecap="round"/>', 16)
ICO_CPU     = _svg('<rect x="4" y="4" width="16" height="16" rx="2" stroke="#00ff87" stroke-width="1.8"/>'
                   '<path d="M9 9h6v6H9z" fill="#00ff87" opacity="0.3"/>'
                   '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3" stroke="#00ff87" stroke-width="1.5"/>', 14)
ICO_LEXICON = _svg('<path d="M4 5.5A2 2 0 0 1 6 3.5h13V19H6a2 2 0 0 0-2 2z" stroke="#00d2ff" stroke-width="1.7" stroke-linejoin="round"/>'
                   '<path d="M4 19.5A2 2 0 0 1 6 17.5h13" stroke="#00d2ff" stroke-width="1.7" stroke-linejoin="round"/>', 14)

def section_header(icon_svg, text, tag="h2"):
    st.markdown(
        f'<{tag} class="ui-section-heading">{icon_svg}<span>{text}</span></{tag}>',
        unsafe_allow_html=True,
    )

def diag_row(icon_svg, text, color):
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;gap:8px;padding:4px 0;'
        f'color:{color};font-size:0.88rem;line-height:1.4;">'
        f'<div style="flex-shrink:0;margin-top:2px;display:flex;align-items:center;">{icon_svg}</div>'
        f'<div style="flex-grow:1;">{text}</div>'
        f'</div>',
        unsafe_allow_html=True)

def get_diag_row_html(icon_svg, text, color):
    return (
        f'<div style="display:flex;align-items:flex-start;gap:8px;padding:6px 0;'
        f'color:{color};font-size:0.88rem;line-height:1.4;">'
        f'<div style="flex-shrink:0;margin-top:2px;display:flex;align-items:center;">{icon_svg}</div>'
        f'<div style="flex-grow:1;">{text}</div>'
        f'</div>'
    )

# ── Page header ───────────────────────────────────────────────────────────────
render_page_header(
    "Notícias do portfólio",
    "Acompanhe eventos recentes e o tom das notícias sobre seus ativos.",
    "news",
)


# ─── ALGORITMO PLN DE SENTIMENTO (Lexicon-Based Fallback) ────────────────────────
def analise_sentimento_pln(title, summary):
    # Converte para minúsculas
    text = (title + " " + summary).lower()
    
    # Normalização de acentos para robustez do matching
    def remover_acentos(txt):
        return "".join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')
    
    text_normalized = remover_acentos(text)
    
    # Lexicon Financeiro em Português
    pos_words = [
        'alta', 'lucro', 'lucros', 'recorde', 'recordes', 'crescimento', 'crescimentos', 'descoberta', 
        'descobertas', 'acordo', 'acordos', 'aprova', 'aprovou', 'aprovado', 'aprovados', 'aprovacao', 
        'expande', 'expandiu', 'expansao', 'liderança', 'lider', 'lidera', 'recuperacao', 'recuperou', 
        'forte', 'fortes', 'positivo', 'positiva', 'positivos', 'ganho', 'ganhos', 'ganhou', 'compra', 
        'compras', 'dividendos', 'dividendo', 'jcp', 'eficiencia', 'eficiente', 'modernizacao', 
        'parceria', 'parcerias', 'valorizacao', 'valorizou', 'descarbonizacao', 'melhora', 'melhorou', 
        'subiu', 'subiram', 'superou', 'superaram', 'estabilizou', 'estabilidade', 'confortavel'
    ]
    
    neg_words = [
        'queda', 'quedas', 'prejuizo', 'prejuizos', 'greve', 'greves', 'suspende', 'suspendeu', 
        'inadimplencia', 'atraso', 'atrasos', 'pressao', 'pressoes', 'perda', 'perdas', 'reducao', 
        'rebaixado', 'rebaixada', 'rebaixamento', 'concorrencia', 'fraca', 'fraco', 'sofre', 'sofreu', 
        'paralisacao', 'divida', 'dividas', 'pessimista', 'crise', 'risco', 'riscos', 'cair', 'caiu', 
        'recua', 'recuou', 'defasagem', 'alavancagem'
    ]
    
    # Compostos Semânticos (Modificadores)
    pos_compounds = [
        'reduz divida', 'reduz dividas', 'reduz custo', 'reduz custos', 
        'reduz inadimplencia', 'reducao de divida', 'reducao de dividas', 
        'reducao de custos', 'reducao de custo', 'queda da inadimplencia',
        'queda de inadimplencia', 'reduzindo divida', 'reduz alavancagem',
        'reduzir divida', 'reducao de alavancagem', 'renegociacao de dividas'
    ]
    
    neg_compounds = [
        'aumento de custos', 'aumento de despesas', 'alta da inadimplencia', 
        'aumento de divida', 'aumento de dividas', 'aumento da inadimplencia'
    ]
    
    matched_pos = []
    matched_neg = []
    
    # 1. Verifica compostos primeiro e limpa do texto para evitar contagem dupla
    for comp in pos_compounds:
        if comp in text_normalized:
            matched_pos.append(comp)
            text_normalized = text_normalized.replace(comp, "")
            
    for comp in neg_compounds:
        if comp in text_normalized:
            matched_neg.append(comp)
            text_normalized = text_normalized.replace(comp, "")
            
    # 2. Tokenização simples com regex
    words = re.findall(r'\b\w+\b', text_normalized)
    for w in words:
        if w in pos_words:
            matched_pos.append(w)
        elif w in neg_words:
            matched_neg.append(w)
            
    # 3. Cálculo matemático do score
    num_pos = len(matched_pos)
    num_neg = len(matched_neg)
    
    if num_pos + num_neg > 0:
        score = (num_pos - num_neg) / (num_pos + num_neg)
    else:
        score = 0.0
        
    # Classificação por limiar
    if score >= 0.20:
        sentiment = "Otimista"
    elif score <= -0.20:
        sentiment = "Pessimista"
    else:
        sentiment = "Neutro"
        
    return {
        "sentiment": sentiment,
        "score": round(score, 2),
        "pos_terms": matched_pos,
        "neg_terms": matched_neg,
        "raw_text_length": len(words)
    }

_INTENSITY_LEVELS = ("Alto", "Médio-Alto", "Médio", "Baixo-Médio", "Baixo")


def _intensidade_sentimento(score):
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

# ─── REAL-TIME NEWS RSS FETCHING ──────────────────────────────────────────────
ticker_to_name = {
    'PETR3': 'Petrobras', 'PETR4': 'Petrobras',
    'VALE3': 'Vale',
    'ITUB3': 'Itaú', 'ITUB4': 'Itaú',
    'BBDC3': 'Bradesco', 'BBDC4': 'Bradesco',
    'BBAS3': 'Banco do Brasil',
    'WEGE3': 'Weg',
    'MGLU3': 'Magazine Luiza',
    'ABEV3': 'Ambev',
    'ELET3': 'Eletrobras', 'ELET6': 'Eletrobras',
    'RENT3': 'Localiza',
    'LREN3': 'Lojas Renner',
    'PRIO3': 'PetroRio',
    'HAPV3': 'Hapvida',
    'SANB11': 'Santander',
    'VVAR3': 'Via Varejo', 'BHIA3': 'Casas Bahia',
    'GGBR4': 'Gerdau',
    'ITSA4': 'Itaúsa',
    'SUZB3': 'Suzano',
    'JBSS3': 'JBS',
    'UGPA3': 'Ultrapar',
    'RADL3': 'RaiaDrogasil',
    'EQTL3': 'Equatorial',
    'CSAN3': 'Cosan',
    'CPFE3': 'CPFL Energia',
    'SBSP3': 'Sabesp',
    'TAEE11': 'Taesa',
    'KLBN11': 'Klabin'
}

NEWS_REQUEST_TIMEOUT_SECONDS = 8

@st.cache_data(ttl=600, show_spinner=False)
def get_brazilian_news(ticker_name):
    name = ticker_to_name.get(ticker_name, "")
    query = f"{ticker_name} OR \"{name}\"" if name else ticker_name
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=NEWS_REQUEST_TIMEOUT_SECONDS) as response:
            xml_data = response.read()
        return {"ok": True, "items": parse_rss_items(xml_data)}
    except Exception as exc:
        logger.warning("news RSS fetch/parse failed: %s", exc)
        logger.debug("news RSS failure details", exc_info=True)
        return {"ok": False, "items": []}


# ─── DEEP LEARNING MODEL LOAD (FinBERT-PT-BR) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_finbert_pipeline():
    try:
        from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
        model_name = "lucas-leme/FinBERT-PT-BR"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        nlp = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
        return nlp
    except ModuleNotFoundError as exc:
        logger.info("FinBERT opcional indisponível; usando PLN léxico: %s", exc)
        return None
    except Exception:
        logger.warning("FinBERT pipeline load failed", exc_info=True)
        return None



def analise_sentimento_finbert(title, summary, nlp):
    text = title
    if summary:
        text += " " + summary
        
    if nlp is None:
        # Fallback to the lexicon-based model
        res_pln = analise_sentimento_pln(title, summary)
        scores = [
            {'label': 'POSITIVE', 'score': max(0.0, res_pln['score']) if res_pln['sentiment'] == 'Otimista' else 0.0},
            {'label': 'NEGATIVE', 'score': abs(res_pln['score']) if res_pln['sentiment'] == 'Pessimista' else 0.0},
            {'label': 'NEUTRAL', 'score': 1.0 if res_pln['sentiment'] == 'Neutro' else 0.0}
        ]
        return {
            "sentiment": res_pln["sentiment"],
            "score": res_pln["score"],
            "scores": scores,
            "is_finbert": False,
            "engine": "PLN Léxico (fallback)",
            "pos_terms": res_pln["pos_terms"],
            "neg_terms": res_pln["neg_terms"],
            "raw_text_length": res_pln["raw_text_length"]
        }
        
    try:
        # FinBERT prediction
        res = nlp([text])[0]
        # Map label scores
        score_dict = {item['label']: item['score'] for item in res}
        pos_score = score_dict.get('POSITIVE', 0.0)
        neg_score = score_dict.get('NEGATIVE', 0.0)
        
        score = pos_score - neg_score
        sentiment = "Otimista" if score >= 0.20 else "Pessimista" if score <= -0.20 else "Neutro"
            
        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "scores": res,
            "is_finbert": True,
            "engine": "FinBERT-PT-BR",
            "raw_text_length": len(text.split())
        }
    except Exception:
        # Fallback on error
        logger.warning("FinBERT sentiment inference failed, falling back to rule-based", exc_info=True)
        res_pln = analise_sentimento_pln(title, summary)
        scores = [
            {'label': 'POSITIVE', 'score': max(0.0, res_pln['score']) if res_pln['sentiment'] == 'Otimista' else 0.0},
            {'label': 'NEGATIVE', 'score': abs(res_pln['score']) if res_pln['sentiment'] == 'Pessimista' else 0.0},
            {'label': 'NEUTRAL', 'score': 1.0 if res_pln['sentiment'] == 'Neutro' else 0.0}
        ]
        return {
            "sentiment": res_pln["sentiment"],
            "score": res_pln["score"],
            "scores": scores,
            "is_finbert": False,
            "engine": "PLN Léxico (fallback)",
            "pos_terms": res_pln["pos_terms"],
            "neg_terms": res_pln["neg_terms"],
            "raw_text_length": res_pln["raw_text_length"]
        }

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_sentiment(title, summary, model_key, _nlp):
    return analise_sentimento_finbert(title, summary, _nlp)


def _restore_saved_portfolio_context() -> None:
    """Restore the last analyzed portfolio after a full-page handoff."""
    if "selected_tickers" in st.session_state:
        return

    saved_tickers, saved_weights = _db.portfolio_get(get_browser_uid())
    tickers = [str(ticker).replace(".SA", "") for ticker in saved_tickers]
    if not tickers:
        return

    normalized_weights = {}
    for raw_ticker, raw_weight in (saved_weights or {}).items():
        ticker = str(raw_ticker).replace(".SA", "").strip().upper()
        try:
            parsed_weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if ticker and math.isfinite(parsed_weight) and parsed_weight >= 0:
            normalized_weights[ticker] = parsed_weight

    raw_weights = {
        ticker: normalized_weights.get(ticker, 0.0) for ticker in tickers
    }
    total_weight = sum(raw_weights.values())
    if total_weight > 0:
        weights = {
            ticker: raw_weights[ticker] / total_weight for ticker in tickers
        }
    else:
        default_weight = 1.0 / len(tickers)
        weights = {ticker: default_weight for ticker in tickers}
    st.session_state["selected_tickers"] = tickers
    st.session_state["portfolio_loaded_tickers"] = tickers
    st.session_state["portfolio_analysis_tickers"] = tickers
    st.session_state["portfolio_loaded"] = True
    st.session_state["peso_manual_df"] = pd.DataFrame(
        {"Peso": [weights[ticker] for ticker in tickers]},
        index=tickers,
    )


_restore_saved_portfolio_context()



# ─── CORE STREAMLIT PAGE LOGIC ───────────────────────────────────────────────

# ─── Validação do portfólio antes de carregar o modelo NLP ───────────────────
_current_tickers = list(st.session_state.get("selected_tickers", []))
_loaded_tickers = list(st.session_state.get("portfolio_loaded_tickers", []))
_analyzed_tickers = list(st.session_state.get("portfolio_analysis_tickers", []))
_has_portfolio_state = (
    "peso_manual_df" in st.session_state
    and st.session_state["peso_manual_df"] is not None
)
_portfolio_ready = (
    _has_portfolio_state
    and bool(st.session_state.get("portfolio_loaded"))
    and bool(_current_tickers)
    and _current_tickers == _loaded_tickers == _analyzed_tickers
)
if not _portfolio_ready:
    _selection_changed = bool(_current_tickers) and (
        _current_tickers != _loaded_tickers
        or _current_tickers != _analyzed_tickers
    )
    if _selection_changed:
        _empty_title = "Atualize o portfólio"
        _empty_message = (
            "A seleção de ativos mudou desde a última análise. "
            "Volte para <strong style=\"color:#61d4c6\">Portfolio</strong> e clique em "
            "<strong>Carregar portfólio</strong> antes de consultar as notícias."
        )
    else:
        _empty_title = "Portfólio não configurado"
        _empty_message = (
            "Configure seu portfólio na página <strong style=\"color:#61d4c6\">Portfolio</strong> "
            "e carregue a análise para filtrar as notícias dos seus ativos."
        )
    empty_state_card(
        icon_svg="""<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" style="opacity:0.4;margin-bottom:1rem">
            <path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v10a2 2 0 01-2 2z" stroke="#94a3b8" stroke-width="1.5"/>
            <path d="M14 4v4h4" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="7" y1="13" x2="17" y2="13" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
            <line x1="7" y1="17" x2="17" y2="17" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
        </svg>""",
        title=_empty_title,
        message=_empty_message,
        cta_label="Ir para Portfolio",
        cta_page="pages/1_Portfolio.py",
    )
    st.stop()

if st.button(
    "Atualizar notícias",
    key="news_refresh",
    use_container_width=True,
    help="Limpa o cache de 10 minutos e busca novas notícias para os ativos da carteira.",
):
    get_brazilian_news.clear()
    st.session_state["noticias_page"] = 1
    st.session_state.pop("_noticias_filter_key", None)
    st.rerun()

# O modelo pesado só é carregado quando há uma carteira válida para analisar.
if "finbert_nlp" not in st.session_state:
    with loading_overlay("Carregando modelo de IA (FinBERT-PT-BR)…"):
        st.session_state["finbert_nlp"] = load_finbert_pipeline()
finbert_nlp = st.session_state["finbert_nlp"]

# Recupera ativos e pesos
peso_df = st.session_state["peso_manual_df"]
tickers = [t.replace(".SA", "") for t in peso_df.index]
pesos = {t.replace(".SA", ""): row.iloc[0] for t, row in peso_df.iterrows()}

# Informar o estado do algoritmo NLP na barra lateral
if finbert_nlp is not None:
    st.sidebar.markdown(
        f'<div style="margin:0.5rem 0;padding:0.6rem 0.85rem;background:rgba(0,255,135,0.06);'
        f'border:1px solid rgba(0,255,135,0.3);border-radius:8px;display:flex;align-items:center;gap:8px;">'
        f'{ICO_CPU}<span style="font-size:0.82rem;color:#b0ffe0;font-weight:600">FinBERT-PT-BR Ativo</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        f'<div style="margin:0.5rem 0;padding:0.6rem 0.85rem;background:rgba(0,210,255,0.06);'
        f'border:1px solid rgba(0,210,255,0.3);border-radius:8px;display:flex;align-items:center;gap:8px;">'
        f'{ICO_LEXICON}<span style="font-size:0.82rem;color:#b8eeff;font-weight:600">PLN Léxico Ativo (Fallback)</span></div>',
        unsafe_allow_html=True,
    )

# Build the feed exclusively from recent, parseable RSS entries.
news_items = []
feed_status = {}
with loading_overlay("Buscando notícias e processando sentimento NLP…", tickers=tickers):
    for t in tickers:
        result = get_brazilian_news(t)
        feed_status[t] = result["ok"]
        for item in result["items"]:
            sentiment_res = _cached_sentiment(
                item["title"], item.get("summary", ""),
                "lucas-leme/FinBERT-PT-BR:v1" if finbert_nlp is not None else "lexicon:v1",
                finbert_nlp,
            )
            news_items.append({
                "ticker": t, "title": item["title"],
                "summary": item["summary"] or "Clique no link do título para ler os detalhes da notícia na fonte oficial.",
                "sentiment": sentiment_res["sentiment"], "score": sentiment_res["score"],
                "scores": sentiment_res.get("scores", []), "is_finbert": sentiment_res.get("is_finbert", False),
                "engine": sentiment_res["engine"], "pos_terms": sentiment_res.get("pos_terms", []),
                "neg_terms": sentiment_res.get("neg_terms", []), "raw_text_length": sentiment_res.get("raw_text_length", 0),
                "provider": item["provider"], "intensity": _intensidade_sentimento(sentiment_res["score"]),
                "pub_time": item["date"], "published": item["published"], "link": item["link"], "peso": pesos[t],
            })
for ticker, succeeded in feed_status.items():
    if not succeeded:
        st.warning(f"Feed de notícias indisponível para {ticker}; nenhum item foi usado na análise.")
    elif not any(item["ticker"] == ticker for item in news_items):
        st.info(f"Nenhuma notícia recente (últimos 7 dias) encontrada para {ticker}.")

live_news_items = news_items

# Cálculos de sentimentos consolidados baseados nas métricas dinâmicas do NLP
pos_count = sum(item["sentiment"] == "Otimista" for item in live_news_items)
neg_count = sum(item["sentiment"] == "Pessimista" for item in live_news_items)
neu_count = sum(item["sentiment"] == "Neutro" for item in live_news_items)
avg_score, news_coverage = aggregate_ticker_sentiment(live_news_items, pesos)

# Normaliza score global de -1 a +1 para 0 a 100
if news_coverage > 0:
    normalized_score = int((avg_score + 1.0) / 2.0 * 100)
    sentiment_label = (
        "FORTEMENTE OTIMISTA"
        if normalized_score >= 80
        else "OTIMISTA"
        if normalized_score >= 60
        else "NEUTRO / EQUILIBRADO"
        if normalized_score >= 40
        else "PREOCUPANTE"
        if normalized_score >= 20
        else "CRÍTICO"
    )
    score_color = (
        "#00ff87"
        if normalized_score >= 60
        else "#ffd600"
        if normalized_score >= 40
        else "#ff3d5a"
    )
else:
    avg_score = 0.0
    normalized_score = 50
    sentiment_label = "SEM DADOS RECENTES"
    score_color = "#64748b"

score_display = str(normalized_score) if news_coverage > 0 else "—"

# Exibição do painel principal
col_g1, col_g2 = st.columns([1, 2])

with col_g1:
    # Card do Score de Sentimento
    nlp_engine_label = "FinBERT-PT-BR" if finbert_nlp is not None else "PLN Léxico"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0e1b2f, #080c14); 
                border: 2px solid {score_color}; 
                border-radius: 16px; 
                padding: 1.8rem 1.5rem; 
                text-align: center; 
                box-shadow: 0 0 20px {score_color}1a;
                margin-bottom: 1.5rem;">
        <div style="font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase;">Sentimento das notícias disponíveis ({nlp_engine_label})</div>
        <div style="font-size: 3.5rem; font-weight: 900; color: {score_color}; font-family: 'JetBrains Mono', monospace; margin: 0.5rem 0;">
            {score_display}<span style="font-size: 1.5rem; font-weight: 500; color: #94a3b8;">/100</span>
        </div>
        <div style="font-size: 0.85rem; font-weight: 700; color: {score_color}; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.8rem;">
            {sentiment_label}
        </div>
        <div style="display: flex; justify-content: space-around; border-top: 1px solid #1e293b; padding-top: 0.8rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
            <div>
                <span style="color: #4ade80; font-weight: 700;">{pos_count}</span>
                <div style="color: #94a3b8; font-size: 0.65rem;">Positivas</div>
            </div>
            <div>
                <span style="color: #60a5fa; font-weight: 700;">{neu_count}</span>
                <div style="color: #94a3b8; font-size: 0.65rem;">Neutras</div>
            </div>
            <div>
                <span style="color: #f87171; font-weight: 700;">{neg_count}</span>
                <div style="color: #94a3b8; font-size: 0.65rem;">Negativas</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_g2:
    # Gráfico de Distribuição do Sentimento por Ativo
    asset_sentiments = []
    for t in tickers:
        t_items = [x for x in live_news_items if x["ticker"] == t]
        pos = sum(1 for x in t_items if x["sentiment"] == "Otimista")
        neu = sum(1 for x in t_items if x["sentiment"] == "Neutro")
        neg = sum(1 for x in t_items if x["sentiment"] == "Pessimista")
        asset_sentiments.append({"Ativo": t, "Otimista": pos, "Neutro": neu, "Pessimista": neg})
        
    df_sent = pd.DataFrame(asset_sentiments)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Pessimista', y=df_sent['Ativo'], x=df_sent['Pessimista'], 
        orientation='h', marker=dict(color='#f87171')
    ))
    fig.add_trace(go.Bar(
        name='Neutro', y=df_sent['Ativo'], x=df_sent['Neutro'], 
        orientation='h', marker=dict(color='#60a5fa')
    ))
    fig.add_trace(go.Bar(
        name='Otimista', y=df_sent['Ativo'], x=df_sent['Otimista'], 
        orientation='h', marker=dict(color='#4ade80')
    ))
    
    fig.update_layout(
        barmode='stack',
        title="Volume de Notícias e Distribuição por Ativo",
        xaxis=dict(title="", dtick=1),
        yaxis=dict(title="Ativos"),
        height=200 + len(tickers) * 35,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    apply_plotly_theme(fig)
    fig.update_layout(
        title=dict(text="Notícias por ativo", x=0, xanchor="left"),
        height=max(280, 190 + len(tickers) * 40),
        margin=dict(l=48, r=12, t=46, b=96),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="left",
            x=0,
        ),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
    )

st.markdown("---")


# ─── NEWS FEED LISTING ───────────────────────────────────────────────────────

# Filtro lateral/superior de notícias
section_header(ICO_NEWS, "Feed de Notícias da Carteira", "h2")
st.caption(
    "O sentimento resume notícias recentes disponíveis, agregadas primeiro por ativo; "
    "não é previsão de retorno. Os filtros abaixo afetam apenas o feed."
)


col_filter, col_sort, col_sentiment, col_intensity = st.columns([1, 1, 1, 1])
with col_filter:
    selected_ticker = st.selectbox(
        "Filtrar por ativo",
        ["Todos os Ativos"] + [f"{t} ({sum(1 for x in news_items if x['ticker']==t)} notícias)" for t in tickers],
    )
    # Normaliza a seleção (remove o sufixo de contagem)
    selected_ticker_clean = selected_ticker.split(" (")[0] if selected_ticker != "Todos os Ativos" else "Todos os Ativos"
with col_sort:
    sort_mode = st.selectbox(
        "Ordenar por",
        ["Mais recentes", "Maior intensidade", "Mais otimistas", "Mais pessimistas"]
    )
with col_sentiment:
    sentiment_filter = st.selectbox(
        "Filtrar por sentimento",
        ["Todos", "Otimistas", "Neutras", "Pessimistas"],
        help="Mostra apenas notícias classificadas pelo PLN com o sentimento escolhido.",
    )
    intensity_filter = st.selectbox(
        "Filtrar por intensidade do sentimento",
        ["Todos", *_INTENSITY_LEVELS],
        help="Faixas de intensidade baseadas apenas no score de sentimento; não representam materialidade financeira.",
    )


filtered_news = news_items if selected_ticker_clean == "Todos os Ativos" else [x for x in news_items if x["ticker"] == selected_ticker_clean]
if sentiment_filter != "Todos":
    _sentiment_value = {
        "Otimistas": "Otimista",
        "Neutras": "Neutro",
        "Pessimistas": "Pessimista",
    }[sentiment_filter]
    filtered_news = [x for x in filtered_news if x["sentiment"] == _sentiment_value]
if intensity_filter != "Todos":
    filtered_news = [
        x for x in filtered_news if x["intensity"] == intensity_filter
    ]

# RSS dates are parsed timezone-aware UTC datetimes; newest first.
if sort_mode == "Mais recentes":
    filtered_news = sorted(filtered_news, key=lambda x: x["published"], reverse=True)
elif sort_mode == "Maior intensidade":
    intensity_rank = {level: rank for rank, level in enumerate(_INTENSITY_LEVELS)}
    filtered_news = sorted(filtered_news, key=lambda x: intensity_rank.get(x["intensity"], 5))
elif sort_mode == "Mais otimistas":
    filtered_news = sorted(filtered_news, key=lambda x: -x["score"])
elif sort_mode == "Mais pessimistas":
    filtered_news = sorted(filtered_news, key=lambda x: x["score"])

total_news = len(filtered_news)
pos_f = sum(1 for x in filtered_news if x["sentiment"] == "Otimista")
neg_f = sum(1 for x in filtered_news if x["sentiment"] == "Pessimista")
neu_f = total_news - pos_f - neg_f
st.caption(
    f"Exibindo {total_news} notícias recentes · {pos_f} otimistas · "
    f"{neu_f} neutras · {neg_f} pessimistas"
)
st.metric("Cobertura de notícias da carteira", f"{news_coverage:.1%}",
          help="Soma do peso dos ativos com pelo menos uma notícia recente; ativos sem notícias não são tratados como neutros.")

if not filtered_news:
    st.info(
        "Nenhuma notícia corresponde aos filtros atuais. "
        "Tente outro filtro ou aguarde notícias recentes para os ativos selecionados."
    )


# Paginação
ITEMS_PER_PAGE = 15
page_key = f"{selected_ticker}_{sort_mode}_{sentiment_filter}_{intensity_filter}"
if st.session_state.get("_noticias_filter_key") != page_key:
    st.session_state["noticias_page"] = 1
    st.session_state["_noticias_filter_key"] = page_key
if "noticias_page" not in st.session_state:
    st.session_state["noticias_page"] = 1

n_show = st.session_state["noticias_page"] * ITEMS_PER_PAGE
news_to_show = filtered_news[:n_show]

for news in news_to_show:
    badge_bg = "rgba(74, 222, 128, 0.1)" if news["sentiment"] == "Otimista" else \
               "rgba(248, 113, 113, 0.1)" if news["sentiment"] == "Pessimista" else "rgba(96, 165, 250, 0.1)"
    badge_color = "#4ade80" if news["sentiment"] == "Otimista" else \
                  "#f87171" if news["sentiment"] == "Pessimista" else "#60a5fa"
                  
    intensity_color = "#4ade80" if news["intensity"] == "Baixo" else \
                      "#ffd600" if "Médio" in news["intensity"] else "#ff3d5a"

    # Escape external feed content before embedding it in custom HTML.
    _news_ticker = escape(str(news["ticker"]))
    _news_provider = escape(str(news["provider"]))
    _news_pub_time = escape(str(news["pub_time"]))
    _news_sentiment = escape(str(news["sentiment"]).upper())
    _news_intensity = escape(str(news["intensity"]).upper())
    _news_engine = escape(str(news["engine"]))
    _news_title = escape(str(news["title"]))
    _news_summary = escape(str(news["summary"]))
    _raw_link = str(news.get("link", "")).strip()
    _parsed_link = urllib.parse.urlparse(_raw_link)
    _safe_link = (
        escape(_raw_link, quote=True)
        if _parsed_link.scheme in {"http", "https"}
        else None
    )
    title_html = (
        f'<a class="news-title-link" href="{_safe_link}" target="_blank" '
        f'rel="noopener noreferrer">{_news_title}</a>'
        if _safe_link
        else _news_title
    )

    # News Card Container
    st.markdown(f"""
    <div style="background-color: var(--panel-bg); 
                border: 1px solid var(--border-color); 
                border-radius: 12px; 
                padding: 1.2rem; 
                margin-bottom: 0.5rem; 
                box-shadow: var(--shadow-dark);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="background: rgba(0, 210, 255, 0.1); color: var(--secondary-color); border: 1px solid rgba(0, 210, 255, 0.25); border-radius: 4px; padding: 0.1rem 0.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700;">
                    {_news_ticker}
                </span>
                <span style="color: var(--text-muted); font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;">
                    {_news_provider} • {_news_pub_time} • {_news_engine}
                </span>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}40; border-radius: 10rem; padding: 0.15rem 0.5rem; font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem; font-weight: 700;">
                    {_news_sentiment}
                </span>
                <span style="font-size: 0.7rem; color: #94a3b8; font-weight: 600;">
                    INTENSIDADE DO SENTIMENTO: <span style="color: {intensity_color}; font-weight: 800;">{_news_intensity}</span>
                </span>
            </div>
        </div>
        <h4 style="margin: 0.3rem 0 0.5rem 0 !important; font-size: 1rem !important; font-weight: 600; line-height: 1.4; color: var(--text-main);">
            {title_html}
        </h4>
        <p style="margin: 0; color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin-bottom: 0.6rem;">
            {_news_summary}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # NLP Explainer Expander
    if news.get("is_finbert", False):
        with st.expander(f"🧠 Detalhes do Algoritmo PLN (FinBERT-PT-BR - Score: {news['score']})"):
            scores_list = news.get("scores", [])
            scores_dict = {item['label']: item['score'] for item in scores_list}
            pos_prob = scores_dict.get('POSITIVE', 0.0) * 100
            neg_prob = scores_dict.get('NEGATIVE', 0.0) * 100
            neu_prob = scores_dict.get('NEUTRAL', 0.0) * 100
            
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b; border-radius: 12px; padding: 1rem; margin-top: 0.2rem;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase;">
                    Distribuição de Probabilidade — FinBERT-PT-BR (LLM)
                </div>
                
                <div style="margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #e2e8f0; margin-bottom: 3px;">
                        <span style="font-weight: 500;">Otimista (POSITIVE)</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #00ff87;">{pos_prob:.1f}%</span>
                    </div>
                    <div style="background: #0f172a; border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid #1e293b;">
                        <div style="background: #00ff87; width: {pos_prob:.1f}%; height: 100%; box-shadow: 0 0 8px #00ff8780;"></div>
                    </div>
                </div>
                
                <div style="margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #e2e8f0; margin-bottom: 3px;">
                        <span style="font-weight: 500;">Pessimista (NEGATIVE)</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #ff3d5a;">{neg_prob:.1f}%</span>
                    </div>
                    <div style="background: #0f172a; border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid #1e293b;">
                        <div style="background: #ff3d5a; width: {neg_prob:.1f}%; height: 100%; box-shadow: 0 0 8px #ff3d5a80;"></div>
                    </div>
                </div>
                
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #e2e8f0; margin-bottom: 3px;">
                        <span style="font-weight: 500;">Neutro (NEUTRAL)</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #ffd600;">{neu_prob:.1f}%</span>
                    </div>
                    <div style="background: #0f172a; border-radius: 4px; height: 6px; overflow: hidden; border: 1px solid #1e293b;">
                        <div style="background: #ffd600; width: {neu_prob:.1f}%; height: 100%; box-shadow: 0 0 8px #ffd60080;"></div>
                    </div>
                </div>
                <p style="font-size:0.7rem; color:#64748b; margin-top:8px; margin-bottom:0;">
                    Classificação baseada em modelo de linguagem BERT pré-treinado em finanças. Tamanho do texto: {news['raw_text_length']} palavras.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.expander(f"📝 Detalhes do Algoritmo PLN (Léxico - Score: {news['score']})"):
            col_exp1, col_exp2 = st.columns([1, 1])
            with col_exp1:
                st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin-bottom:2px; font-weight:600;'> termos positivos encontrados </p>", unsafe_allow_html=True)
                if news["pos_terms"]:
                    pos_html = " ".join([f"<span style='background:rgba(74, 222, 128, 0.15); color:#4ade80; border:1px solid #4ade8040; border-radius:4px; padding:2px 6px; font-size:0.72rem; font-family:\"JetBrains Mono\", monospace;'>{escape(str(w))}</span>" for w in news["pos_terms"]])
                    st.markdown(pos_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='font-size:0.72rem; color:#64748b; font-style:italic;'>Nenhum</span>", unsafe_allow_html=True)
                    
                st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin-top:8px; margin-bottom:2px; font-weight:600;'> termos negativos encontrados </p>", unsafe_allow_html=True)
                if news["neg_terms"]:
                    neg_html = " ".join([f"<span style='background:rgba(248, 113, 113, 0.15); color:#f87171; border:1px solid #f8717140; border-radius:4px; padding:2px 6px; font-size:0.72rem; font-family:\"JetBrains Mono\", monospace;'>{escape(str(w))}</span>" for w in news["neg_terms"]])
                    st.markdown(neg_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='font-size:0.72rem; color:#64748b; font-style:italic;'>Nenhum</span>", unsafe_allow_html=True)
            with col_exp2:
                st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin-bottom:2px; font-weight:600;'> equação do score pln </p>", unsafe_allow_html=True)
                pos_len = len(news["pos_terms"])
                neg_len = len(news["neg_terms"])
                denom = pos_len + neg_len
                denom_str = str(denom) if denom > 0 else "1 (suavizado)"
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b; border-radius: 8px; padding: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
                    Score = (Pos - Neg) / (Pos + Neg)<br>
                    Score = ({pos_len} - {neg_len}) / {denom_str}<br>
                    <b>Score Final = {news['score']}</b>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:0.7rem; color:#64748b; margin-top:4px;'>Tamanho do texto tokenizado: {news['raw_text_length']} palavras.</p>", unsafe_allow_html=True)
                
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

# Botão "Ver mais"
remaining = len(filtered_news) - n_show
if remaining > 0:
    col_vm, _ = st.columns([1, 3])
    with col_vm:
        if st.button(f"Ver mais {min(remaining, ITEMS_PER_PAGE)} notícias  ({remaining} restantes)"):
            st.session_state["noticias_page"] += 1
            st.rerun()
elif len(filtered_news) > ITEMS_PER_PAGE:
    st.caption(f"✓ Todas as {len(filtered_news)} notícias carregadas.")

st.markdown("---")

# Painel de Decisão de Notícias / Insights de Alocação
section_header(ICO_TARGET, "Insights Estratégicos & Análise de Risco Qualitativo", "h2")

insights_html = []

if not live_news_items:
    insights_html.append(get_diag_row_html(ICO_WARN,
        "<b>Dados recentes indisponíveis:</b> Nenhuma observação de sentimento pode ser feita.", "#64748b"))
else:
    coverage_pct = news_coverage * 100
    insights_html.append(get_diag_row_html(ICO_IDEA,
        f"<b>Cobertura:</b> há notícias recentes para {coverage_pct:.1f}% do peso da carteira; a parcela sem cobertura foi excluída do sentimento.", "#00d2ff"))
    means = {}
    for item in live_news_items:
        means.setdefault(item["ticker"], []).append(item["score"])
    for ticker, scores in means.items():
        mean_score = sum(scores) / len(scores)
        label = "favorável" if mean_score >= 0.2 else "desfavorável" if mean_score <= -0.2 else "misto/neutro"
        insights_html.append(get_diag_row_html(ICO_WARN,
            f"<b>{escape(str(ticker))}:</b> tom médio {label} ({mean_score:+.2f}); observação textual, sem implicação de retorno.", "#ffd600"))

st.markdown(f"""
<div class="financial-panel">
    {"".join(insights_html)}
</div>
""", unsafe_allow_html=True)
