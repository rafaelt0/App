import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import fundamentus
import pandas as pd
import seaborn as sns
import warnings
import pypfopt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
import datetime
from scipy.stats import kurtosis, skew
from pypfopt import plotting
import re
import traceback
import time

from utils import db as _db
from utils.charts import apply_plotly_theme

import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Configurar temas de plotagem escuros
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#080c14'
plt.rcParams['axes.facecolor'] = '#0e1524'
plt.rcParams['text.color'] = '#f8fafc'
plt.rcParams['axes.labelcolor'] = '#94a3b8'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['grid.color'] = '#1e293b'
plt.rcParams['font.family'] = 'sans-serif'

# Customização do Plotly para o tema Obsidian Neo-Financial
def clean_numeric_column(col):
    col = col.astype(str).str.strip()
    col = col.str.replace(r'[^0-9,.\-]', '', regex=True)
    col = col.str.replace(',', '.')
    return pd.to_numeric(col, errors='coerce')

def get_ev_ebitda_context(setor: str):
    """Returns (alt_metric, reason) when EV/EBITDA doesn't apply, or None if it applies normally."""
    s = setor.lower() if setor else ""
    if any(k in s for k in ("banco", "crédito", "credito", "câmbio", "cambio")):
        return ("P/L · P/VP", "Bancos: resultado financeiro é a atividade-fim — EV/EBITDA não se aplica.")
    if any(k in s for k in ("seguro", "previdência", "previdencia", "resseguro")):
        return ("P/L · P/VP", "Seguradoras: lucro atrelado ao resultado financeiro (float) — EV/EBITDA não se aplica.")
    if any(k in s for k in ("holding", "participação", "participacao")):
        return ("Desconto sobre NAV", "Holdings: receita de equivalência patrimonial — EBITDA é quase nulo ou negativo.")
    if any(k in s for k in ("tecnologia", "software", "internet")):
        return ("EV/Sales", "Tech em crescimento: EBITDA frequentemente negativo pelo reinvestimento agressivo.")
    if any(k in s for k in ("exploração", "exploracao", "pré-operacional", "pre-operacional")):
        return ("EV/Recursos · DCF", "Empresa pré-operacional: sem receita, EBITDA estruturalmente negativo.")
    return None


@st.cache_data(ttl=3600)
def get_fundamentus_data(tickers):
    """Busca dados fundamentalistas com SQLite cache (4h) + retry automático."""
    cache_key = f"fund_{'_'.join(sorted(tickers))}"
    cached = _db.cache_get(cache_key, ttl=14400)
    if cached:
        try:
            df = pd.read_json(cached)
            if not df.empty:
                return df
        except Exception:
            pass
    last_exc = None
    for attempt in range(3):
        try:
            raw = [fundamentus.get_papel(t) for t in tickers]
            results = [r for r in raw if r is not None]
            if not results:
                raise RuntimeError(
                    f"Nenhum dado retornado pelo Fundamentus para: {', '.join(tickers)}. "
                    "Verifique se os tickers estão corretos."
                )
            result = pd.concat(results)
            _db.cache_set(cache_key, result.to_json())
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1)
                continue
    raise last_exc

@st.cache_data(ttl=3600)
def get_yfinance_data(tickers_yf, start, interval):
    """Busca cotações do Yahoo Finance com retry automático."""
    today = datetime.date.today()
    for attempt in range(3):
        try:
            return yf.download(tickers_yf, start=start, end=today, interval=interval)['Close']
        except OSError as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise

@st.cache_data(ttl=86400)
def get_sorted_tickers_by_liquidity(tickers_list):
    try:
        import fundamentus.resultado as fzr
        df = fzr.get_resultado_raw()
        df = df.sort_values(by='Liq.2meses', ascending=False)
        sorted_all = df.index.tolist()
        sorted_filtered = [t for t in sorted_all if t in tickers_list]
        remaining = [t for t in tickers_list if t not in sorted_filtered]
        return sorted_filtered + remaining
    except Exception:
        return tickers_list


@st.cache_data(ttl=14400, show_spinner=False)
def get_hist_fundamentals(ticker_sa: str):
    """Fetch annual income statement + balance sheet from yfinance (4h cache)."""
    t_yf = yf.Ticker(ticker_sa)
    out = {}
    try:
        fin = t_yf.financials
        if fin is not None and not fin.empty:
            out['fin'] = fin.to_json()
    except Exception:
        pass
    try:
        bs = t_yf.balance_sheet
        if bs is not None and not bs.empty:
            out['bs'] = bs.to_json()
    except Exception:
        pass
    return out


def _hist_parse(json_str):
    """Parse a yfinance JSON financials/balance-sheet string into a year-indexed DataFrame."""
    df_p = pd.read_json(json_str)
    try:
        df_p.columns = pd.to_datetime(df_p.columns.astype('int64'), unit='ms').year
    except Exception:
        try:
            df_p.columns = pd.to_datetime(df_p.columns).year
        except Exception:
            pass
    return df_p


def _hist_row(df_p, *keys):
    """Return the first matching row from df_p by trying each key in order."""
    for k in keys:
        if k in df_p.index:
            return df_p.loc[k]
    return None


def _build_hist_df(tkr: str):
    """Returns a DataFrame indexed by year with Receita, Lucro, Margens, ROE."""
    raw = get_hist_fundamentals(tkr + ".SA")
    if not raw:
        return None

    fin_df = bs_df = None
    if 'fin' in raw:
        try:
            fin_df = _hist_parse(raw['fin'])
        except Exception:
            pass
    if 'bs' in raw:
        try:
            bs_df = _hist_parse(raw['bs'])
        except Exception:
            pass

    records = {}

    if fin_df is not None:
        rev = _hist_row(fin_df, 'Total Revenue', 'Revenue')
        ni  = _hist_row(fin_df, 'Net Income', 'Net Income Common Stockholders',
                        'Net Income Applicable To Common Shares',
                        'Net Income Including Noncontrolling Interests')
        eb  = _hist_row(fin_df, 'EBIT', 'Operating Income', 'Ebit')

        for y in sorted(fin_df.columns.tolist()):
            y = int(y)
            rec = records.setdefault(y, {})
            r_v = float(rev[y]) if (rev is not None and y in rev.index and pd.notna(rev[y])) else None
            n_v = float(ni[y])  if (ni  is not None and y in ni.index  and pd.notna(ni[y]))  else None
            e_v = float(eb[y])  if (eb  is not None and y in eb.index  and pd.notna(eb[y]))  else None
            if r_v is not None:
                rec['Receita'] = r_v
            if n_v is not None:
                rec['Lucro Líquido'] = n_v
            if r_v and n_v is not None and abs(r_v) > 0:
                rec['Margem Líquida (%)'] = n_v / r_v * 100
            if r_v and e_v is not None and abs(r_v) > 0:
                rec['Margem EBIT (%)'] = e_v / r_v * 100

    if bs_df is not None and fin_df is not None:
        eq = _hist_row(bs_df, 'Stockholders Equity', 'Common Stock Equity',
                       'Total Stockholder Equity', 'Total Equity Gross Minority Interest')
        ni = _hist_row(fin_df, 'Net Income', 'Net Income Common Stockholders',
                       'Net Income Applicable To Common Shares',
                       'Net Income Including Noncontrolling Interests')
        if eq is not None and ni is not None:
            for y in sorted(bs_df.columns.tolist()):
                y = int(y)
                e_v = float(eq[y]) if y in eq.index and pd.notna(eq[y]) else None
                n_v = float(ni[y]) if y in ni.index and pd.notna(ni[y]) else None
                if e_v and n_v is not None and abs(e_v) > 0:
                    records.setdefault(y, {})['ROE (%)'] = n_v / e_v * 100

    if not records:
        return None
    df_h = pd.DataFrame(records).T.sort_index()
    df_h.index.name = 'Ano'
    return df_h


# Mapa de renomeação de colunas do fundamentus para identificadores internos
_FUNDAMENTUS_RENAME = {
    'P/L':        'PL',
    'P/VP':       'PVP',
    'EV/EBITDA':  'EV_EBITDA',
    'EV/EBIT':    'EV_EBIT',
    'PSR':        'PSR',
    'ROE':        'ROE',
    'ROIC':       'ROIC',
    'Mrg Ebit':   'Marg_EBIT',
    'Mrg. Líq.':  'Marg_Liquida',
    'Div.Yield':  'Div_Yield',
}

# Múltiplos a comparar na seção de peers: (coluna interna, nome display, menor=melhor?, categoria)
MULTIPLES_CFG = [
    ("PL",          "P/L",             True,  "Valuation"),
    ("PVP",         "P/VP",            True,  "Valuation"),
    ("EV_EBITDA",   "EV/EBITDA",       True,  "Valuation"),
    ("EV_EBIT",     "EV/EBIT",         True,  "Valuation"),
    ("PSR",         "PSR",             True,  "Valuation"),
    ("ROE",         "ROE (%)",         False, "Rentabilidade"),
    ("ROIC",        "ROIC (%)",        False, "Rentabilidade"),
    ("Marg_EBIT",   "Margem EBIT (%)", False, "Rentabilidade"),
    ("Marg_Liquida","Marg. Líq. (%)", False, "Rentabilidade"),
    ("Div_Yield",   "Div. Yield (%)",  False, "Yield"),
]
COLS_NEEDED = [c[0] for c in MULTIPLES_CFG]


@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_peers(_setores):
    """Busca todos os tickers listados no fundamentus e mapeia colunas para identificadores internos.

    O argumento ``_setores`` é recebido (como tupla) apenas para que o Streamlit inclua o setor
    na chave de cache; os dados retornados cobrem toda a B3 para permitir filtragem posterior
    por setor individual sem chamadas extras à API.
    """
    import fundamentus.resultado as fzr
    try:
        raw = fzr.get_resultado_raw()
        df2 = pd.DataFrame(index=raw.index)
        for src, dest in _FUNDAMENTUS_RENAME.items():
            if src in raw.columns:
                df2[dest] = raw[src]
        df2 = df2.drop_duplicates(keep='first')
        return df2
    except Exception:
        return pd.DataFrame()


# ── Funções utilitárias de formatação e renderização ──────────────────────────

def format_large_br_currency(value):
    """Formata valor em R$ com sufixo B/M."""
    if value >= 1e9:
        return f"R$ {value / 1e9:,.2f} B"
    elif value >= 1e6:
        return f"R$ {value / 1e6:,.2f} M"
    else:
        return f"R$ {value:,.2f}"


def format_large_number(value):
    """Formata número grande com sufixo B/M/K."""
    if value >= 1e9:
        return f"{value / 1e9:,.2f} B"
    elif value >= 1e6:
        return f"{value / 1e6:,.2f} M"
    elif value >= 1e3:
        return f"{value / 1e3:,.1f} K"
    else:
        return f"{value:,.0f}"


def extract_debt_metric(row, aliases):
    """Tenta extrair uma métrica testando vários nomes de coluna possíveis."""
    for name in aliases:
        if name in row.index:
            v = pd.to_numeric(
                str(row[name]).replace(',', '.').strip('%').strip(),
                errors='coerce'
            )
            if not pd.isna(v):
                return v
    return None


def render_sector_cards(ticker_name, row):
    """Renderiza cards de setor/subsetor para um ticker."""
    emp = row["Empresa"]
    setor = row["Setor"]
    sub = row["Subsetor"]
    metrics = [
        ("Empresa", emp, "#38bdf8"),
        ("Setor", setor, "#4ade80"),
        ("Subsetor", sub, "#fbbf24"),
    ]
    cards_html = "".join(
        f'<div class="mcard"><div class="mcard-label">{lbl}</div>'
        f'<div class="mcard-value" style="color:{clr};font-size:0.95rem">{val}</div></div>'
        for lbl, val, clr in metrics
    )
    st.markdown(f'<div class="mcard-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_price_cards(ticker_name, row):
    """Renderiza cards de preço/mercado para um ticker."""
    cot = row["Cotação"]
    min_52 = row["Mínimo (52 semanas)"]
    max_52 = row["Máximo (52 semanas)"]
    vol = row["Volume Médio (2 meses)"]
    val_merc = row["Valor de Mercado"]
    data_ult = row["Data Última Cotação"]
    metrics = [
        ("Cotação", f"R$ {cot:,.2f}", "#38bdf8"),
        ("Mín. 52 Sem.", f"R$ {min_52:,.2f}", "#f87171"),
        ("Máx. 52 Sem.", f"R$ {max_52:,.2f}", "#4ade80"),
        ("Vol. Médio", format_large_number(vol), "#fb7185"),
        ("Val. de Mercado", format_large_br_currency(val_merc), "#fbbf24"),
    ]
    cards_html = "".join(
        f'<div class="mcard"><div class="mcard-label">{lbl}</div>'
        f'<div class="mcard-value" style="color:{clr}">{val}</div></div>'
        for lbl, val, clr in metrics
    )
    st.markdown(f'<div class="mcard-grid">{cards_html}</div>', unsafe_allow_html=True)
    st.caption(f"Última cotação registrada: {data_ult} para {ticker_name}")


def render_ticker_cards(row, setor=""):
    """Renderiza cards de indicadores fundamentalistas para um ticker."""
    # 1. Valuation Section
    st.markdown("""
<div style="margin: 1.2rem 0 0.6rem 0; display: flex; align-items: center; gap: 6px;">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="1" x2="12" y2="23"></line>
        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
    </svg>
    <span style="font-weight: 700; color: #00d2ff; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Valuation</span>
</div>
""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("P/L", f"{row['P/L']:.2f}", help="Preço/Lucro: quantas vezes o mercado paga pelo lucro anual. Menor = mais barato.")
    with c2:
        st.metric("P/VP", f"{row['P/VP']:.2f}", help="Preço/Valor Patrimonial: compara o preço de mercado com o valor contábil. <1 pode indicar desconto.")
    with c3:
        ev_val = row['EV/EBITDA']
        if pd.isna(ev_val) or abs(ev_val) < 0.01:
            ctx = get_ev_ebitda_context(setor)
            if ctx:
                alt, reason = ctx
                st.metric("EV/EBITDA", "—",
                          delta="→ " + alt, delta_color="off", help=reason)
            else:
                st.metric("EV/EBITDA", "N/D",
                          help="Dado não disponível para este ticker via Fundamentus.")
        else:
            st.metric("EV/EBITDA", f"{ev_val:.2f}",
                      help="Enterprise Value / EBITDA: múltiplo de valuation que considera a dívida. Menor = mais barato.")

    # 2. Rentabilidade Section
    st.markdown("""
<div style="margin: 1.5rem 0 0.6rem 0; display: flex; align-items: center; gap: 6px;">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
        <polyline points="17 6 23 6 23 12"></polyline>
    </svg>
    <span style="font-weight: 700; color: #00ff87; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Rentabilidade</span>
</div>
""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        roe_val = row['ROE']
        st.metric("ROE", f"{roe_val:.2f}%",
                  delta="Forte" if roe_val > 15 else ("Fraco" if roe_val < 5 else "Moderado"),
                  delta_color="normal" if roe_val > 15 else ("inverse" if roe_val < 5 else "off"),
                  help="Return on Equity: lucro gerado para cada R$ de patrimônio. Acima de 15% é considerado bom.")
    with c2:
        roic_val = row['ROIC']
        st.metric("ROIC", f"{roic_val:.2f}%",
                  delta="Forte" if roic_val > 12 else ("Fraco" if roic_val < 5 else "Moderado"),
                  delta_color="normal" if roic_val > 12 else ("inverse" if roic_val < 5 else "off"),
                  help="Return on Invested Capital: eficiência no uso de todo o capital (próprio + dívida).")
    with c3:
        ml_val = row['Margem Líquida']
        st.metric("Margem Líquida", f"{ml_val:.2f}%",
                  delta="Alta" if ml_val > 15 else ("Baixa" if ml_val < 5 else "Média"),
                  delta_color="normal" if ml_val > 15 else ("inverse" if ml_val < 5 else "off"),
                  help="Percentual da receita que vira lucro líquido. Quanto maior, mais lucrativa a empresa.")
    with c4:
        mebit_val = row['Margem EBIT']
        st.metric("Margem EBIT", f"{mebit_val:.2f}%",
                  delta="Alta" if mebit_val > 15 else ("Baixa" if mebit_val < 5 else "Média"),
                  delta_color="normal" if mebit_val > 15 else ("inverse" if mebit_val < 5 else "off"),
                  help="Margem operacional antes de juros e impostos. Mede a eficiência operacional.")

    # 3. Crescimento & Yield Section
    st.markdown("""
<div style="margin: 1.5rem 0 0.6rem 0; display: flex; align-items: center; gap: 6px;">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffd600" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
    </svg>
    <span style="font-weight: 700; color: #ffd600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Crescimento & Yield</span>
</div>
""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dy_val = row['Dividend Yield']
        st.metric("Dividend Yield", f"{dy_val:.2f}%",
                  delta="Alto" if dy_val > 6 else ("Baixo" if dy_val < 2 else "Moderado"),
                  delta_color="normal" if dy_val > 4 else "off",
                  help="Dividendos pagos divididos pelo preço. Percentual de retorno em dividendos ao ano.")
    with c2:
        cr_val = row['Crescimento Receita 5 anos']
        st.metric("Crescimento Receita (5 anos)", f"{cr_val:.2f}%",
                  delta="Forte" if cr_val > 10 else ("Negativo" if cr_val < 0 else "Moderado"),
                  delta_color="normal" if cr_val > 10 else ("inverse" if cr_val < 0 else "off"),
                  help="Taxa de crescimento anual composta da receita nos últimos 5 anos.")


def render_debt_panel(ticker_name, row):
    """Renderiza painel de saúde financeira (endividamento) para um ticker."""
    DEBT_COL_ALIASES = {
        "div_brut_patrim": ["Dív.Brut/Patrim.", "Div_Brut_Patrim", "Div.Brut/Patrim."],
        "liq_corrente":    ["Liq. Corr.", "Liq_Corr", "Liq. Corr"],
        "ev_ebit":         ["EV_EBIT", "EV/EBIT"],
    }
    db_val = extract_debt_metric(row, DEBT_COL_ALIASES["div_brut_patrim"])
    lc_val = extract_debt_metric(row, DEBT_COL_ALIASES["liq_corrente"])
    ev_ebit_val = extract_debt_metric(row, DEBT_COL_ALIASES["ev_ebit"])
    # Fundamentus remove o decimal de múltiplos (EV/EBIT 12,5× → armazenado como 1250)
    if ev_ebit_val is not None:
        ev_ebit_val = ev_ebit_val / 100.0

    if db_val is None and lc_val is None and ev_ebit_val is None:
        st.info("Dados de endividamento não disponíveis via Fundamentus para este ticker.")
        return

    debt_cards = {}
    debt_colors = {}

    if db_val is not None:
        debt_cards["Dívida / Patrimônio"] = f"{db_val:.2f}×"
        debt_colors["Dívida / Patrimônio"] = "#ff3d5a" if db_val > 3 else ("#ffd600" if db_val > 1.5 else "#00ff87")

    if lc_val is not None:
        debt_cards["Liquidez Corrente"] = f"{lc_val:.2f}×"
        debt_colors["Liquidez Corrente"] = "#ff3d5a" if lc_val < 1 else ("#ffd600" if lc_val < 1.5 else "#00ff87")

    if ev_ebit_val is not None:
        debt_cards["EV / EBIT"] = f"{ev_ebit_val:.1f}×"
        debt_colors["EV / EBIT"] = "#ff3d5a" if ev_ebit_val > 20 else ("#ffd600" if ev_ebit_val > 12 else "#00ff87")

    cards_html = "".join(
        f'<div class="mcard"><div class="mcard-label">{lbl}</div>'
        f'<div class="mcard-value" style="color:{debt_colors[lbl]}">{val}</div></div>'
        for lbl, val in debt_cards.items()
    )
    st.markdown(f'<div class="mcard-grid">{cards_html}</div>', unsafe_allow_html=True)

    diags = []
    if db_val is not None:
        if db_val > 3:
            diags.append(("⚠️", f"Dívida/PL de {db_val:.1f}× é elevada — verifique capacidade de pagamento", "#ff3d5a"))
        elif db_val > 1.5:
            diags.append(("⚡", f"Dívida/PL de {db_val:.1f}× é moderada — monitorar", "#ffd600"))
        else:
            diags.append(("✓", f"Dívida/PL de {db_val:.1f}× é saudável", "#00ff87"))

    if lc_val is not None:
        if lc_val < 1:
            diags.append(("⚠️", f"Liquidez Corrente {lc_val:.2f}× < 1 — risco de dificuldade de caixa", "#ff3d5a"))
        elif lc_val < 1.5:
            diags.append(("⚡", f"Liquidez Corrente {lc_val:.2f}× — margem estreita", "#ffd600"))
        else:
            diags.append(("✓", f"Liquidez Corrente {lc_val:.2f}× — empresa com boa folga de caixa", "#00ff87"))

    for icon, msg, color in diags:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:8px;padding:0.35rem 0;'
            f'font-size:0.83rem;color:{color};">'
            f'<span style="flex-shrink:0">{icon}</span><span>{msg}</span></div>',
            unsafe_allow_html=True
        )


def _render_star_button(tkr):
    """Renderiza botão de favoritar/desfavoritar da watchlist."""
    starred = _db.wl_has(tkr)
    label = "★ Favoritado" if starred else "☆ Favoritar"
    if st.button(label, key=f"star_{tkr}",
                 help="Remover dos favoritos" if starred else "Salvar nos favoritos"):
        if starred:
            _db.wl_remove(tkr)
        else:
            _db.wl_add(tkr)
        st.rerun()


def color_veredicto(val):
    """Estilo CSS para coluna Veredicto na tabela de percentis."""
    m = {"Favorável": "#00ff8722", "Neutro": "#ffd60022", "Desfavorável": "#ff3d5a22"}
    c = {"Favorável": "#00ff87",   "Neutro": "#ffd600",   "Desfavorável": "#ff3d5a"}
    return f'background-color:{m.get(val,"")};color:{c.get(val,"")};font-weight:600'


def color_pct(val):
    """Estilo CSS para coluna Percentil na tabela de percentis."""
    if val >= 70:
        return 'color:#00ff87;font-weight:700'
    if val >= 40:
        return 'color:#ffd600;font-weight:700'
    return 'color:#ff3d5a;font-weight:700'


def _get_setor(df, ticker):
    """Retorna o setor de um ticker a partir do DataFrame fundamentus.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame retornado por ``get_fundamentus_data``.
    ticker : str
        Código do ticker (sem sufixo .SA).
    """
    if 'Setor' not in df.columns or ticker not in df.index:
        return ''
    val = df.loc[ticker, 'Setor']
    if isinstance(val, pd.Series):
        val = val.iloc[-1]
    return str(val) if not pd.isna(val) else ''


def _render_hist_section(tkr):
    """Renderiza seção de histórico fundamentalista (receita, margens, ROE) para um ticker."""
    with st.spinner(f"Buscando histórico de {tkr}..."):
        df_h = _build_hist_df(tkr)
    if df_h is None or df_h.empty:
        st.info(f"Dados históricos não disponíveis para {tkr} via yfinance.")
        return

    tab_rev, tab_marg, tab_roe = st.tabs(["Receita & Lucro", "Margens", "ROE"])

    with tab_rev:
        cols_rev = [c for c in ['Receita', 'Lucro Líquido'] if c in df_h.columns]
        if not cols_rev:
            st.info("Dados de receita não disponíveis.")
        else:
            df_rev = df_h[cols_rev].dropna(how='all')
            max_abs = df_rev.abs().max().max()
            scale, unit = (1e9, "R$ Bilhões") if max_abs >= 1e9 else (1e6, "R$ Milhões") if max_abs >= 1e6 else (1, "R$")
            df_rev = df_rev / scale
            clr = {'Receita': '#00d2ff', 'Lucro Líquido': '#00ff87'}
            fig_rev = go.Figure()
            for col in cols_rev:
                vals = df_rev[col].fillna(0)
                fig_rev.add_trace(go.Bar(
                    name=col,
                    x=df_rev.index.astype(str),
                    y=vals,
                    marker_color=clr.get(col, '#94a3b8'),
                    text=[f"{v:.1f}" if v != 0 else "" for v in vals],
                    textposition='outside',
                    textfont=dict(size=10, color='#f8fafc')
                ))
            fig_rev.update_layout(
                barmode='group', xaxis_title="Ano", yaxis_title=unit,
                height=360, margin=dict(t=20, b=40, l=40, r=20),
            )
            apply_plotly_theme(fig_rev)
            st.plotly_chart(fig_rev, use_container_width=True)

    with tab_marg:
        cols_m = [c for c in ['Margem Líquida (%)', 'Margem EBIT (%)'] if c in df_h.columns]
        if not cols_m:
            st.info("Dados de margem não disponíveis.")
        else:
            df_m = df_h[cols_m].dropna(how='all')
            clr_m = {'Margem Líquida (%)': '#00ff87', 'Margem EBIT (%)': '#ffd600'}
            fig_m = go.Figure()
            for col in cols_m:
                vals = df_m[col]
                fig_m.add_trace(go.Scatter(
                    name=col,
                    x=df_m.index.astype(str), y=vals,
                    mode='lines+markers+text',
                    line=dict(color=clr_m.get(col, '#94a3b8'), width=2.5),
                    marker=dict(size=8),
                    text=[f"{v:.1f}%" if pd.notna(v) else "" for v in vals],
                    textposition='top center',
                    textfont=dict(size=10, color='#f8fafc')
                ))
            fig_m.update_layout(
                xaxis_title="Ano", yaxis_title="Margem (%)",
                yaxis=dict(ticksuffix="%"),
                height=360, margin=dict(t=20, b=40, l=40, r=20),
            )
            apply_plotly_theme(fig_m)
            st.plotly_chart(fig_m, use_container_width=True)

    with tab_roe:
        if 'ROE (%)' not in df_h.columns or df_h['ROE (%)'].dropna().empty:
            st.info("Dados de ROE histórico não disponíveis.")
        else:
            df_roe = df_h[['ROE (%)']].dropna()
            fig_roe = go.Figure(go.Scatter(
                x=df_roe.index.astype(str), y=df_roe['ROE (%)'],
                mode='lines+markers+text',
                line=dict(color='#a855f7', width=2.5),
                marker=dict(size=9, color='#a855f7'),
                fill='tozeroy', fillcolor='rgba(168,85,247,0.08)',
                text=[f"{v:.1f}%" if pd.notna(v) else "" for v in df_roe['ROE (%)']],
                textposition='top center',
                textfont=dict(size=10, color='#f8fafc')
            ))
            fig_roe.add_hline(
                y=15, line_dash="dash", line_color="#00ff87", line_width=1.5,
                annotation_text="Referência: 15%", annotation_position="top right",
                annotation_font=dict(color="#00ff87", size=10)
            )
            fig_roe.update_layout(
                xaxis_title="Ano", yaxis_title="ROE (%)",
                yaxis=dict(ticksuffix="%"),
                height=360, margin=dict(t=20, b=40, l=40, r=20),
            )
            apply_plotly_theme(fig_roe)
            st.plotly_chart(fig_roe, use_container_width=True)


st.set_page_config(
    page_title="B3 Explorer — Análise Quantitativa de Ações",
    page_icon="b3.webp",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── SVG Icon Library ─────────────────────────────────────────────────────────
def _svg(body, size=14):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" style="vertical-align:-2px;margin-right:5px">'
            f'{body}</svg>')

ICO_COMPASS = _svg('<circle cx="12" cy="12" r="10" stroke="#00ff87" stroke-width="1.8"/>'
                   '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill="#00ff87"/>', 16)
ICO_SECTOR  = _svg('<rect x="3" y="3" width="7" height="9" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<rect x="14" y="3" width="7" height="5" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<rect x="3" y="16" width="7" height="5" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<rect x="14" y="12" width="7" height="9" rx="1" stroke="#00d2ff" stroke-width="1.8"/>', 16)
ICO_MARKET  = _svg('<path d="M3 3v18h18" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round"/>'
                   '<path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>', 16)
ICO_METRICS = _svg('<rect x="3" y="3" width="18" height="18" rx="3" stroke="#94a3b8" stroke-width="1.5"/>'
                   '<line x1="7" y1="9"  x2="17" y2="9"  stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="7" y1="13" x2="14" y2="13" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>'
                   '<line x1="7" y1="17" x2="15" y2="17" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>', 16)
ICO_CHART   = _svg('<rect x="3" y="12" width="3" height="9" rx="1" fill="#00ff87"/>'
                   '<rect x="9" y="7"  width="3" height="14" rx="1" fill="#00d2ff"/>'
                   '<rect x="15" y="9" width="3" height="12" rx="1" fill="#ffd600"/>', 16)
ICO_STATS   = _svg('<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" stroke="#a855f7" stroke-width="1.8"/>'
                   '<line x1="4" y1="22" x2="4" y2="15" stroke="#a855f7" stroke-width="1.8"/>', 16)
ICO_RULER   = _svg('<rect x="2" y="7" width="20" height="10" rx="2" stroke="#ffd600" stroke-width="1.8"/>'
                   '<line x1="6"  y1="7" x2="6"  y2="12" stroke="#ffd600" stroke-width="1.5"/>'
                   '<line x1="10" y1="7" x2="10" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
                   '<line x1="14" y1="7" x2="14" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
                   '<line x1="18" y1="7" x2="18" y2="12" stroke="#ffd600" stroke-width="1.5"/>', 16)
ICO_SHIELD  = _svg('<path d="M12 2l8 4v6c0 5-4 8.5-8 10C8 20.5 4 17 4 12V6l8-4z" stroke="#00ff87" stroke-width="1.8" fill="none"/>'
                   '<path d="M9 12l2 2 4-4" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>', 16)
ICO_BOX     = _svg('<rect x="3" y="7" width="18" height="14" rx="2" stroke="#94a3b8" stroke-width="1.8"/>'
                   '<path d="M8 7V5a4 4 0 018 0v2" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="12" y1="12" x2="12" y2="16" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="10" y1="14" x2="14" y2="14" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>', 16)
ICO_RADAR   = _svg('<polygon points="12 2 22 8.5 22 19.5 12 22 2 19.5 2 8.5" stroke="#a855f7" stroke-width="1.8" fill="none"/>'
                   '<polygon points="12 6 18 10 18 17 12 19 6 17 6 10" stroke="#a855f7" stroke-width="1.2" fill="none" opacity="0.6"/>'
                   '<line x1="12" y1="2" x2="12" y2="22" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>'
                   '<line x1="2" y1="8.5" x2="22" y2="19.5" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>'
                   '<line x1="2" y1="19.5" x2="22" y2="8.5" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>', 16)
ICO_INFO    = _svg('<circle cx="12" cy="12" r="10" stroke="#00d2ff" stroke-width="1.8"/>'
                   '<line x1="12" y1="16" x2="12" y2="12" stroke="#00d2ff" stroke-width="2" stroke-linecap="round"/>'
                   '<line x1="12" y1="8" x2="12" y2="8.01" stroke="#00d2ff" stroke-width="2" stroke-linecap="round"/>', 16)
ICO_NEWS    = _svg('<rect x="3" y="4" width="18" height="16" rx="2" stroke="#00ff87" stroke-width="1.8"/>'
                   '<line x1="7" y1="8" x2="17" y2="8" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
                   '<line x1="7" y1="12" x2="13" y2="12" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>'
                   '<line x1="7" y1="16" x2="15" y2="16" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>', 16)


def section_header(icon_svg, text, tag="h3"):
    st.markdown(
        f'<{tag} style="display:flex;align-items:center;gap:6px;margin-bottom:.4rem">'
        f'{icon_svg}<span>{text}</span></{tag}>',
        unsafe_allow_html=True)


# Sidebar Principal
st.sidebar.markdown("""
<div style="padding:0.5rem 0;border-bottom:1px solid #1e293b;margin-bottom:1rem">
  <span style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;color:#94a3b8;text-transform:uppercase">
    Navegação
  </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="padding:1rem 0 0.5rem 0;border-bottom:1px solid #1e293b;margin-bottom:1rem;">
  <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;color:#64748b;text-transform:uppercase;margin-bottom:0.75rem;">Fluxo de Análise</div>
  <div style="display:flex;flex-direction:column;gap:0.35rem;">
    <div style="display:flex;align-items:center;gap:0.6rem;">
      <div style="width:22px;height:22px;border-radius:50%;background:#00ff87;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 0 8px rgba(0,255,135,0.4);">
        <span style="font-size:0.65rem;font-weight:800;color:#080c14;">1</span>
      </div>
      <span style="font-size:0.8rem;font-weight:700;color:#00ff87;">Análise Fundamentalista</span>
    </div>
    <div style="width:1px;height:12px;background:#1e293b;margin-left:11px;"></div>
    <div style="display:flex;align-items:center;gap:0.6rem;opacity:0.45;">
      <div style="width:22px;height:22px;border-radius:50%;background:#1e293b;border:1.5px solid #334155;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <span style="font-size:0.65rem;font-weight:700;color:#64748b;">2</span>
      </div>
      <span style="font-size:0.8rem;font-weight:600;color:#64748b;">Portfolio</span>
    </div>
    <div style="width:1px;height:12px;background:#1e293b;margin-left:11px;"></div>
    <div style="display:flex;align-items:center;gap:0.6rem;opacity:0.35;">
      <div style="width:22px;height:22px;border-radius:50%;background:#1e293b;border:1.5px solid #334155;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <span style="font-size:0.65rem;font-weight:700;color:#64748b;">3</span>
      </div>
      <span style="font-size:0.8rem;font-weight:600;color:#64748b;">Simulação</span>
    </div>
    <div style="width:1px;height:12px;background:#1e293b;margin-left:11px;"></div>
    <div style="display:flex;align-items:center;gap:0.6rem;opacity:0.25;">
      <div style="width:22px;height:22px;border-radius:50%;background:#1e293b;border:1.5px solid #334155;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <span style="font-size:0.65rem;font-weight:700;color:#64748b;">4</span>
      </div>
      <span style="font-size:0.8rem;font-weight:600;color:#64748b;">Notícias</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# CSS customizado
try:
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass



# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero main-hero">
    <div class="page-hero-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60" fill="none">
          <!-- X-axis baseline -->
          <line x1="4" y1="54" x2="56" y2="54" stroke="#1e293b" stroke-width="1.5"/>
          <!-- Candle 1 — bullish green -->
          <line x1="14" y1="10" x2="14" y2="50" stroke="#334155" stroke-width="1.5"/>
          <rect x="10" y="22" width="8" height="18" rx="2" fill="#00ff87"/>
          <!-- Candle 2 — bearish red -->
          <line x1="30" y1="8" x2="30" y2="46" stroke="#334155" stroke-width="1.5"/>
          <rect x="26" y="16" width="8" height="20" rx="2" fill="#ff3d5a"/>
          <!-- Candle 3 — bullish green, stronger -->
          <line x1="46" y1="6" x2="46" y2="44" stroke="#334155" stroke-width="1.5"/>
          <rect x="42" y="12" width="8" height="22" rx="2" fill="#00ff87"/>
          <!-- Trend line (cyan) -->
          <path d="M6 50 L22 36 L38 42 L54 18"
                stroke="#00d2ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          <!-- Arrow head -->
          <path d="M48 14 L54 18 L50 24"
                stroke="#00d2ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
    </div>
    <div class="page-hero-content">
        <h1 class="page-hero-title">B3 Explorer</h1>
        <p class="page-hero-subtitle">Plataforma quantitativa de análise de ações brasileiras — Fundamentalismo, Otimização de Portfólio e Simulação Monte Carlo.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# Carrega lista de ações da B3 com setores para filtragem inicial

data = pd.read_csv('acoes-listadas-b3.csv')

if 'Setor' not in data.columns:
    st.error("O arquivo CSV precisa conter a coluna 'Setor' para o filtro funcionar.")
    st.stop()

# Cria listas de tickers e setores para seleção
stocks = list(data['Ticker'].values)
setores = sorted(data['Setor'].dropna().unique())
setores.insert(0, "Todos")
_ticker_setor = dict(zip(data['Ticker'], data['Setor']))

# ── Watchlist (sidebar) ───────────────────────────────────────────────────────
_watchlist = _db.wl_get()
if _watchlist:
    st.sidebar.markdown("""
<div style="padding:0.6rem 0 0.3rem 0;border-bottom:1px solid #1e293b;margin-bottom:0.5rem">
  <span style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#ffd600;text-transform:uppercase">
    ⭐ Favoritos
  </span>
</div>""", unsafe_allow_html=True)
    for _wt in _watchlist:
        _c1, _c2 = st.sidebar.columns([5, 1])
        with _c1:
            if st.button(_wt, key=f"wl_load_{_wt}", use_container_width=True,
                         help=_ticker_setor.get(_wt, "")):
                _cur = st.session_state.get("selected_tickers", [])
                if _wt not in _cur:
                    st.session_state["selected_tickers"] = _cur + [_wt]
                st.rerun()
        with _c2:
            if st.button("✕", key=f"wl_rm_{_wt}", help="Remover dos favoritos"):
                _db.wl_remove(_wt)
                st.rerun()
    st.sidebar.markdown("<div style='margin-bottom:0.75rem'></div>", unsafe_allow_html=True)

st.sidebar.subheader("Escolha o setor.")

# Permite filtro por setor na barra lateral
setores_selecionados = st.sidebar.multiselect(
    'Escolha um ou mais setores:', setores, default=[]
)

if st.session_state.get("selected_tickers"):
    if st.sidebar.button("🗑 Limpar seleção de ações", use_container_width=True):
        st.session_state["selected_tickers"] = []
        st.rerun()

#selecionar Todos ou nada, mostra todos os tickers
if "Todos" in setores_selecionados or not setores_selecionados:
    tickers_filtrados = data['Ticker'].tolist()
else:
    tickers_filtrados = data[data['Setor'].isin(setores_selecionados)]['Ticker'].tolist()

# Ordenar por liquidez para colocar maiores empresas no topo
tickers_filtrados = get_sorted_tickers_by_liquidity(tickers_filtrados)

section_header(ICO_COMPASS, "Escolha ações para explorar", "h3")
n_disponíveis = len(tickers_filtrados)
setor_label = "todos os setores" if (not setores_selecionados or "Todos" in setores_selecionados) else ", ".join(setores_selecionados[:2]) + ("..." if len(setores_selecionados) > 2 else "")
st.caption(f"{n_disponíveis} ações disponíveis em {setor_label}, ordenadas por liquidez.")

if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = []

# Remove any stale tickers that are no longer in the filtered list
st.session_state["selected_tickers"] = [
    t for t in st.session_state["selected_tickers"] if t in tickers_filtrados
]

tickers = st.multiselect(
    'Escolha sua ação. Selecione a página desejada e as configurações na barra lateral.',
    options=tickers_filtrados,
    format_func=lambda t: f"{t}  ·  {_ticker_setor.get(t, '')}",
    key="selected_tickers"
)

if not tickers:
    # ── Onboarding ─────────────────────────────────────────────────────────────
    st.markdown("""
<div style="margin:1.5rem 0 0.5rem 0;padding:1.5rem;background:linear-gradient(135deg,rgba(0,255,135,0.04) 0%,rgba(0,210,255,0.04) 100%);border:1px solid rgba(0,255,135,0.12);border-radius:14px;">
  <h3 style="margin:0 0 0.3rem 0;font-size:1.1rem;background:linear-gradient(135deg,#f8fafc 40%,#00ff87 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Bem-vindo ao B3 Explorer
  </h3>
  <p style="color:#94a3b8;font-size:0.85rem;margin:0 0 1.2rem 0;line-height:1.6">
    Plataforma de análise quantitativa de ações da B3. Escolha um ticker no campo acima
    ou clique em uma ação abaixo para começar.
  </p>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00ff87;text-transform:uppercase;margin-bottom:0.5rem">① Escolha</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Filtre por setor na barra lateral ou digite o código da ação no campo de busca acima.</div>
    </div>
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#00d2ff;text-transform:uppercase;margin-bottom:0.5rem">② Analise</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Veja P/L, EV/EBITDA, ROE, ROIC, endividamento e posição no setor vs peers automaticamente.</div>
    </div>
    <div style="background:rgba(14,23,38,0.6);border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;color:#a855f7;text-transform:uppercase;margin-bottom:0.5rem">③ Valoração</div>
      <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">Use a página <b style="color:#a855f7">Valuation</b> para o DCF completo McKinsey/Koller em 8 etapas.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Quick-start grid
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#64748b;'
        'text-transform:uppercase;margin:1.2rem 0 0.6rem 0">Início rápido — ações populares</div>',
        unsafe_allow_html=True)

    _QUICK = [
        ("WEGE3",  "Máquinas",      "#00ff87"),
        ("PETR4",  "Petróleo",       "#ffd600"),
        ("VALE3",  "Mineração",      "#f87171"),
        ("ITUB4",  "Banco",          "#00d2ff"),
        ("RENT3",  "Locação",        "#a855f7"),
        ("ABEV3",  "Bebidas",        "#fb923c"),
        ("EGIE3",  "Energia",        "#34d399"),
        ("RADL3",  "Farmácia",       "#60a5fa"),
    ]
    _cols = st.columns(4)
    for i, (tkr, setor, cor) in enumerate(_QUICK):
        with _cols[i % 4]:
            st.markdown(
                f'<div style="text-align:center;padding:0.2rem 0 0.1rem 0;'
                f'font-size:0.62rem;color:{cor};font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.06em">{setor}</div>',
                unsafe_allow_html=True)
            if st.button(tkr, key=f"qs_{tkr}", use_container_width=True,
                         help=f"Analisar {tkr} — {setor}"):
                st.session_state["selected_tickers"] = [tkr]
                st.rerun()

    # Sector shortcuts
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#64748b;'
        'text-transform:uppercase;margin:1.4rem 0 0.6rem 0">Explorar por setor</div>',
        unsafe_allow_html=True)
    _SETORES_DEST = [
        "Bancos", "Petróleo e Gás", "Mineração", "Energia Elétrica",
        "Tecnologia", "Bebidas", "Saúde", "Varejo",
    ]
    _sc = st.columns(4)
    for i, _s in enumerate(_SETORES_DEST):
        with _sc[i % 4]:
            if st.button(_s, key=f"setor_qs_{_s}", use_container_width=True):
                # Match against available sectors (partial match)
                _match = [s for s in setores if _s.lower() in s.lower() or s.lower() in _s.lower()]
                if _match:
                    st.session_state["selected_sectors_hint"] = _match[0]
                st.toast(f'Filtre por "{_s}" no seletor de setores na barra lateral →', icon="↙")

    st.markdown(
        '<div style="margin-top:1.2rem;padding:0.75rem 1rem;background:rgba(0,0,0,0.2);'
        'border-radius:8px;border-left:3px solid #334155">'
        '<span style="font-size:0.78rem;color:#64748b">💡 <b style="color:#94a3b8">Dica:</b> '
        'Use o campo de busca acima para digitar qualquer ticker da B3. '
        'O filtro de setor na barra lateral reduz a lista para facilitar a escolha.</span>'
        '</div>',
        unsafe_allow_html=True)

if tickers:
    st.markdown(f"""
    <div style="background:rgba(0,255,135,0.05);border:1px solid rgba(0,255,135,0.2);border-radius:10px;padding:0.6rem 1rem;display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
        <span style="font-size:0.85rem;color:#94a3b8;">{len(tickers)} ação(ões) selecionada(s) — analise o portfólio completo na página <strong style="color:#00ff87">Portfolio</strong></span>
        <span style="font-size:0.75rem;color:#00ff87;font-family:'JetBrains Mono',monospace;font-weight:700;">← barra lateral</span>
    </div>
    """, unsafe_allow_html=True)

# Só executa análise se houver pelo menos uma ação selecionada
if tickers:
    try:
        # 1. Exibir tela de carregamento glassmorphic
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">Buscando indicadores fundamentalistas na B3...</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Buscar dados usando funções cacheadas
        df = get_fundamentus_data(tickers)

        tickers_yf = [t + ".SA" for t in tickers]

        # 3. Limpar tela de carregamento
        loading_placeholder.empty()

        # Export button for fundamental data
        try:
            export_cols = ['Empresa', 'Setor', 'Subsetor', 'Cotacao', 'Min_52_sem', 'Max_52_sem',
                           'Marg_Liquida', 'Marg_EBIT', 'ROE', 'ROIC', 'Div_Yield',
                           'Cres_Rec_5a', 'PL', 'EV_EBITDA', 'PVP']
            export_cols_exist = [c for c in export_cols if c in df.columns]
            df_export = df[export_cols_exist].copy()
            csv_data = df_export.to_csv().encode('utf-8')
            st.download_button(
                label="⬇ Exportar dados fundamentalistas (CSV)",
                data=csv_data,
                file_name=f"fundamentus_{'+'.join(tickers)}_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        except Exception:
            pass

        section_header(ICO_SECTOR, "Setor", "h3")
        df_sector = df[['Empresa', 'Setor', 'Subsetor']]
        
        if len(tickers) > 1:
            tabs_sector = st.tabs(tickers)
            for tab, ticker in zip(tabs_sector, tickers):
                with tab:
                    if ticker in df_sector.index:
                        row_s = df_sector.loc[ticker]
                        if isinstance(row_s, pd.DataFrame):
                            row_s = row_s.iloc[-1]
                        render_sector_cards(ticker, row_s)
                    else:
                        st.warning(f"Sem dados de setor para {ticker}")
        else:
            ticker = tickers[0]
            if ticker in df_sector.index:
                row_s = df_sector.loc[ticker]
                if isinstance(row_s, pd.DataFrame):
                    row_s = row_s.iloc[-1]
                render_sector_cards(ticker, row_s)
            else:
                st.warning(f"Sem dados de setor para {ticker}")


        # Informações de mercado em caixas estilizadas
        section_header(ICO_MARKET, "Informações de Mercado", "h3")
        df_price = df[['Cotacao', 'Min_52_sem', 'Max_52_sem', 'Vol_med_2m', 
                       'Valor_de_mercado', 'Data_ult_cot']]
        df_price.columns = ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)",
                            "Volume Médio (2 meses)", "Valor de Mercado", "Data Última Cotação"]

        # Limpa colunas numéricas para evitar erros de formatação
        for col in ["Cotação", "Mínimo (52 semanas)", "Máximo (52 semanas)", 
                    "Volume Médio (2 meses)", "Valor de Mercado"]:
            df_price[col] = clean_numeric_column(df_price[col]).fillna(0)

        if len(tickers) > 1:
            tabs_price = st.tabs(tickers)
            for tab, ticker in zip(tabs_price, tickers):
                with tab:
                    if ticker in df_price.index:
                        row_p = df_price.loc[ticker]
                        if isinstance(row_p, pd.DataFrame):
                            row_p = row_p.iloc[-1]
                        render_price_cards(ticker, row_p)
                    else:
                        st.warning(f"Sem dados de mercado para {ticker}")
        else:
            ticker = tickers[0]
            if ticker in df_price.index:
                row_p = df_price.loc[ticker]
                if isinstance(row_p, pd.DataFrame):
                    row_p = row_p.iloc[-1]
                render_price_cards(ticker, row_p)
            else:
                st.warning(f"Sem dados de mercado para {ticker}")


        # Indicadores Fundamentalistas
        section_header(ICO_METRICS, "Indicadores Financeiros", "h3")
        df_ind = df[['Marg_Liquida','Marg_EBIT','ROE','ROIC','Div_Yield',
                     'Cres_Rec_5a','PL','EV_EBITDA','PVP','Empresa']].drop_duplicates(keep='last')
        df_ind.columns = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC",
                          "Dividend Yield", "Crescimento Receita 5 anos", "P/L", "EV/EBITDA", "P/VP", "Empresa"]

        # Transforma tudo em numérico para poder filtrar e aplicar estilos
        for col in df_ind.columns.drop('Empresa'):
            df_ind[col] = clean_numeric_column(df_ind[col])

        # Corrige o bug de parsing da biblioteca fundamentus (removeu o decimal)
        for col in ["P/L", "EV/EBITDA", "P/VP"]:
            if col in df_ind.columns:
                df_ind[col] = df_ind[col] / 100.0

        # Colunas percentuais
        pct_cols = ["Margem Líquida", "Margem EBIT", "ROE", "ROIC", "Dividend Yield", "Crescimento Receita 5 anos"]
        for col in pct_cols:
            df_ind[col] = df_ind[col]

        df_ind = df_ind.fillna(0)

        # Remove duplicate indices if any
        df_ind = df_ind[~df_ind.index.duplicated(keep='last')]

        def _get_setor(ticker):
            if 'Setor' not in df.columns or ticker not in df.index:
                return ''
            val = df.loc[ticker, 'Setor']
            if isinstance(val, pd.Series):
                val = val.iloc[-1]
            return str(val) if not pd.isna(val) else ''

        # Exibição dos cards
        if len(tickers) > 1:
            tabs_tickers = st.tabs(tickers)
            for idx, ticker in enumerate(tickers):
                with tabs_tickers[idx]:
                    _render_star_button(ticker)
                    if ticker in df_ind.index:
                        render_ticker_cards(df_ind.loc[ticker], setor=_get_setor(ticker))
        else:
            ticker = tickers[0]
            _render_star_button(ticker)
            if ticker in df_ind.index:
                render_ticker_cards(df_ind.loc[ticker], setor=_get_setor(ticker))

        # ── Histórico Fundamentalista ─────────────────────────────────────────
        st.markdown("---")
        section_header(ICO_CHART, "Histórico Fundamentalista", "h3")
        st.caption("Evolução anual de receita, lucro, margens e ROE — últimos 4 anos (fonte: yfinance / relatórios anuais).")

        if len(tickers) > 1:
            tabs_hist = st.tabs(tickers)
            for _h_idx, _h_tkr in enumerate(tickers):
                with tabs_hist[_h_idx]:
                    _render_hist_section(_h_tkr)
        else:
            _render_hist_section(tickers[0])

        # ── Saúde Financeira ─────────────────────────────────────────────────
        st.markdown("---")
        section_header(ICO_SHIELD, "Saúde Financeira", "h3")
        st.caption("Endividamento e liquidez da empresa. "
                   "Dívida/PL acima de 3x e Liquidez abaixo de 1x são sinais de alerta.")

        if len(tickers) > 1:
            tabs_debt = st.tabs(tickers)
            for tab, ticker in zip(tabs_debt, tickers):
                with tab:
                    if ticker in df.index:
                        row_debt = df.loc[ticker]
                        if isinstance(row_debt, pd.DataFrame):
                            row_debt = row_debt.iloc[-1]
                        render_debt_panel(ticker, row_debt)
                    else:
                        st.warning(f"Sem dados para {ticker}")
        else:
            ticker = tickers[0]
            if ticker in df.index:
                row_debt = df.loc[ticker]
                if isinstance(row_debt, pd.DataFrame):
                    row_debt = row_debt.iloc[-1]
                render_debt_panel(ticker, row_debt)

        # ── Link para página de Valuation ────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div style="background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.25);'
            'border-radius:10px;padding:1rem 1.25rem;margin:0.5rem 0">'
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
            '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" '
            'stroke="#a855f7" stroke-width="2" stroke-linecap="round"><line x1="12" y1="1" x2="12" y2="23"/>'
            '<path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" fill="none"/></svg>'
            '<span style="color:#a855f7;font-weight:700;font-size:0.95rem">Valuation McKinsey/Koller</span>'
            '</div>'
            '<p style="color:#94a3b8;font-size:0.82rem;margin:0">'
            'O módulo de valuation completo (Enterprise DCF, NOPLAT, ROIC, Continuing Value, WACC, '
            'Sensitivity, Múltiplos) foi movido para a página <b style="color:#f8fafc">Valuation</b> '
            'no menu lateral — implementado com a metodologia McKinsey/Koller '
            '(<i>Measuring and Managing the Value of Companies</i>).'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )

        # ── Comparação Visual de Múltiplos ───────────────────────────────────
        if len(tickers) > 1:
            st.markdown("""
<h4 style="display:flex;align-items:center;gap:6px;margin-top:1.5rem;margin-bottom:.4rem">
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"></line>
    <line x1="12" y1="20" x2="12" y2="4"></line>
    <line x1="6" y1="20" x2="6" y2="14"></line>
  </svg>
  <span>Comparação Gráfica de Múltiplos</span>
</h4>
""", unsafe_allow_html=True)
            
            comparable_indicators = [
                "P/L", "P/VP", "EV/EBITDA", "ROE", "ROIC", "Dividend Yield", 
                "Margem Líquida", "Margem EBIT", "Crescimento Receita 5 anos"
            ]
            
            col_chart_sel, _ = st.columns([1, 1])
            with col_chart_sel:
                selected_comp_mult = st.selectbox(
                    "Selecione o indicador para o gráfico comparativo",
                    comparable_indicators,
                    key="comp_mult_select_key"
                )
            
            df_chart = df_ind[[selected_comp_mult]].copy()
            
            is_pct = selected_comp_mult in ["ROE", "ROIC", "Dividend Yield", "Margem Líquida", "Margem EBIT", "Crescimento Receita 5 anos"]
            text_labels = []
            for v in df_chart[selected_comp_mult].values:
                if pd.isna(v):
                    text_labels.append("N/D")
                elif selected_comp_mult == "EV/EBITDA" and abs(v) < 0.01:
                    text_labels.append("N/A")
                elif is_pct:
                    text_labels.append(f"{v:.2f}%")
                else:
                    text_labels.append(f"{v:.2f}")
            
            fig_comp = go.Figure(go.Bar(
                x=df_chart.index.tolist(),
                y=df_chart[selected_comp_mult].values,
                marker=dict(
                    color=df_chart[selected_comp_mult].values,
                    colorscale="Viridis",
                    showscale=False
                ),
                text=text_labels,
                textposition="outside",
                textfont=dict(size=10, color="#f8fafc")
            ))
            
            fig_comp.update_layout(
                title=dict(
                    text=f"Comparativo de {selected_comp_mult} — Ações Selecionadas",
                    font=dict(size=14, color="#f8fafc")
                ),
                xaxis_title="Ação",
                yaxis_title=f"{selected_comp_mult} (%)" if is_pct else selected_comp_mult,
                height=350,
                margin=dict(t=50, b=40, l=40, r=40)
            )
            apply_plotly_theme(fig_comp)
            st.plotly_chart(fig_comp, use_container_width=True)

        # ── Comparação de Múltiplos do Setor ──────────────────────────────────
        st.markdown("---")
        st.markdown("""
<h3 style="display:flex;align-items:center;gap:8px;margin-bottom:.25rem">
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none"
       style="vertical-align:-3px">
    <circle cx="7"  cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>
    <circle cx="17" cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>
    <line x1="10" y1="12" x2="14" y2="12" stroke="#00d2ff" stroke-width="1.8"/>
    <rect x="4" y="3" width="6" height="4" rx="1" fill="#00ff87" opacity="0.7"/>
    <rect x="14" y="3" width="6" height="4" rx="1" fill="#ffd600" opacity="0.7"/>
    <rect x="4" y="17" width="6" height="4" rx="1" fill="#a855f7" opacity="0.7"/>
    <rect x="14" y="17" width="6" height="4" rx="1" fill="#ff3d5a" opacity="0.7"/>
  </svg>
  Comparação de Múltiplos do Setor
</h3>
<p style="color:#94a3b8;font-size:0.85rem;margin-top:0;margin-bottom:1rem">
  Posicionamento da(s) ação(ões) selecionada(s) em relação a todos os pares do setor na B3.
</p>
""", unsafe_allow_html=True)

        rank_df = pd.DataFrame()  # Inicializa; será populado se houver dados de peers

        # Detecta setor(es) das ações selecionadas
        setores_ativas = df['Setor'].dropna().unique().tolist()

        # Mapa ticker -> setor individual
        ticker_setor_map = {}
        for t in tickers:
            sr = data[data['Ticker'] == t]
            if not sr.empty:
                ticker_setor_map[t] = sr['Setor'].values[0]

        if not setores_ativas:
            st.info("Setor não identificado para comparação.")
        else:
            # Mostra setor de cada ativo individualmente
            setor_info = " | ".join([f"**{t}**: {ticker_setor_map.get(t, '?')}" for t in tickers])
            st.caption(f"Setores detectados: {setor_info}")

            # Busca todos os tickers do setor via fundamentus (função cacheada no nível do módulo)
            with st.spinner("Buscando peers do setor..."):
                peers_raw = get_sector_peers(tuple(setores_ativas))

            if peers_raw.empty:
                st.warning("Não foi possível buscar os pares do setor.")
            else:
                # fundamentus.get_resultado() retorna DataFrame com tickers como índice
                # Filtra somente os múltiplos que existem
                cols_available = [c for c in COLS_NEEDED if c in peers_raw.columns]
                peers_df = peers_raw[cols_available].copy()

                # Converte tudo para numérico
                for col in cols_available:
                    peers_df[col] = pd.to_numeric(
                        peers_df[col].astype(str).str.replace(',', '.').str.replace(r'[^\d.\-]', '', regex=True),
                        errors='coerce'
                    )

                # Limpa outliers e valores não aplicáveis (por célula, sem remover a linha inteira)
                # Fundamentus usa 0 para indicar "não aplicável" em muitos múltiplos
                for mult, name, lower_better, _ in MULTIPLES_CFG:
                    if mult not in peers_df.columns:
                        continue
                    # Converte zeros exatos em NaN (0 = não aplicável no fundamentus)
                    peers_df.loc[peers_df[mult] == 0, mult] = pd.NA
                    if lower_better:
                        # Remove apenas valores negativos ou absurdamente altos (> 500)
                        peers_df.loc[
                            (peers_df[mult].notna()) &
                            ((peers_df[mult] < 0) | (peers_df[mult] >= 500)),
                            mult
                        ] = pd.NA

                peers_df = peers_df.dropna(how='all')

                # Tickers selecionados no setor
                selected_in_peers = [t for t in tickers if t in peers_df.index]
                if not selected_in_peers:
                    # tenta com .SA removido
                    selected_in_peers = tickers

                # ── Tabela de percentis ─────────────────────────────────────
                st.markdown("#### Posicionamento por Percentil")
                st.caption("Verde = favorável | Vermelho = desfavorável | Cinza = neutro. "
                           "O percentil indica onde a ação está no ranking do setor (100 = melhor).")

                rows = []
                for mult, name, lower_better, categoria in MULTIPLES_CFG:
                    if mult not in peers_df.columns:
                        continue
                    col_data = peers_df[mult].dropna()
                    if col_data.empty:
                        continue

                    for t in selected_in_peers:
                        if t not in peers_df.index or pd.isna(peers_df.loc[t, mult]):
                            continue
                        val = peers_df.loc[t, mult]

                        # Obtém o setor da empresa individual
                        setor_row = data[data['Ticker'] == t]
                        if not setor_row.empty:
                            setor_t = setor_row['Setor'].values[0]
                            tickers_do_setor_t = data[data['Setor'] == setor_t]['Ticker'].tolist()
                            if t not in tickers_do_setor_t:
                                tickers_do_setor_t.append(t)
                            col_data_sector = col_data[col_data.index.isin(tickers_do_setor_t)]
                        else:
                            col_data_sector = col_data

                        if col_data_sector.empty:
                            col_data_sector = col_data

                        setor_med = col_data_sector.median()
                        setor_mean = col_data_sector.mean()
                        n_peers = len(col_data_sector)

                        # Percentil: % de peers piores que esta ação neste múltiplo
                        if lower_better:
                            pct = (col_data_sector > val).sum() / n_peers * 100
                        else:
                            pct = (col_data_sector < val).sum() / n_peers * 100

                        # Veredicto
                        if pct >= 70:
                            veredicto = "Favorável"
                            cor = "#00ff87"
                        elif pct >= 40:
                            veredicto = "Neutro"
                            cor = "#ffd600"
                        else:
                            veredicto = "Desfavorável"
                            cor = "#ff3d5a"

                        rows.append({
                            "Ação": t,
                            "Setor": ticker_setor_map.get(t, "N/D"),
                            "Categoria": categoria,
                            "Múltiplo": name,
                            "Valor": round(val, 2),
                            "Mediana Setor": round(setor_med, 2),
                            "Média Setor": round(setor_mean, 2),
                            "Peers (n)": n_peers,
                            "Percentil": round(pct, 1),
                            "Veredicto": veredicto,
                            "_cor": cor,
                        })

                if rows:
                    rank_df = pd.DataFrame(rows)

                    display_df = rank_df.drop(columns=["_cor"])
                    styled_rank = (
                        display_df.style
                        .map(color_veredicto, subset=["Veredicto"])
                        .map(color_pct,       subset=["Percentil"])
                        .format({"Valor": "{:.2f}", "Mediana Setor": "{:.2f}",
                                 "Média Setor": "{:.2f}", "Percentil": "{:.1f}%"})
                        .set_properties(**{"font-family": "JetBrains Mono, monospace", "font-size": "0.82rem"})
                    )
                    st.dataframe(styled_rank, use_container_width=True, hide_index=True)

                    # ── Gráfico Comparativo de Percentis no Setor ───────────
                    st.markdown("#### Performance Relativa no Setor")
                    st.caption("Percentil de posicionamento no setor para cada indicador (100% representa o melhor posicionamento no setor).")
                    
                    tab_labels_perf = [f"{t} ({ticker_setor_map.get(t, '?')})" for t in selected_in_peers]
                    tabs_perf = st.tabs(tab_labels_perf)
                    for i, ticker_s in enumerate(selected_in_peers):
                        with tabs_perf[i]:
                            setor_name = ticker_setor_map.get(ticker_s, 'N/D')
                            sub_rank = rank_df[rank_df["Ação"] == ticker_s]
                            n_peers_display = sub_rank["Peers (n)"].max() if not sub_rank.empty else 0
                            st.caption(f"Comparado contra **{int(n_peers_display)} peers** do setor **{setor_name}**")
                            fig_pct = px.bar(
                                sub_rank,
                                x="Múltiplo",
                                y="Percentil",
                                color="Ação",
                                color_discrete_sequence=['#00ff87'],
                                title=None,
                                labels={"Percentil": "Percentil no Setor (%)", "Múltiplo": "Múltiplo / Indicador"}
                            )
                            fig_pct.update_layout(
                                yaxis=dict(range=[0, 105], ticksuffix="%"),
                                height=380,
                                margin=dict(t=30, b=40, l=40, r=40),
                                showlegend=False
                            )
                            apply_plotly_theme(fig_pct)
                            st.plotly_chart(fig_pct, use_container_width=True)

                    # ── Gráfico de barras por múltiplo ───────────────────────
                    st.markdown("#### Distribuição do Setor por Múltiplo")
                    mult_opcoes = [m[1] for m in MULTIPLES_CFG if m[0] in peers_df.columns]
                    mult_sel = st.selectbox("Selecione o múltiplo para visualizar", mult_opcoes, key="mult_sel")

                    mult_key = next((m[0] for m in MULTIPLES_CFG if m[1] == mult_sel), None)
                    if mult_key and mult_key in peers_df.columns:
                        tab_labels_dist = [f"{t} ({ticker_setor_map.get(t, '?')})" for t in selected_in_peers]
                        tabs_dist = st.tabs(tab_labels_dist)
                        for i, ticker_s in enumerate(selected_in_peers):
                            with tabs_dist[i]:
                                setor_row = data[data['Ticker'] == ticker_s]
                                if not setor_row.empty:
                                    setor_t = setor_row['Setor'].values[0]
                                    tickers_do_setor_t = data[data['Setor'] == setor_t]['Ticker'].tolist()
                                    if ticker_s not in tickers_do_setor_t:
                                        tickers_do_setor_t.append(ticker_s)
                                    
                                    col_plot = peers_df[peers_df.index.isin(tickers_do_setor_t)][mult_key].dropna().sort_values()
                                    
                                    if col_plot.empty:
                                        st.warning(f"Dados indisponíveis para o setor de {ticker_s}.")
                                        continue
                                        
                                    bar_colors = ['#1e293b'] * len(col_plot)
                                    if ticker_s in col_plot.index:
                                        bar_colors[list(col_plot.index).index(ticker_s)] = '#00ff87'
                                        
                                    fig_mult = go.Figure(go.Bar(
                                        x=col_plot.index.tolist(),
                                        y=col_plot.values,
                                        marker_color=bar_colors,
                                        marker_line_width=0,
                                        text=[f"{v:.1f}" for v in col_plot.values],
                                        textposition="outside",
                                        textfont=dict(size=8, color="#94a3b8"),
                                    ))

                                    # Mediana
                                    med_val = col_plot.median()
                                    fig_mult.add_hline(
                                        y=med_val,
                                        line_dash="dash", line_color="#ffd600", line_width=1.5,
                                        annotation_text=f"Mediana: {med_val:.1f}",
                                        annotation_position="top left",
                                        annotation_font=dict(color="#ffd600", size=11)
                                    )
                                    fig_mult.update_layout(
                                        title=f"{mult_sel} — Peers de {ticker_s} no Setor: {setor_t} ({len(col_plot)} empresas)",
                                        xaxis_title="Ticker",
                                        yaxis_title=mult_sel,
                                        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                                        height=420,
                                        showlegend=False,
                                    )
                                    apply_plotly_theme(fig_mult)
                                    st.plotly_chart(fig_mult, use_container_width=True)

                    # ── Scorecard de valuation ────────────────────────────────
                    st.markdown("#### Scorecard de Valuation do Setor")
                    score_cols = st.columns(len(selected_in_peers))

                    for i, ticker_s in enumerate(selected_in_peers):
                        sub = rank_df[rank_df["Ação"] == ticker_s]
                        if sub.empty:
                            continue
                        total_score = sub["Percentil"].mean()
                        fav  = (sub["Veredicto"] == "Favorável").sum()
                        neut = (sub["Veredicto"] == "Neutro").sum()
                        desf = (sub["Veredicto"] == "Desfavorável").sum()

                        sc = "#00ff87" if total_score >= 60 else "#ffd600" if total_score >= 40 else "#ff3d5a"
                        label = "ATRATIVO" if total_score >= 60 else "NEUTRO" if total_score >= 40 else "CARO/FRACO"

                        with score_cols[i]:
                            st.markdown(f"""
<div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:2px solid {sc};
            border-radius:14px;padding:1.2rem;text-align:center;
            box-shadow:0 0 18px {sc}33;margin-bottom:.5rem">
  <div style="font-size:1.1rem;font-weight:700;color:#94a3b8;letter-spacing:.06em">{ticker_s}</div>
  <div style="font-size:2.8rem;font-weight:900;color:{sc};font-family:'JetBrains Mono',monospace;
              text-shadow:0 0 12px {sc}66;line-height:1.1">{total_score:.0f}</div>
  <div style="font-size:0.6rem;color:#64748b;letter-spacing:.12em">PERCENTIL MÉDIO</div>
  <div style="font-size:0.8rem;font-weight:700;color:{sc};margin-top:.4rem;letter-spacing:.06em">{label}</div>
  <div style="font-size:0.72rem;color:#94a3b8;margin-top:.6rem;display:flex;justify-content:center;gap:.8rem">
    <span style="color:#00ff87">✓ {fav} fav.</span>
    <span style="color:#ffd600">~ {neut} neut.</span>
    <span style="color:#ff3d5a">✗ {desf} desf.</span>
  </div>
</div>
""", unsafe_allow_html=True)
                else:
                    st.info("Nenhum ticker selecionado encontrado nos dados de peers do setor.")




        # ── Síntese do Analista ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
<h3 style="display:flex;align-items:center;gap:8px;margin-bottom:.5rem">
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="#a855f7" stroke-width="1.8"/>
    <path d="M12 8v4l3 3" stroke="#a855f7" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <span style="background:linear-gradient(135deg,#f8fafc,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Síntese do Analista</span>
</h3>
""", unsafe_allow_html=True)

        usar_percentis = not rank_df.empty

        sintese_items = []
        for t in tickers:
            if t not in df_ind.index:
                continue
            r = df_ind.loc[t]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[-1]
            nome = df.loc[t, 'Empresa'] if t in df.index else t
            if isinstance(nome, pd.Series):
                nome = nome.iloc[0]

            pontos_pos = []
            pontos_neg = []
            alertas = []
            fonte_label = ""

            if usar_percentis:
                # Usa percentis do setor já calculados
                ticker_rank = rank_df[rank_df["Ação"] == t]
                fonte_label = "vs. peers do setor"
                for _, rr in ticker_rank.iterrows():
                    mult = rr["Múltiplo"]
                    pct = rr["Percentil"]
                    val = rr["Valor"]
                    med = rr["Mediana Setor"]
                    vs = f"mediana do setor: {med:.2f}"
                    if pct >= 70:
                        pontos_pos.append(f"{mult} no percentil <b>{pct:.0f}°</b> do setor — {val:.2f} ({vs})")
                    elif pct < 30:
                        pontos_neg.append(f"{mult} no percentil <b>{pct:.0f}°</b> do setor — {val:.2f} ({vs})")
                # Alertas extras de P/L absoluto
                pl = float(r.get('P/L', 0) or 0)
                if pl < 0:
                    alertas.append(f"P/L negativo ({pl:.1f}x) — empresa registrou prejuízo")
            else:
                # Fallback: thresholds absolutos com contexto
                fonte_label = "thresholds de mercado"
                roe = float(r.get('ROE', 0) or 0)
                roic = float(r.get('ROIC', 0) or 0)
                pl = float(r.get('P/L', 0) or 0)
                dy = float(r.get('Dividend Yield', 0) or 0)
                ml = float(r.get('Margem Líquida', 0) or 0)
                cr = float(r.get('Crescimento Receita 5 anos', 0) or 0)

                if roe > 15: pontos_pos.append(f"ROE forte ({roe:.1f}% — acima dos 15% de referência)")
                elif roe < 5: pontos_neg.append(f"ROE fraco ({roe:.1f}% — abaixo dos 5% mínimos)")

                if roic > 12: pontos_pos.append(f"ROIC sólido ({roic:.1f}% — acima dos 12%)")
                elif roic < 5: pontos_neg.append(f"ROIC baixo ({roic:.1f}%)")

                if 0 < pl < 15: pontos_pos.append(f"P/L atrativo ({pl:.1f}x — abaixo de 15x)")
                elif pl > 30: pontos_neg.append(f"P/L elevado ({pl:.1f}x — acima de 30x)")
                elif pl < 0: alertas.append(f"P/L negativo ({pl:.1f}x) — empresa com prejuízo")

                if dy > 5: pontos_pos.append(f"Dividend Yield elevado ({dy:.1f}%)")

                if cr > 10: pontos_pos.append(f"Crescimento de receita forte ({cr:.1f}% a.a.)")
                elif cr < 0: pontos_neg.append(f"Receita em queda ({cr:.1f}% a.a.)")

                if ml > 15: pontos_pos.append(f"Margem líquida saudável ({ml:.1f}%)")
                elif 0 <= ml < 5: alertas.append(f"Margem líquida comprimida ({ml:.1f}%)")
                elif ml < 0: pontos_neg.append(f"Margem negativa ({ml:.1f}%)")

            n_pos = len(pontos_pos)
            n_neg = len(pontos_neg)
            if n_pos >= 3 or (n_pos > n_neg and n_pos >= 2):
                veredicto = ("ATRATIVO", "#00ff87")
            elif n_neg >= 3 or (n_neg > n_pos and n_neg >= 2):
                veredicto = ("FRACO", "#ff3d5a")
            else:
                veredicto = ("NEUTRO", "#ffd600")

            pos_html = "".join([f'<li style="color:#00ff87;margin-bottom:4px;line-height:1.5;">{p}</li>' for p in pontos_pos])
            neg_html = "".join([f'<li style="color:#ff3d5a;margin-bottom:4px;line-height:1.5;">{p}</li>' for p in pontos_neg])
            ale_html = "".join([f'<li style="color:#ffd600;margin-bottom:4px;line-height:1.5;">{p}</li>' for p in alertas])
            sem_dados = '<li style="color:#64748b;">Dados de peers insuficientes para análise completa.</li>' if not pontos_pos and not pontos_neg and not alertas else ""

            sintese_items.append(f"""
<div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1px solid #1e293b;border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;flex-wrap:wrap;gap:0.5rem;">
    <div>
      <span style="font-family:'JetBrains Mono',monospace;font-weight:800;color:#00d2ff;font-size:1.05rem;">{t}</span>
      <span style="font-size:0.78rem;color:#64748b;margin-left:0.5rem;">{nome}</span>
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;">
      <span style="font-size:0.65rem;color:#475569;font-style:italic;">{fonte_label}</span>
      <span style="background:rgba(0,0,0,0.3);border:1px solid {veredicto[1]}40;border-radius:6px;padding:0.2rem 0.75rem;font-size:0.72rem;font-weight:800;color:{veredicto[1]};letter-spacing:0.08em;">{veredicto[0]}</span>
    </div>
  </div>
  <ul style="margin:0;padding-left:1.2rem;font-size:0.82rem;list-style:disc;">
    {pos_html}{neg_html}{ale_html}{sem_dados}
  </ul>
</div>
""")

        if sintese_items:
            st.markdown("".join(sintese_items), unsafe_allow_html=True)

        # ── Próximo Passo ────────────────────────────────────────────────────
        tickers_str = ", ".join(tickers[:3]) + ("..." if len(tickers) > 3 else "")
        st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(0,255,135,0.06),rgba(0,210,255,0.03));border:1px solid rgba(0,255,135,0.25);border-radius:14px;padding:1.2rem 1.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-top:0.5rem;">
  <div>
    <div style="font-size:0.72rem;color:#64748b;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.3rem;">Próximo Passo</div>
    <div style="font-size:0.95rem;font-weight:700;color:#f8fafc;">Abra <span style="color:#00ff87">Portfolio</span> na barra lateral</div>
    <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.2rem;">Monte e otimize a carteira com {tickers_str}</div>
  </div>
  <div style="font-size:1.8rem;opacity:0.6;">→</div>
</div>
""", unsafe_allow_html=True)

    except OSError as e:
        st.cache_data.clear()
        st.error(f"Erro de I/O ao buscar dados. O cache foi limpo automaticamente.")
        st.caption(f"Detalhe técnico: {e}")
        if st.button("🔄 Tentar novamente", key="retry_os_error"):
            st.rerun()
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        st.caption(f"```\n{traceback.format_exc()}\n```")






