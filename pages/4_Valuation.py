import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import db as _db
from utils.charts import apply_plotly_theme
from utils.ui import load_css, loading_overlay
from utils.valuation import (
    calc_cv,
    calc_dcf,
    compute_roic_series,
    compute_year_metrics,
)
from utils.home_data import MULTIPLES_CFG, compute_sector_ranking, get_sector_peers
from utils.home_render import color_pct, color_veredicto, render_hist_section
from utils.market_data import clean_numeric_column

load_css()

# ─── Constants ────────────────────────────────────────────────────────────────
ERP_MATURE = 5.0  # ERP de mercado maduro (EUA). Rf=Selic já embute o risco-país
# (juros nominais brasileiros carregam prêmio de inflação/risco doméstico via a
# reação do BCB), então somar um CRP à parte aqui contaria o risco Brasil duas
# vezes no ke. Ver Damodaran: quando Rf é a taxa local (não o Treasury), usa-se
# apenas o ERP do mercado maduro, sem CRP adicional.


# ─── Shared helpers ────────────────────────────────────────────────────────────
def _fmt(v, prefix="R$"):
    if v is None:
        return "—"
    s = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e9:
        return f"{s}{prefix} {a / 1e9:,.2f} bi"
    if a >= 1e6:
        return f"{s}{prefix} {a / 1e6:,.1f} mi"
    return f"{s}{prefix} {a:,.0f}"


def _pct(v):
    return f"{v:.1f}%" if v is not None else "—"


def _card(label, value, vc="#f8fafc", badge_text="", badge_style="", tip=""):
    t = f' title="{tip}"' if tip else ""
    bdg = (
        f'<div style="font-size:0.65rem;font-weight:700;font-family:monospace;'
        f'padding:1px 7px;border-radius:4px;margin-top:4px;{badge_style}">{badge_text}</div>'
        if badge_text
        else ""
    )
    return (
        f'<div class="mcard"{t}>'
        f'<div class="mcard-label">{label}</div>'
        f'<div class="mcard-value" style="color:{vc}">{value}</div>'
        f"{bdg}</div>"
    )


def _green_badge(txt):
    return (
        txt,
        "color:#00ff87;background:rgba(0,255,135,0.08);border:1px solid rgba(0,255,135,0.2)",
    )


def _red_badge(txt):
    return (
        txt,
        "color:#ff3d5a;background:rgba(255,61,90,0.08);border:1px solid rgba(255,61,90,0.2)",
    )


def _neutral_badge(txt):
    return (
        txt,
        "color:#94a3b8;background:rgba(148,163,184,0.08);border:1px solid rgba(148,163,184,0.15)",
    )


# ─── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_selic():
    try:
        from bcb import sgs

        # Série 432 (BCB) = Meta Selic definida pelo Copom, já em % a.a. — não é taxa
        # diária, então não deve ser reanualizada com (1+r)^252.
        taxa = sgs.get(432, start=dt.date.today() - dt.timedelta(days=30))
        return round(taxa.iloc[-1, 0] / 100, 4)
    except Exception:
        return 0.105


@st.cache_data(ttl=3600, show_spinner=False)
def get_koller_data(ticker_b3: str):
    try:
        t = yf.Ticker(f"{ticker_b3}.SA")
        inc = t.financials
        cf = t.cashflow
        bs = t.balance_sheet
        info = t.info or {}

        def _v(df, keys, col_i=0):
            if df is None or df.empty or col_i >= len(df.columns):
                return None
            for k in keys:
                if k in df.index:
                    try:
                        v = pd.to_numeric(df.loc[k].iloc[col_i], errors="coerce")
                        if not pd.isna(v):
                            return float(v)
                    except Exception:
                        continue
            return None

        if inc is None or inc.empty:
            return None

        n = min(len(inc.columns), 4)
        years = []
        for i in range(n):
            date = inc.columns[i]
            ylbl = str(date.year) if hasattr(date, "year") else str(date)[:4]
            rev = _v(inc, ["Total Revenue", "Revenue"], i)
            ebit = _v(inc, ["EBIT", "Operating Income"], i)
            pretx = _v(inc, ["Pretax Income"], i)
            taxex = _v(inc, ["Tax Provision", "Income Tax Expense"], i)
            intr = _v(inc, ["Interest Expense", "Interest Expense Non Operating"], i)
            da = _v(cf, ["Depreciation And Amortization", "Depreciation"], i)
            capex = _v(cf, ["Capital Expenditure"], i)
            dwc = _v(cf, ["Change In Working Capital"], i)
            ppe = _v(bs, ["Net PPE", "Net Property Plant And Equipment"], i)
            gw = _v(bs, ["Goodwill And Other Intangible Assets", "Goodwill"], i) or 0.0
            ca = _v(bs, ["Current Assets", "Total Current Assets"], i)
            cl = _v(bs, ["Current Liabilities", "Total Current Liabilities"], i)
            cash = _v(
                bs,
                [
                    "Cash And Cash Equivalents",
                    "Cash Cash Equivalents And Short Term Investments",
                ],
                i,
            )
            dbt_s = (
                _v(bs, ["Current Debt", "Current Portion Of Long Term Debt"], i) or 0.0
            )
            dbt_l = (
                _v(
                    bs,
                    ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
                    i,
                )
                or 0.0
            )
            equity = _v(
                bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"], i
            )

            if ebit is None or rev is None:
                continue

            metrics = compute_year_metrics(
                revenue=rev,
                ebit=ebit,
                pretax_income=pretx,
                tax_expense=taxex,
                da=da,
                capex=capex,
                delta_wc=dwc,
                ppe=ppe,
                goodwill=gw,
                current_assets=ca,
                current_liabilities=cl,
                cash=cash,
                debt_short=dbt_s,
                debt_long=dbt_l,
                equity=equity,
                interest_expense=intr,
            )
            years.append({"year": ylbl, "revenue": rev, "ebit": ebit, **metrics})

        if not years:
            return None

        ys = compute_roic_series(sorted(years, key=lambda x: x["year"]))

        shares = (
            info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 1
        )
        beta = float(info.get("beta") or 1.0)
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        lat = ys[-1]
        kd_est = (
            abs(lat["interest"]) / lat["debt"] * 100
            if lat.get("interest") and lat["debt"] > 0
            else None
        )

        # Implied growth (revenue CAGR)
        rev_cagr = None
        if len(ys) >= 2 and ys[0]["revenue"] and ys[-1]["revenue"]:
            n_yrs = len(ys) - 1
            rev_cagr = ((ys[-1]["revenue"] / ys[0]["revenue"]) ** (1 / n_yrs) - 1) * 100

        return {
            "years": ys,
            "latest": lat,
            "shares": shares,
            "beta": beta,
            "price": price,
            "total_debt": lat["debt"],
            "cash": lat["cash"],
            "equity_book": lat["equity"],
            "kd_est": kd_est,
            "rev_cagr": rev_cagr,
        }
    except Exception as e:
        return {"_error": str(e)}


# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-hero" style="border-left-color:#a855f7">
  <div class="page-hero-icon">
    <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 64 64" fill="none">
      <rect x="4" y="4" width="56" height="56" rx="14" fill="#0e1726"/>
      <line x1="32" y1="10" x2="32" y2="54" stroke="#a855f7" stroke-width="3" stroke-linecap="round"/>
      <path d="M44 18H26a8 8 0 0 0 0 16h12a8 8 0 0 1 0 16H18"
            stroke="#a855f7" stroke-width="3" stroke-linecap="round" fill="none"/>
    </svg>
  </div>
  <div class="page-hero-content">
    <h1 class="page-hero-title" style="background:linear-gradient(135deg,#f8fafc 40%,#a855f7 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
      Valuation — McKinsey / Koller
    </h1>
    <p class="page-hero-subtitle">
      Enterprise DCF completo em 8 etapas: NOPLAT → Invested Capital → ROIC histórico →
      Projeção → Continuing Value → WACC/CAPM → Enterprise Value → Validação por múltiplos.
    </p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Ticker input ──────────────────────────────────────────────────────────────
col_t, col_hint = st.columns([2, 5])
with col_t:
    b3_stocks = sorted(pd.read_csv("acoes-listadas-b3.csv")["Ticker"].tolist())
    defaults = st.session_state.get("selected_tickers", [])
    default_ticker = defaults[0] if defaults else None
    default_idx = (
        b3_stocks.index(default_ticker) + 1 if default_ticker in b3_stocks else 0
    )
    ticker = (
        st.selectbox(
            "Ticker B3",
            options=[""] + b3_stocks,
            index=default_idx,
            format_func=lambda t: "Selecione um ticker..." if t == "" else t,
            help="Dados via yfinance (4 anos) + Selic BCB.",
        )
        .strip()
        .upper()
    )

if not ticker:
    with col_hint:
        st.info(
            "Digite um ticker B3 para iniciar o valuation completo (metodologia Koller)."
        )
    st.stop()

# ─── Load data ─────────────────────────────────────────────────────────────────
with loading_overlay(f"Carregando dados financeiros de {ticker}...", tickers=[ticker]):
    selic = get_selic()
    kdata = get_koller_data(ticker)

if kdata is None or "_error" in (kdata or {}):
    err_detail = kdata.get("_error", "") if isinstance(kdata, dict) else ""
    st.error(
        f"Não foi possível carregar dados do yfinance para **{ticker}.SA**.\n\n"
        f"**Causas comuns:** ticker inválido, empresa sem demonstrações financeiras "
        f"no yfinance (bancos, FIIs, BDRs), ou instabilidade temporária da API.\n\n"
        + (
            f"_Detalhe técnico: {err_detail}_"
            if err_detail
            else "Tente PETR4, WEGE3, VALE3, RENT3 ou outro ticker de empresa operacional."
        )
    )
    st.stop()

ys = kdata["years"]
latest = kdata["latest"]
beta = kdata["beta"]
price = kdata["price"]
shares = kdata["shares"]
t_debt = kdata["total_debt"]
cash_v = kdata["cash"]
kd_est = kdata["kd_est"]
cagr = kdata["rev_cagr"]

# ─── Session state (persists slider values across tabs) ────────────────────────
skey = f"kval_{ticker}"
if skey not in st.session_state:
    _b = max(min(beta, 2.5), 0.3)
    _ke = round(min(selic * 100 + _b * ERP_MATURE, 22.0), 1)
    _kd = round(kd_est, 1) if kd_est else 8.0
    _mkt_equity = price * shares if price and shares else 0
    _equity_for_weight = _mkt_equity if _mkt_equity > 0 else latest["equity"]
    _ev0 = max(_equity_for_weight + t_debt - cash_v, 1)
    _ew = round(max(min(_equity_for_weight / _ev0, 0.95), 0.3), 2)
    _roic_hist = next((y["roic"] for y in reversed(ys) if y.get("roic")), 12.0) or 12.0
    _g1 = round(min(max(cagr or 5.0, 0.0), 20.0), 1)
    st.session_state[skey] = {
        "ke": _ke,
        "kd": _kd,
        "e_weight": _ew,
        "g1": _g1,
        "g2": round(_g1 * 0.5, 1),
        "gt": 3.5,
        "roic_cv": round(min(_roic_hist, 20.0), 1),
        "roic_proj": round(min(_roic_hist, 30.0), 1),
        "model": "Enterprise DCF",
    }
ss = st.session_state[skey]

# ─── WACC (live, derived from sliders) ─────────────────────────────────────────
tax_rate = latest["tax_rate"]
wacc_dec = ss["ke"] / 100 * ss["e_weight"] + ss["kd"] / 100 * (1 - ss["e_weight"]) * (
    1 - tax_rate
)

# ─── 8 Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(
    [
        "1 · Modelo",
        "2 · Histórico",
        "3 · Drivers",
        "4 · Projeção",
        "5 · Valor Terminal",
        "6 · WACC",
        "7 · Resultados",
        "8 · Múltiplos",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Modelo
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("#### Escolha o modelo de valuation adequado")

    model_data = [
        (
            "Enterprise DCF com WACC",
            "Estrutura de capital estável (D/V constante)",
            "#00ff87",
            "✅ Este modelo",
        ),
        (
            "APV — Adjusted Present Value",
            "Estrutura de capital muda significativamente",
            "#00d2ff",
            "Alternativo",
        ),
        (
            "Economic Profit",
            "Foco em destruição/criação de valor por período",
            "#ffd600",
            "Check interpretativo",
        ),
        (
            "Equity Cash Flow",
            "Banco, seguradora ou financeira",
            "#f87171",
            "Para financeiras",
        ),
        (
            "Múltiplos de Mercado",
            "Checagem rápida ou sem projeções detalhadas",
            "#94a3b8",
            "Validação",
        ),
    ]

    for name, when, color, badge_txt in model_data:
        is_sel = name.startswith("Enterprise")
        border = (
            f"border-left:3px solid {color}"
            if is_sel
            else "border-left:3px solid #1e293b"
        )
        bg = "rgba(0,255,135,0.03)" if is_sel else "transparent"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:0.7rem 1rem;margin-bottom:0.4rem;border-radius:8px;{border};background:{bg};">'
            f'<div><div style="font-weight:700;color:#f8fafc;font-size:0.9rem">{name}</div>'
            f'<div style="color:#64748b;font-size:0.78rem;margin-top:2px">{when}</div></div>'
            f'<div style="font-size:0.7rem;font-weight:700;color:{color};'
            f"background:{color}18;border:1px solid {color}44;"
            f'padding:2px 10px;border-radius:20px;white-space:nowrap">{badge_txt}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.info(
        "**Este modelo usa Enterprise DCF** como primary + Economic Profit como check interpretativo. "
        "O princípio fundamental: **Valor = f(ROIC, Crescimento, Custo de Capital)**. "
        "Crescimento só cria valor quando ROIC > WACC."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Histórico
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("#### Análise Histórica — NOPLAT, Invested Capital, ROIC, FCF")

    if len(ys) < 2:
        st.warning("Dados históricos insuficientes no yfinance para este ticker.")
    else:
        # Table
        rows_hist = []
        for y in ys:
            rows_hist.append(
                {
                    "Ano": y["year"],
                    "Receita": _fmt(y["revenue"]),
                    "EBIT": _fmt(y["ebit"]),
                    "Tax (%)": f"{y['tax_rate'] * 100:.0f}%",
                    "NOPLAT": _fmt(y["noplat"]),
                    "D&A": _fmt(y["da"]),
                    "CapEx": _fmt(-y["capex"]),
                    "ΔWC": _fmt(y["delta_wc"]),
                    "FCF": _fmt(y["fcf"]),
                    "IC": _fmt(y["ic"]),
                    "ROIC (%)": _pct(y.get("roic")),
                }
            )

        df_hist = pd.DataFrame(rows_hist).set_index("Ano")
        st.dataframe(df_hist, use_container_width=True)

        # ROIC & FCF charts side by side
        col_r, col_f = st.columns(2)

        with col_r:
            roic_vals = [y.get("roic") for y in ys]
            years_lbl = [y["year"] for y in ys]
            wacc_pct = wacc_dec * 100

            fig_roic = go.Figure()
            fig_roic.add_trace(
                go.Bar(
                    x=years_lbl,
                    y=roic_vals,
                    marker_color=[
                        "#00ff87" if (r and r > wacc_pct) else "#ff3d5a"
                        for r in roic_vals
                    ],
                    name="ROIC (%)",
                    text=[f"{r:.1f}%" if r else "—" for r in roic_vals],
                    textposition="outside",
                )
            )
            fig_roic.add_hline(
                y=wacc_pct,
                line_dash="dash",
                line_color="#ffd600",
                annotation_text=f"WACC {wacc_pct:.1f}%",
                annotation_position="bottom right",
            )
            fig_roic.update_layout(
                title="ROIC histórico vs WACC", height=280, yaxis_title="ROIC (%)"
            )
            apply_plotly_theme(fig_roic)
            st.plotly_chart(fig_roic, use_container_width=True)

        with col_f:
            fcf_vals = [y["fcf"] / 1e6 for y in ys]
            fig_fcf = go.Figure(
                go.Bar(
                    x=years_lbl,
                    y=fcf_vals,
                    marker_color=["#00ff87" if v >= 0 else "#ff3d5a" for v in fcf_vals],
                    text=[f"R$ {v:,.0f}M" for v in fcf_vals],
                    textposition="outside",
                )
            )
            fig_fcf.update_layout(
                title="Free Cash Flow histórico (R$ mi)",
                height=280,
                yaxis_title="FCF (R$ mi)",
            )
            apply_plotly_theme(fig_fcf)
            st.plotly_chart(fig_fcf, use_container_width=True)

        # IC breakdown for latest year
        st.markdown("**Composição do Invested Capital — ano mais recente**")
        lat = ys[-1]
        ic_cards = (
            _card(
                "WCO",
                _fmt(lat["wco"]),
                "#00d2ff",
                tip="Working Capital Operacional: Recebíveis + Estoques − Fornecedores (excl. caixa excedente e dívida financeira)",
            )
            + _card(
                "PP&E Líq.",
                _fmt(lat["ppe"]),
                "#f8fafc",
                tip="Ativo imobilizado líquido de depreciação",
            )
            + _card(
                "Goodwill + Intang.",
                _fmt(lat["goodwill"]),
                "#ffd600",
                tip="Goodwill e intangíveis adquiridos",
            )
            + _card(
                "IC Total",
                _fmt(lat["ic"]),
                "#00ff87",
                tip="Invested Capital = WCO + PP&E + Goodwill",
            )
        )
        st.markdown(f'<div class="mcard-grid">{ic_cards}</div>', unsafe_allow_html=True)

        # Histórico Fundamentalista (Receita, Lucro, Margens, ROE)
        st.markdown("---")
        st.markdown("**Evolução Anual — Receita, Lucro, Margens e ROE**")
        st.caption(
            "Últimos 4 anos (fonte: yfinance / relatórios anuais). "
            "Complementa o NOPLAT/ROIC acima com a visão contábil tradicional."
        )
        render_hist_section(ticker)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Drivers de Valor
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("#### Drivers de Valor — Calibração com Evidências Empíricas")

    roic_vals_clean = [y["roic"] for y in ys if y.get("roic") is not None]
    roic_med = float(np.median(roic_vals_clean)) if roic_vals_clean else None
    wacc_pct = wacc_dec * 100

    # Value spread
    spread_cards_html = ""
    if roic_med is not None:
        spread = roic_med - wacc_pct
        spread_lbl, spread_style = (
            _green_badge(f"ROIC > WACC: +{spread:.1f}pp — CRIA VALOR")
            if spread > 0
            else _red_badge(f"ROIC < WACC: {spread:.1f}pp — DESTRÓI VALOR")
        )
        spread_cards_html += _card(
            "ROIC Mediano",
            _pct(roic_med),
            "#00ff87" if roic_med > wacc_pct else "#ff3d5a",
            badge_text=spread_lbl,
            badge_style=spread_style,
        )
    spread_cards_html += _card("WACC Estimado", _pct(wacc_pct), "#ffd600")
    spread_cards_html += _card(
        "CAGR Receita",
        _pct(cagr),
        "#00d2ff",
        tip="Taxa de crescimento anual composta da receita histórica",
    )
    st.markdown(
        f'<div class="mcard-grid">{spread_cards_html}</div>', unsafe_allow_html=True
    )

    # Benchmarks table
    st.markdown("**Benchmarks Empíricos (Koller):**")
    benchmarks = [
        (
            "ROIC mediano — mercado amplo",
            "~10–12%",
            "Referência de médio prazo para qualquer setor",
        ),
        (
            "ROIC de alto desempenho",
            "Declina para ~15% em 15 anos",
            "Mean reversion: vantagem competitiva se deteriora com tempo",
        ),
        (
            "Crescimento real mediano (40 anos)",
            "~6,3% real / ~10,2% nominal",
            "Ponto de partida empírico para projeções de crescimento",
        ),
        (
            "Crescimento > 20% real",
            "Cai para ~8% em 5 anos",
            "Empresas de alto crescimento raramente sustentam por mais de 5 anos",
        ),
        (
            "Fortune 50",
            "~1% real após ingresso no ranking",
            "Grandes empresas maduras crescem próximo ao PIB",
        ),
    ]
    for label, val, note in benchmarks:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0;'
            f'border-bottom:1px solid #1e293b;">'
            f'<div><span style="color:#f8fafc;font-size:0.85rem">{label}</span>'
            f'<div style="color:#64748b;font-size:0.72rem">{note}</div></div>'
            f'<span style="color:#00d2ff;font-family:monospace;font-size:0.85rem;'
            f'font-weight:700;white-space:nowrap;margin-left:1rem">{val}</span></div>',
            unsafe_allow_html=True,
        )

    st.warning(
        "**Regra de ouro:** Seja cético com crescimento > 10% por mais de 5 anos, "
        "a menos que haja justificativa estratégica sólida."
    )

    # Economic Profit (check interpretativo)
    if roic_med is not None and latest["ic"] > 0:
        ep = (roic_med - wacc_pct) / 100 * latest["ic"]
        ep_ps = ep / shares if shares > 0 else None
        st.markdown("**Economic Profit (check Koller) — ano mais recente:**")
        ep_cards = _card(
            "Economic Profit",
            _fmt(ep),
            "#00ff87" if ep > 0 else "#ff3d5a",
            tip="EP = (ROIC − WACC) × IC. Positivo = cria valor; negativo = destrói.",
        ) + (
            _card(
                "EP / Ação",
                f"R$ {ep_ps:,.2f}",
                "#00ff87" if ep_ps and ep_ps > 0 else "#ff3d5a",
            )
            if ep_ps
            else ""
        )
        st.markdown(f'<div class="mcard-grid">{ep_cards}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Projeção
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("#### Projeção de Performance — 10 Anos")

    col_sl1, col_sl2, col_sl3 = st.columns(3)
    with col_sl1:
        g1 = st.slider(
            "Crescimento Anos 1–5 (%)",
            0.0,
            30.0,
            value=float(ss["g1"]),
            step=0.5,
            key=f"{skey}_g1",
            help=f"CAGR histórico da receita: {_pct(cagr)}. Seja conservador.",
        )
        ss["g1"] = g1

    with col_sl2:
        g2 = st.slider(
            "Crescimento Anos 6–10 (%)",
            0.0,
            20.0,
            value=float(ss["g2"]),
            step=0.5,
            key=f"{skey}_g2",
            help="Fase de desaceleração — costuma ser metade do crescimento da fase 1.",
        )
        ss["g2"] = g2

    with col_sl3:
        roic_proj = st.slider(
            "ROIC — Período de Projeção (%)",
            1.0,
            40.0,
            value=float(ss["roic_proj"]),
            step=0.5,
            key=f"{skey}_roic_proj",
            help="ROIC marginal usado para calcular o reinvestimento (g/ROIC) nos anos 1–10. "
            "Pode diferir do ROIC na perpetuidade, definido na aba Valor Terminal.",
        )
        ss["roic_proj"] = roic_proj

    noplat0 = latest["noplat"]
    roic_cv = ss["roic_cv"]

    if noplat0 <= 0:
        st.error(
            "NOPLAT negativo — não é possível projetar o DCF. "
            "Verifique se o EBIT é positivo nos dados históricos."
        )
    else:
        _, _, _, proj_rows, _, _ = calc_dcf(
            noplat0, g1, g2, ss["gt"], wacc_dec, roic_cv, roic_proj
        )

        df_proj = pd.DataFrame(
            [
                {
                    "Ano": f"T+{r['t']}",
                    "g (%)": f"{r['g_pct']:.1f}%",
                    "NOPLAT": _fmt(r["noplat"]),
                    "Reinv. (%)": f"{r['reinv_pct']:.1f}%",
                    "FCF": _fmt(r["fcf"]),
                    "PV (FCF)": _fmt(r["pv"]),
                }
                for r in proj_rows
            ]
        )
        df_proj = df_proj.set_index("Ano")
        st.dataframe(df_proj, use_container_width=True)

        # Waterfall chart
        fig_wf = go.Figure(
            go.Bar(
                x=[f"T+{r['t']}" for r in proj_rows],
                y=[r["fcf"] / 1e6 for r in proj_rows],
                marker_color=[
                    "#00ff87" if r["fcf"] >= 0 else "#ff3d5a" for r in proj_rows
                ],
            )
        )
        fig_wf.update_layout(
            title="FCF Projetado (R$ mi)", height=240, yaxis_title="FCF (R$ mi)"
        )
        apply_plotly_theme(fig_wf)
        st.plotly_chart(fig_wf, use_container_width=True)

        st.caption(
            f"NOPLAT base: {_fmt(noplat0)} | "
            f"Reinvestimento = g / ROIC_proj ({roic_proj:.1f}%) por período"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Valor Terminal
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("#### Valor Terminal — Continuing Value (Gordon Growth adaptado)")

    st.markdown("""
```
CV_T = NOPLAT_{T+1} × (1 − g / ROIC_cv) / (WACC − g)
```
""")

    col_cv1, col_cv2 = st.columns(2)
    with col_cv1:
        gt = st.slider(
            "Crescimento Terminal — g (%)",
            1.0,
            6.0,
            value=float(ss["gt"]),
            step=0.25,
            key=f"{skey}_gt",
            help="Taxa de crescimento na perpetuidade. PIB nominal BR ≈ 3–4%.",
        )
        ss["gt"] = gt

    with col_cv2:
        roic_cv = st.slider(
            "ROIC na Perpetuidade — ROIC_cv (%)",
            5.0,
            35.0,
            value=float(ss["roic_cv"]),
            step=0.5,
            key=f"{skey}_roic_cv",
            help="Use WACC se não há moat sustentável; ROIC histórico se há vantagem competitiva.",
        )
        ss["roic_cv"] = roic_cv

    if latest["noplat"] > 0:
        noplat0 = latest["noplat"]
        noplat_11 = (
            noplat0
            * (1 + ss["g1"] / 100) ** 5
            * (1 + ss["g2"] / 100) ** 5
            * (1 + gt / 100)
        )
        cv = calc_cv(noplat_11, gt, roic_cv, wacc_dec)

        if cv is not None:
            pv_exp, pv_cv, ev_total, _, _, _ = calc_dcf(
                noplat0, ss["g1"], ss["g2"], gt, wacc_dec, roic_cv, ss["roic_proj"]
            )
            cv_pct = pv_cv / ev_total * 100 if ev_total > 0 else 0

            cv_cards = (
                _card(
                    "NOPLAT T+11",
                    _fmt(noplat_11),
                    "#f8fafc",
                    tip="NOPLAT do primeiro ano após o horizonte explícito",
                )
                + _card(
                    "Continuing Value",
                    _fmt(cv),
                    "#a855f7",
                    tip="CV não descontado — valor da perpetuidade",
                )
                + _card(
                    "PV(CV)",
                    _fmt(pv_cv),
                    "#a855f7",
                    tip="Valor presente do Continuing Value",
                )
                + _card(
                    "CV / EV Total",
                    f"{cv_pct:.0f}%",
                    "#ffd600" if cv_pct > 80 else "#f8fafc",
                    tip="CV geralmente representa 60–80% do Enterprise Value. Acima de 80% requer atenção extra.",
                )
            )
            st.markdown(
                f'<div class="mcard-grid">{cv_cards}</div>', unsafe_allow_html=True
            )

            if cv_pct > 80:
                st.warning(
                    f"⚠️ CV representa **{cv_pct:.0f}%** do Enterprise Value — "
                    f"alta sensibilidade às premissas de `g` e `ROIC_cv`. "
                    f"Faça análise de sensibilidade na aba Resultados."
                )

            # Fórmula detalhada
            reinv_tv = min(gt / 100 / (roic_cv / 100), 0.99) if roic_cv > 0 else 0
            fcf_mult_tv = max(1 - reinv_tv, 0.01)
            st.markdown(
                f'<div style="font-size:0.78rem;color:#64748b;padding:0.5rem 0;line-height:1.8">'
                f'<b style="color:#94a3b8">Fórmula:</b><br>'
                f"NOPLAT_{{T+11}} = {_fmt(noplat_11)}<br>"
                f"Taxa de reinvestimento = g / ROIC_cv = {gt:.1f}% / {roic_cv:.1f}% = {reinv_tv * 100:.0f}%<br>"
                f"FCF_mult = 1 − {reinv_tv * 100:.0f}% = {fcf_mult_tv * 100:.0f}%<br>"
                f"CV = {_fmt(noplat_11)} × {fcf_mult_tv * 100:.0f}% / ({wacc_dec * 100:.1f}% − {gt:.1f}%) = {_fmt(cv)}"
                f"</div>",
                unsafe_allow_html=True,
            )

            if abs(roic_cv - wacc_dec * 100) < 1:
                st.info(
                    "Quando ROIC_cv ≈ WACC, o crescimento não cria valor adicional — "
                    "use a fórmula simplificada: CV = NOPLAT_{T+11} / WACC."
                )
        else:
            st.error(
                "WACC ≤ g terminal — modelo indefinido. Aumente o WACC ou reduza g."
            )
    else:
        st.error("NOPLAT negativo — não é possível calcular o Continuing Value.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — WACC
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("#### Custo de Capital — WACC / CAPM")

    col_ke, col_kd = st.columns(2)

    with col_ke:
        st.markdown("**6a. Custo do Equity — CAPM**")
        rf_pct = selic * 100
        _beta_c = max(min(beta, 2.5), 0.3)

        st.markdown(
            f'<div style="font-size:0.8rem;color:#94a3b8;padding:0.4rem 0;">'
            f'Rf (Selic): <b style="color:#00d2ff">{rf_pct:.2f}%</b> · '
            f'β (yfinance): <b style="color:#f8fafc">{beta:.2f}</b> · '
            f'ERP (mercado maduro, s/ CRP): <b style="color:#ffd600">{ERP_MATURE:.1f}%</b><br>'
            f"ke = Rf + β × ERP = {rf_pct:.1f}% + {_beta_c:.2f} × {ERP_MATURE:.0f}% ≈ "
            f'<b style="color:#00ff87">{rf_pct + _beta_c * ERP_MATURE:.1f}%</b>'
            f"</div>",
            unsafe_allow_html=True,
        )

        ke = st.slider(
            "ke — Custo do Equity (%)",
            6.0,
            25.0,
            value=float(ss["ke"]),
            step=0.25,
            key=f"{skey}_ke",
            help="CAPM: Rf + β × ERP. Rf=Selic já embute o risco-país; ERP é o "
            "prêmio de mercado maduro (~5%), sem CRP adicional (evita dupla contagem).",
        )
        ss["ke"] = ke

    with col_kd:
        st.markdown("**6b. Custo da Dívida**")
        if kd_est:
            st.markdown(
                f'<div style="font-size:0.8rem;color:#94a3b8;padding:0.4rem 0;">'
                f'Estimado: Juros / Dívida = <b style="color:#00d2ff">{kd_est:.1f}%</b> a.a.'
                f"</div>",
                unsafe_allow_html=True,
            )

        kd = st.slider(
            "kd — Custo da Dívida (%)",
            4.0,
            20.0,
            value=float(ss["kd"]),
            step=0.25,
            key=f"{skey}_kd",
            help="Yield to maturity da dívida de longo prazo. Estimado de Juros/Dívida.",
        )
        ss["kd"] = kd

    st.markdown("**6c. Estrutura de Capital (pesos de mercado)**")
    e_weight = (
        st.slider(
            "Participação do Equity — E/(E+D) (%)",
            20.0,
            95.0,
            value=float(ss["e_weight"] * 100),
            step=1.0,
            key=f"{skey}_ew",
            help="Use valor de mercado, não contábil. "
            f"Estimado: {ss['e_weight'] * 100:.0f}%",
        )
        / 100
    )
    ss["e_weight"] = e_weight
    d_weight = 1 - e_weight

    wacc_dec = ke / 100 * e_weight + kd / 100 * d_weight * (1 - tax_rate)
    wacc_pct = wacc_dec * 100

    st.markdown("**WACC resultante:**")
    wacc_cards = (
        _card("ke (Equity)", _pct(ke), "#00d2ff")
        + _card("kd × (1−t) (Dívida)", _pct(kd * (1 - tax_rate)), "#f87171")
        + _card("E/V", f"{e_weight * 100:.0f}%", "#f8fafc")
        + _card(
            "WACC",
            _pct(wacc_pct),
            "#ffd600",
            tip=f"WACC = {ke:.1f}%×{e_weight * 100:.0f}% + {kd:.1f}%×{d_weight * 100:.0f}%×(1−{tax_rate * 100:.0f}%)",
        )
    )
    st.markdown(f'<div class="mcard-grid">{wacc_cards}</div>', unsafe_allow_html=True)

    # WACC waterfall
    fig_ww = go.Figure(
        go.Bar(
            x=["ke × E/V", "kd × (1−t) × D/V", "WACC"],
            y=[ke * e_weight, kd * (1 - tax_rate) * d_weight, wacc_pct],
            marker_color=["#00d2ff", "#f87171", "#ffd600"],
            text=[
                f"{v:.2f}%"
                for v in [ke * e_weight, kd * (1 - tax_rate) * d_weight, wacc_pct]
            ],
            textposition="outside",
        )
    )
    fig_ww.update_layout(title="Decomposição do WACC", height=260, yaxis_title="(%)")
    apply_plotly_theme(fig_ww)
    st.plotly_chart(fig_ww, use_container_width=True)

    st.caption(
        f"Alíquota efetiva (histórica): {tax_rate * 100:.0f}% · "
        f"Beta yfinance: {beta:.2f} · "
        f"Suavização Bloomberg: β_adj = 0,33 + 0,67 × {beta:.2f} = {0.33 + 0.67 * beta:.2f}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Resultados
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("#### Enterprise Value → Equity Value → Preço por Ação")

    noplat0 = latest["noplat"]

    if noplat0 <= 0:
        st.error("NOPLAT base negativo — não é possível calcular o Enterprise Value.")
    elif wacc_dec <= ss["gt"] / 100:
        st.error(
            "WACC ≤ g terminal — modelo indefinido. Ajuste os parâmetros na aba WACC."
        )
    else:
        pv_exp, pv_cv, ev, proj_rows, cv, noplat_11 = calc_dcf(
            noplat0, ss["g1"], ss["g2"], ss["gt"], wacc_dec, ss["roic_cv"], ss["roic_proj"]
        )

        net_debt = t_debt - cash_v
        equity_val = ev - net_debt
        price_iv = equity_val / shares if shares > 0 else None
        upside = (
            ((price_iv / price - 1) * 100) if price and price > 0 and price_iv else None
        )
        ms = (
            (1 - price / price_iv) * 100
            if price and price_iv and price_iv > 0
            else None
        )

        # Summary cards
        vc_up = "#00ff87" if (upside and upside > 0) else "#ff3d5a"
        verdict = (
            "SUBAVALIADO"
            if upside and upside > 15
            else "SOBREAVALIADO"
            if upside and upside < -15
            else "PRÓXIMO DO JUSTO"
        )
        vcolor = (
            "#00ff87"
            if verdict == "SUBAVALIADO"
            else ("#ff3d5a" if verdict == "SOBREAVALIADO" else "#ffd600")
        )

        res_cards = (
            _card(
                "PV (FCFs Explic.)",
                _fmt(pv_exp),
                "#00d2ff",
                tip="Soma do valor presente dos FCFs projetados (anos 1–10)",
            )
            + _card(
                "PV (Continuing Value)",
                _fmt(pv_cv),
                "#a855f7",
                tip="Valor presente do Continuing Value (perpetuidade)",
            )
            + _card(
                "Enterprise Value",
                _fmt(ev),
                "#ffd600",
                tip="Enterprise Value = PV(FCFs) + PV(CV)",
            )
            + _card(
                "(−) Dívida Líquida",
                _fmt(-net_debt),
                "#f87171",
                tip="Dívida total − Caixa. Ponte firma → ação.",
            )
            + _card(
                "Equity Value",
                _fmt(equity_val),
                "#00ff87",
                tip="Equity Value = Enterprise Value − Dívida Líquida",
            )
        )
        if price_iv:
            res_cards += (
                _card("IV por Ação", f"R$ {price_iv:,.2f}", "#00ff87")
                + _card("Preço Atual", f"R$ {price:,.2f}", "#94a3b8")
                + _card(
                    "Potencial",
                    f"{upside:+.1f}%",
                    vc_up,
                    badge_text=verdict,
                    badge_style=f"color:{vcolor};background:{vcolor}18;border:1px solid {vcolor}44",
                )
                + _card(
                    "Margem de Seg.",
                    f"{ms:.1f}%" if ms else "—",
                    "#00ff87"
                    if ms and ms > 20
                    else ("#ff3d5a" if ms and ms < 0 else "#ffd600"),
                )
            )
        st.markdown(
            f'<div class="mcard-grid">{res_cards}</div>', unsafe_allow_html=True
        )

        # Price vs IV bar
        if price_iv and price > 0:
            total_r = max(price_iv, price) * 1.15
            pp = min(price / total_r * 100, 100)
            ip = min(price_iv / total_r * 100, 100)
            bar_c = "#00ff87" if price_iv > price else "#ff3d5a"
            st.markdown(
                f"""
<div style="margin:0.8rem 0">
  <div style="font-size:0.68rem;color:#64748b;font-weight:600;text-transform:uppercase;
              letter-spacing:0.05em;margin-bottom:0.4rem">Preço Atual vs IV (Koller DCF)</div>
  <div style="position:relative;height:28px;background:#1e293b;border-radius:6px;overflow:hidden">
    <div style="position:absolute;top:0;left:0;width:{pp:.1f}%;height:100%;
                background:rgba(148,163,184,0.2);border-radius:6px 0 0 6px"></div>
    <div style="position:absolute;top:0;left:{ip:.1f}%;width:3px;height:100%;
                background:{bar_c};transform:translateX(-50%);box-shadow:0 0 8px {bar_c}88"></div>
    <div style="position:absolute;top:0;left:{pp:.1f}%;width:3px;height:100%;
                background:#94a3b8;transform:translateX(-50%)"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:0.3rem">
    <span style="font-size:0.72rem;color:#94a3b8">R$ {price:,.2f} (atual)</span>
    <span style="font-size:0.85rem;font-weight:800;color:{vcolor}">{verdict}</span>
    <span style="font-size:0.72rem;color:#a855f7">IV: R$ {price_iv:,.2f}</span>
  </div>
</div>""",
                unsafe_allow_html=True,
            )

        # CV proportion waterfall
        cv_pct = pv_cv / ev * 100 if ev > 0 else 0
        fig_bridge = go.Figure(
            go.Bar(
                x=["PV (FCFs Explícitos)", "PV (Continuing Value)", "Enterprise Value"],
                y=[pv_exp / 1e9, pv_cv / 1e9, ev / 1e9],
                marker_color=["#00d2ff", "#a855f7", "#ffd600"],
                text=[
                    f"R$ {v / 1e9:,.2f} bi\n({v / ev * 100:.0f}%)"
                    for v in [pv_exp, pv_cv, ev]
                ],
                textposition="outside",
            )
        )
        fig_bridge.update_layout(
            title=f"Composição do Enterprise Value (CV = {cv_pct:.0f}% do EV)",
            height=280,
            yaxis_title="R$ bilhões",
        )
        apply_plotly_theme(fig_bridge)
        st.plotly_chart(fig_bridge, use_container_width=True)

        # Sensitivity heatmap
        with st.expander(
            "📊 Sensibilidade — WACC × g terminal (Preço por Ação)", expanded=True
        ):
            st.caption(
                "CV representa ~60–80% do Enterprise Value. Sensibilidades em g e WACC são críticas. "
                "Verde = subavaliado vs. preço atual. Vermelho = sobreavaliado."
            )
            wacc_range = [
                w / 100
                for w in range(max(6, int(wacc_pct) - 4), min(22, int(wacc_pct) + 5))
            ]
            gt_range = [g / 10 for g in range(10, 61, 5)]  # 1.0 → 6.0%

            z_matrix, z_text = [], []
            for w in wacc_range:
                row_z, row_t = [], []
                for g in gt_range:
                    if w <= g / 100:
                        row_z.append(None)
                        row_t.append("N/A")
                        continue
                    _, _, ev_s, _, _, _ = calc_dcf(
                        noplat0, ss["g1"], ss["g2"], g, w, ss["roic_cv"], ss["roic_proj"]
                    )
                    eq_s = ev_s - net_debt
                    piv_s = eq_s / shares if shares > 0 else None
                    up_s = (
                        (piv_s / price - 1) * 100
                        if price and price > 0 and piv_s
                        else None
                    )
                    row_z.append(up_s)
                    row_t.append(f"{up_s:+.0f}%" if up_s is not None else "N/A")
                z_matrix.append(row_z)
                z_text.append(row_t)

            fig_s = go.Figure(
                go.Heatmap(
                    z=z_matrix,
                    x=[f"{g:.1f}%" for g in gt_range],
                    y=[f"{w * 100:.1f}%" for w in wacc_range],
                    colorscale="RdYlGn",
                    zmid=0,
                    zmin=-80,
                    zmax=80,
                    text=z_text,
                    texttemplate="%{text}",
                    textfont=dict(size=10, color="white"),
                    hovertemplate="WACC: %{y}<br>g terminal: %{x}<br>Potencial: %{text}<extra></extra>",
                )
            )
            fig_s.update_layout(
                xaxis_title="g terminal (%)",
                yaxis_title="WACC (%)",
                height=300,
                margin=dict(t=10, b=50, l=70, r=10),
            )
            apply_plotly_theme(fig_s)
            st.plotly_chart(fig_s, use_container_width=True)

        # Checklist de consistência
        with st.expander("✅ Checklist de Consistência (Koller)", expanded=False):
            roic_lat = next((y["roic"] for y in reversed(ys) if y.get("roic")), None)
            checks = [
                (True, "NOPLAT base positivo — LPA operacional disponível"),
                (len(ys) >= 2, f"Histórico: {len(ys)} anos disponíveis"),
                (
                    ss["gt"] <= 5.0,
                    f"g terminal ({ss['gt']:.1f}%) ≤ PIB nominal de longo prazo (~5%)",
                ),
                (ss["e_weight"] > 0.2, "Pesos de mercado usados (não contábeis)"),
                (
                    cash_v > 0,
                    "Ativos não-operacionais (caixa) separados do FCF",
                ),
                (
                    cv_pct <= 80,
                    f"CV = {cv_pct:.0f}% do EV — dentro do intervalo razoável (< 80%)",
                ),
                (
                    roic_lat is not None and roic_lat > 0,
                    f"ROIC histórico disponível: {_pct(roic_lat)}",
                ),
            ]
            for ok, txt in checks:
                icon = "✅" if ok else "⚠️"
                color = "#00ff87" if ok else "#ffd600"
                st.markdown(
                    f'<div style="padding:3px 0;font-size:0.82rem;color:{color}">'
                    f"{icon} {txt}</div>",
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Múltiplos
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("#### Validação por Múltiplos de Mercado")

    st.info(
        "Múltiplos são **contexto**, não substitutos do DCF. "
        "Use sempre múltiplos forward quando possível."
    )

    noplat0 = latest["noplat"]
    ebit0 = latest["ebit"]
    rev0 = latest["revenue"]

    if noplat0 > 0 and ebit0 and rev0:
        # Implied multiples from our DCF
        _, _, ev_dcf, _, _, _ = calc_dcf(
            noplat0, ss["g1"], ss["g2"], ss["gt"], wacc_dec, ss["roic_cv"], ss["roic_proj"]
        )

        ebitda_approx = ebit0 + latest["da"]  # EBITDA ≈ EBIT + D&A
        ebita_approx = ebit0  # EBITA ≈ EBIT (sem amortização explícita)

        mult_rows = [
            (
                "EV/EBITDA",
                ev_dcf / ebitda_approx if ebitda_approx > 0 else None,
                "Mais comum; neutro para D&A",
                "Koller: preferido para comparação entre empresas do mesmo setor",
            ),
            (
                "EV/EBITA",
                ev_dcf / ebita_approx if ebita_approx > 0 else None,
                "Melhor para comparação entre indústrias",
                "Remove o efeito de políticas de depreciação diferentes",
            ),
            (
                "EV/NOPLAT",
                ev_dcf / noplat0 if noplat0 > 0 else None,
                "Consistente com DCF; ajusta impostos",
                "Múltiplo mais alinhado ao Enterprise DCF — equivalente ao P/E sem alavancagem",
            ),
            (
                "EV/Receita",
                ev_dcf / rev0 if rev0 > 0 else None,
                "Para early-stage ou margens muito baixas",
                "Útil quando EBITDA é negativo ou muito volátil",
            ),
        ]

        if price and price > 0 and shares > 0:
            mkt_cap = price * shares
            lpa = noplat0 / shares
            mult_rows.insert(
                2,
                (
                    "P/E (implícito)",
                    mkt_cap / (noplat0) if noplat0 > 0 else None,
                    "Inclui efeito de alavancagem financeira",
                    "Calculado sobre NOPLAT (proxy do lucro operacional líquido)",
                ),
            )

        st.markdown("**Múltiplos implícitos pelo DCF vs mercado atual:**")
        for name, val, desc, note in mult_rows:
            val_str = f"{val:.1f}×" if val is not None else "—"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
                f'padding:0.65rem 0;border-bottom:1px solid #1e293b;">'
                f'<div style="flex:1">'
                f'<span style="color:#f8fafc;font-weight:700;font-size:0.88rem">{name}</span>'
                f'<div style="color:#64748b;font-size:0.72rem;margin-top:2px">{desc}</div>'
                f'<div style="color:#475569;font-size:0.68rem;margin-top:1px;font-style:italic">{note}</div>'
                f"</div>"
                f'<span style="color:#a855f7;font-family:monospace;font-weight:800;font-size:1rem;'
                f'margin-left:1rem;white-space:nowrap">{val_str}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Regras Koller para uso de múltiplos:**")
        rules = [
            "Use múltiplos **forward** (12m projetados) sempre que possível — LTM pode distorcer",
            "Ajuste para diferenças de crescimento e ROIC entre comparáveis (PEG ratio)",
            "Múltiplos são contexto para triangular o DCF, não substitutos",
            "Nunca inclua ativos não-operacionais no FCF **e** no ajuste de EV (double-counting)",
        ]
        for r in rules:
            st.markdown(
                f'<div style="padding:3px 0;font-size:0.82rem;color:#94a3b8">'
                f"· {r}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.warning(
            "NOPLAT, EBIT ou Receita indisponíveis para calcular os múltiplos implícitos."
        )

    # ── Comparação de Múltiplos do Setor ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Comparação de Múltiplos do Setor")
    st.caption(
        "Posicionamento por percentil em relação a todos os pares do setor na B3 "
        "(fonte: Fundamentus). Verde = favorável · Vermelho = desfavorável · Cinza = neutro."
    )

    _b3_data = pd.read_csv("acoes-listadas-b3.csv")
    _setor_ticker = (
        _b3_data[_b3_data["Ticker"] == ticker]["Setor"].values[0]
        if ticker in _b3_data["Ticker"].values
        else None
    )

    if not _setor_ticker:
        st.info("Setor não identificado no CSV B3 — comparação de peers indisponível.")
    else:
        _peers_tickers = _b3_data[_b3_data["Setor"] == _setor_ticker]["Ticker"].tolist()
        st.caption(
            f"Setor: **{_setor_ticker}** · {len(_peers_tickers)} empresas comparáveis"
        )

        with loading_overlay("Carregando múltiplos do setor..."):
            _peers_raw = get_sector_peers((_setor_ticker,))

        if _peers_raw.empty:
            st.warning("Não foi possível carregar dados de peers do Fundamentus.")
        else:
            _rank_df = compute_sector_ranking(
                _peers_raw, ticker, _setor_ticker, _b3_data
            )
            if _rank_df.empty:
                st.info("Dados de múltiplos não disponíveis para os peers deste setor.")
            else:
                n_peers_display = int(_rank_df["Peers (n)"].max())
                if n_peers_display < 5:
                    st.caption(
                        f"⚠️ Amostra pequena: {n_peers_display} peer(s) disponível(is) "
                        "no setor (ideal Koller: 5–15) — use os múltiplos com cautela."
                    )

                st.markdown("**Posicionamento por Percentil**")
                _styled_rank = (
                    _rank_df.style.map(color_veredicto, subset=["Veredicto"])
                    .map(color_pct, subset=["Percentil"])
                    .format(
                        {
                            "Valor": "{:.2f}",
                            "Mediana Setor": "{:.2f}",
                            "Média Setor": "{:.2f}",
                            "Percentil": "{:.1f}%",
                        }
                    )
                )
                st.dataframe(_styled_rank, use_container_width=True, hide_index=True)

                # ── Performance relativa no setor ────────────────────────────────
                st.markdown("**Performance Relativa no Setor**")
                _fig_pct = go.Figure(
                    go.Bar(
                        x=_rank_df["Múltiplo"],
                        y=_rank_df["Percentil"],
                        marker_color=[
                            "#00ff87" if p >= 70 else ("#ffd600" if p >= 40 else "#ff3d5a")
                            for p in _rank_df["Percentil"]
                        ],
                        text=[f"{p:.0f}%" for p in _rank_df["Percentil"]],
                        textposition="outside",
                    )
                )
                _fig_pct.update_layout(
                    title=f"{ticker} vs. {n_peers_display} peers do setor {_setor_ticker}",
                    yaxis_title="Percentil no Setor (%)",
                    yaxis=dict(range=[0, 105], ticksuffix="%"),
                    height=320,
                    margin=dict(t=40, b=40, l=40, r=20),
                )
                apply_plotly_theme(_fig_pct)
                st.plotly_chart(_fig_pct, use_container_width=True)

                # ── Distribuição do setor por múltiplo ───────────────────────────
                st.markdown("**Distribuição do Setor por Múltiplo**")
                _mult_sel = st.selectbox(
                    "Selecione o múltiplo para visualizar",
                    _rank_df["Múltiplo"].tolist(),
                    key="val_mult_dist_sel",
                )
                _mult_key = next(
                    (m[0] for m in MULTIPLES_CFG if m[1] == _mult_sel), None
                )
                if _mult_key and _mult_key in _peers_raw.columns:
                    _tickers_do_setor = _b3_data[_b3_data["Setor"] == _setor_ticker][
                        "Ticker"
                    ].tolist()
                    _col_plot = (
                        clean_numeric_column(
                            _peers_raw.loc[
                                _peers_raw.index.isin(_tickers_do_setor), _mult_key
                            ]
                        )
                        .dropna()
                        .sort_values()
                    )
                    if _col_plot.empty:
                        st.warning(f"Dados indisponíveis para o setor de {ticker}.")
                    else:
                        _bar_colors = [
                            "#a855f7" if i == ticker else "#334155"
                            for i in _col_plot.index
                        ]
                        _fig_mult = go.Figure(
                            go.Bar(
                                x=_col_plot.index.tolist(),
                                y=_col_plot.values,
                                marker_color=_bar_colors,
                                marker_line_width=0,
                                text=[f"{v:.1f}" for v in _col_plot.values],
                                textposition="outside",
                                textfont=dict(size=8, color="#94a3b8"),
                            )
                        )
                        _med_val = _col_plot.median()
                        _fig_mult.add_hline(
                            y=_med_val,
                            line_dash="dash",
                            line_color="#ffd600",
                            line_width=1.5,
                            annotation_text=f"Mediana: {_med_val:.1f}",
                            annotation_position="top left",
                            annotation_font=dict(color="#ffd600", size=11),
                        )
                        _fig_mult.update_layout(
                            title=f"{_mult_sel} — Peers de {ticker} no Setor: {_setor_ticker} ({len(_col_plot)} empresas)",
                            xaxis_title="Ticker",
                            yaxis_title=_mult_sel,
                            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                            height=420,
                            showlegend=False,
                        )
                        apply_plotly_theme(_fig_mult)
                        st.plotly_chart(_fig_mult, use_container_width=True)

    # ── Watchlist button ───────────────────────────────────────────────────────
    st.markdown("---")
    _starred = _db.wl_has(ticker)
    if st.button(
        "★ Remover dos Favoritos" if _starred else "☆ Salvar nos Favoritos",
        key="val_watchlist_btn",
        help="Ticker salvo na watchlist da página principal",
    ):
        if _starred:
            _db.wl_remove(ticker)
        else:
            _db.wl_add(ticker)
        st.rerun()
