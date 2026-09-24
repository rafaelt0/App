"""Card/section renderers for Main_Page.py — Streamlit-coupled UI building
blocks, kept separate from the page's data-fetching and top-level flow.
"""

import pandas as pd
from html import escape

import plotly.graph_objects as go
import streamlit as st

from utils import db as _db
from utils.charts import apply_plotly_theme
from utils.formatting import (
    extract_debt_metric,
    format_large_br_currency,
    format_large_number,
    get_ev_ebitda_context,
)
from utils.home_data import build_analyst_synthesis, build_hist_df
from utils.icons import ICO_ALERT, ICO_BOLT, ICO_CHECK_SM
from utils.ui import analyst_synthesis_header, loading_overlay



def _analyst_chip(item, color):
    text, tooltip = item
    return (
        f'<span title="{escape(str(tooltip), quote=True)}" '
        f'style="display:inline-block;background:{color}14;'
        f'border:1px solid {color}40;color:{color};border-radius:999px;'
        f'padding:1px 8px;font-size:0.7rem;font-weight:600;margin:0 4px 4px 0;'
        f'white-space:nowrap;">{escape(str(text))}</span>'
    )


def render_analyst_synthesis(
    df_ind,
    company_data,
    tickers,
    peers_raw,
    b3_data,
):
    """Render peer-relative signals and coverage for the selected tickers."""
    analyst_synthesis_header()
    synthesis_items = []
    sector_lookup = {}
    if isinstance(b3_data, pd.DataFrame) and {"Ticker", "Setor"}.issubset(
        b3_data.columns
    ):
        sector_lookup = (
            b3_data.drop_duplicates("Ticker")
            .set_index("Ticker")["Setor"]
            .to_dict()
        )

    for ticker in tickers:
        if ticker not in df_ind.index:
            continue

        if ticker in company_data.index and "Empresa" in company_data.columns:
            company_name = company_data.loc[ticker, "Empresa"]
            if isinstance(company_name, pd.Series):
                company_name = company_name.iloc[0]
        else:
            company_name = ticker

        sector = sector_lookup.get(str(ticker).strip().upper(), "")
        if not isinstance(sector, str) or not sector.strip():
            sector = ""
        else:
            sector = sector.strip()

        safe_ticker = escape(str(ticker))
        safe_company_name = escape(str(company_name))
        safe_sector = escape(sector or "Setor indisponível")
        synthesis = build_analyst_synthesis(peers_raw, ticker, sector, b3_data)
        verdict_color = synthesis["cor_veredicto"]

        chips_html = "".join(
            _analyst_chip(item, "#00ff87")
            for item in synthesis["pontos_positivos"]
        )
        chips_html += "".join(
            _analyst_chip(item, "#ff3d5a")
            for item in synthesis["pontos_negativos"]
        )
        chips_html += "".join(
            _analyst_chip(item, "#ffd600") for item in synthesis["alertas"]
        )
        if not chips_html:
            chips_html = (
                '<span style="color:#64748b;font-size:0.72rem;">'
                "Dados de peers insuficientes.</span>"
            )

        category_html = "".join(
            '<span style="display:inline-block;color:#cbd5e1;font-size:0.68rem;'
            f'margin:0 10px 3px 0;">{escape(str(category))}: '
            f'{escape(str(summary["veredicto"]))} · '
            f'P{float(summary["percentil"]):.1f} · '
            f'{int(summary["indicadores_validos"])} indicador(es)</span>'
            for category, summary in synthesis["categorias"].items()
        )
        if not category_html:
            category_html = (
                '<span style="color:#64748b;font-size:0.68rem;">'
                "Sem categorias com cobertura suficiente.</span>"
            )

        coverage_text = (
            f"Cobertura: {synthesis['indicadores_validos']} indicador(es) "
            f"válido(s) em {synthesis['categorias_validas']} categoria(s)"
        )
        synthesis_items.append(
            f"""
<div style="background:linear-gradient(135deg,#0e1b2f,#080c14);border:1px solid #1e293b;border-radius:10px;padding:0.55rem 0.85rem;margin-bottom:0.4rem;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.3rem;">
    <div>
      <span style="font-family:'JetBrains Mono',monospace;font-weight:800;color:#00d2ff;font-size:0.88rem;">{safe_ticker}</span>
      <span style="font-size:0.7rem;color:#64748b;margin-left:0.4rem;">{safe_company_name}</span>
      <span style="font-size:0.62rem;color:#475569;font-style:italic;margin-left:0.4rem;">{safe_sector}</span>
    </div>
    <span style="background:rgba(0,0,0,0.3);border:1px solid {verdict_color}40;border-radius:6px;padding:0.1rem 0.6rem;font-size:0.66rem;font-weight:800;color:{verdict_color};letter-spacing:0.06em;">{escape(str(synthesis["veredicto"]))}</span>
  </div>
  <div style="margin-bottom:0.2rem;">{category_html}</div>
  <div style="color:#64748b;font-size:0.62rem;margin-bottom:0.2rem;">{escape(coverage_text)}; mínimo de 3 observações por indicador.</div>
  <div>{chips_html}</div>
</div>
"""
        )

    if synthesis_items:
        st.markdown("".join(synthesis_items), unsafe_allow_html=True)



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
        f'<div class="mcard"><div class="mcard-label">{escape(str(lbl))}</div>'
        f'<div class="mcard-value" style="color:{clr};font-size:0.95rem">{escape(str(val))}</div></div>'
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
    st.markdown(
        """
<div class="metric-section-heading">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#61d4c6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="1" x2="12" y2="23"></line>
        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
    </svg>
    <span>Valuation</span>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "P/L",
            f"{row['P/L']:.2f}",
            help="Preço/Lucro: quantas vezes o mercado paga pelo lucro anual. Menor = mais barato.",
        )
    with c2:
        st.metric(
            "P/VP",
            f"{row['P/VP']:.2f}",
            help="Preço/Valor Patrimonial: compara o preço de mercado com o valor contábil. <1 pode indicar desconto.",
        )
    with c3:
        ev_val = row["EV/EBITDA"]
        if pd.isna(ev_val) or abs(ev_val) < 0.01:
            ctx = get_ev_ebitda_context(setor)
            if ctx:
                alt, reason = ctx
                st.metric(
                    "EV/EBITDA", "—", delta="→ " + alt, delta_color="off", help=reason
                )
            else:
                st.metric(
                    "EV/EBITDA",
                    "N/D",
                    help="Dado não disponível para este ticker via Fundamentus.",
                )
        else:
            st.metric(
                "EV/EBITDA",
                f"{ev_val:.2f}",
                help="Enterprise Value / EBITDA: múltiplo de valuation que considera a dívida. Menor = mais barato.",
            )

    # 2. Rentabilidade Section
    st.markdown(
        """
<div class="metric-section-heading">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8cb4f2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
        <polyline points="17 6 23 6 23 12"></polyline>
    </svg>
    <span>Rentabilidade</span>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        roe_val = row["ROE"]
        st.metric(
            "ROE",
            f"{roe_val:.2f}%",
            delta="Forte" if roe_val > 15 else ("Fraco" if roe_val < 5 else "Moderado"),
            delta_color="normal"
            if roe_val > 15
            else ("inverse" if roe_val < 5 else "off"),
            help="Return on Equity: lucro gerado para cada R$ de patrimônio. Acima de 15% é considerado bom.",
        )
    with c2:
        roic_val = row["ROIC"]
        st.metric(
            "ROIC",
            f"{roic_val:.2f}%",
            delta="Forte"
            if roic_val > 12
            else ("Fraco" if roic_val < 5 else "Moderado"),
            delta_color="normal"
            if roic_val > 12
            else ("inverse" if roic_val < 5 else "off"),
            help="Return on Invested Capital: eficiência no uso de todo o capital (próprio + dívida).",
        )
    with c3:
        ml_val = row["Margem Líquida"]
        st.metric(
            "Margem Líquida",
            f"{ml_val:.2f}%",
            delta="Alta" if ml_val > 15 else ("Baixa" if ml_val < 5 else "Média"),
            delta_color="normal"
            if ml_val > 15
            else ("inverse" if ml_val < 5 else "off"),
            help="Percentual da receita que vira lucro líquido. Quanto maior, mais lucrativa a empresa.",
        )
    with c4:
        mebit_val = row["Margem EBIT"]
        st.metric(
            "Margem EBIT",
            f"{mebit_val:.2f}%",
            delta="Alta" if mebit_val > 15 else ("Baixa" if mebit_val < 5 else "Média"),
            delta_color="normal"
            if mebit_val > 15
            else ("inverse" if mebit_val < 5 else "off"),
            help="Margem operacional antes de juros e impostos. Mede a eficiência operacional.",
        )

    # 3. Crescimento & Yield Section
    st.markdown(
        """
<div class="metric-section-heading">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e7b96b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
    </svg>
    <span>Crescimento &amp; Yield</span>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        dy_val = row["Dividend Yield"]
        st.metric(
            "Dividend Yield",
            f"{dy_val:.2f}%",
            delta="Alto" if dy_val > 6 else ("Baixo" if dy_val < 2 else "Moderado"),
            delta_color="normal" if dy_val > 4 else "off",
            help="Dividendos pagos divididos pelo preço. Percentual de retorno em dividendos ao ano.",
        )
    with c2:
        cr_val = row["Crescimento Receita 5 anos"]
        st.metric(
            "Crescimento Receita (5 anos)",
            f"{cr_val:.2f}%",
            delta="Forte"
            if cr_val > 10
            else ("Negativo" if cr_val < 0 else "Moderado"),
            delta_color="normal"
            if cr_val > 10
            else ("inverse" if cr_val < 0 else "off"),
            help="Taxa de crescimento anual composta da receita nos últimos 5 anos.",
        )


def render_debt_panel(ticker_name, row):
    """Renderiza painel de saúde financeira (endividamento) para um ticker."""
    DEBT_COL_ALIASES = {
        "div_brut_patrim": ["Dív.Brut/Patrim.", "Div_Brut_Patrim", "Div.Brut/Patrim."],
        "liq_corrente": ["Liq. Corr.", "Liq_Corr", "Liq. Corr"],
        "ev_ebit": ["EV_EBIT", "EV/EBIT"],
    }
    db_val = extract_debt_metric(row, DEBT_COL_ALIASES["div_brut_patrim"])
    lc_val = extract_debt_metric(row, DEBT_COL_ALIASES["liq_corrente"])
    ev_ebit_val = extract_debt_metric(row, DEBT_COL_ALIASES["ev_ebit"])
    # Fundamentus remove o decimal de múltiplos (EV/EBIT 12,5× → armazenado como 1250)
    if ev_ebit_val is not None:
        ev_ebit_val = ev_ebit_val / 100.0

    if db_val is None and lc_val is None and ev_ebit_val is None:
        st.info(
            "Dados de endividamento não disponíveis via Fundamentus para este ticker."
        )
        return

    debt_cards = {}
    debt_colors = {}

    if db_val is not None:
        debt_cards["Dívida / Patrimônio"] = f"{db_val:.2f}×"
        debt_colors["Dívida / Patrimônio"] = (
            "#ff3d5a" if db_val > 3 else ("#ffd600" if db_val > 1.5 else "#00ff87")
        )

    if lc_val is not None:
        debt_cards["Liquidez Corrente"] = f"{lc_val:.2f}×"
        debt_colors["Liquidez Corrente"] = (
            "#ff3d5a" if lc_val < 1 else ("#ffd600" if lc_val < 1.5 else "#00ff87")
        )

    if ev_ebit_val is not None:
        debt_cards["EV / EBIT"] = f"{ev_ebit_val:.1f}×"
        debt_colors["EV / EBIT"] = (
            "#ff3d5a"
            if ev_ebit_val > 20
            else ("#ffd600" if ev_ebit_val > 12 else "#00ff87")
        )

    cards_html = "".join(
        f'<div class="mcard"><div class="mcard-label">{lbl}</div>'
        f'<div class="mcard-value" style="color:{debt_colors[lbl]}">{val}</div></div>'
        for lbl, val in debt_cards.items()
    )
    st.markdown(f'<div class="mcard-grid">{cards_html}</div>', unsafe_allow_html=True)

    diags = []
    if db_val is not None:
        if db_val > 3:
            diags.append(
                (
                    ICO_ALERT,
                    f"Dívida/PL de {db_val:.1f}× é elevada — verifique capacidade de pagamento",
                    "#ff3d5a",
                )
            )
        elif db_val > 1.5:
            diags.append(
                (ICO_BOLT, f"Dívida/PL de {db_val:.1f}× é moderada — monitorar", "#ffd600")
            )
        else:
            diags.append((ICO_CHECK_SM, f"Dívida/PL de {db_val:.1f}× é saudável", "#00ff87"))

    if lc_val is not None:
        if lc_val < 1:
            diags.append(
                (
                    ICO_ALERT,
                    f"Liquidez Corrente {lc_val:.2f}× < 1 — risco de dificuldade de caixa",
                    "#ff3d5a",
                )
            )
        elif lc_val < 1.5:
            diags.append(
                (ICO_BOLT, f"Liquidez Corrente {lc_val:.2f}× — margem estreita", "#ffd600")
            )
        else:
            diags.append(
                (
                    ICO_CHECK_SM,
                    f"Liquidez Corrente {lc_val:.2f}× — empresa com boa folga de caixa",
                    "#00ff87",
                )
            )

    for icon, msg, color in diags:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:8px;padding:0.35rem 0;'
            f'font-size:0.83rem;color:{color};">'
            f'<span style="flex-shrink:0">{icon}</span><span>{msg}</span></div>',
            unsafe_allow_html=True,
        )


def render_star_button(tkr, uid):
    """Renderiza botão de favoritar/desfavoritar da watchlist."""
    starred = _db.wl_has(uid, tkr)
    label = "Favoritado" if starred else "Favoritar"
    if st.button(
        label,
        key=f"star_{tkr}",
        type="primary" if starred else "secondary",
        help="Remover dos favoritos" if starred else "Salvar nos favoritos",
    ):
        if starred:
            _db.wl_remove(uid, tkr)
        else:
            _db.wl_add(uid, tkr)
        st.rerun()


def color_veredicto(val):
    """Estilo CSS para coluna Veredicto na tabela de percentis."""
    m = {"Favorável": "#00ff8722", "Neutro": "#ffd60022", "Desfavorável": "#ff3d5a22"}
    c = {"Favorável": "#00ff87", "Neutro": "#ffd600", "Desfavorável": "#ff3d5a"}
    return f"background-color:{m.get(val, '')};color:{c.get(val, '')};font-weight:600"


def color_pct(val):
    """Estilo CSS para coluna Percentil na tabela de percentis."""
    if val >= 70:
        return "color:#00ff87;font-weight:700"
    if val >= 40:
        return "color:#ffd600;font-weight:700"
    return "color:#ff3d5a;font-weight:700"


def get_ticker_setor(df, ticker):
    """Retorna o setor de um ticker a partir do DataFrame fundamentus.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame retornado por ``get_fundamentus_data``.
    ticker : str
        Código do ticker (sem sufixo .SA).
    """
    if "Setor" not in df.columns or ticker not in df.index:
        return ""
    val = df.loc[ticker, "Setor"]
    if isinstance(val, pd.Series):
        val = val.iloc[-1]
    return str(val) if not pd.isna(val) else ""


def render_hist_section(tkr):
    """Renderiza seção de histórico fundamentalista (receita, margens, ROE) para um ticker."""
    with loading_overlay(f"Buscando histórico de {tkr}…", tickers=[tkr]):
        df_h = build_hist_df(tkr)
    if df_h is None or df_h.empty:
        st.info(f"Dados históricos não disponíveis para {tkr} via yfinance.")
        return

    tab_rev, tab_marg, tab_roe = st.tabs(["Receita & Lucro", "Margens", "ROE"])

    with tab_rev:
        cols_rev = [c for c in ["Receita", "Lucro Líquido"] if c in df_h.columns]
        if not cols_rev:
            st.info("Dados de receita não disponíveis.")
        else:
            df_rev = df_h[cols_rev].dropna(how="all")
            max_abs = df_rev.abs().max().max()
            scale, unit = (
                (1e9, "R$ Bilhões")
                if max_abs >= 1e9
                else (1e6, "R$ Milhões")
                if max_abs >= 1e6
                else (1, "R$")
            )
            df_rev = df_rev / scale
            clr = {"Receita": "#00d2ff", "Lucro Líquido": "#00ff87"}
            fig_rev = go.Figure()
            for col in cols_rev:
                vals = df_rev[col].fillna(0)
                fig_rev.add_trace(
                    go.Bar(
                        name=col,
                        x=df_rev.index.astype(str),
                        y=vals,
                        marker_color=clr.get(col, "#94a3b8"),
                        text=[f"{v:.1f}" if v != 0 else "" for v in vals],
                        textposition="outside",
                        textfont=dict(size=10, color="#f8fafc"),
                    )
                )
            fig_rev.update_layout(
                barmode="group",
                xaxis_title="Ano",
                yaxis_title=unit,
                height=360,
                margin=dict(t=20, b=40, l=40, r=20),
            )
            apply_plotly_theme(fig_rev)
            st.plotly_chart(fig_rev, use_container_width=True)

    with tab_marg:
        cols_m = [
            c for c in ["Margem Líquida (%)", "Margem EBIT (%)"] if c in df_h.columns
        ]
        if not cols_m:
            st.info("Dados de margem não disponíveis.")
        else:
            df_m = df_h[cols_m].dropna(how="all")
            clr_m = {"Margem Líquida (%)": "#00ff87", "Margem EBIT (%)": "#ffd600"}
            fig_m = go.Figure()
            for col in cols_m:
                vals = df_m[col]
                fig_m.add_trace(
                    go.Scatter(
                        name=col,
                        x=df_m.index.astype(str),
                        y=vals,
                        mode="lines+markers+text",
                        line=dict(color=clr_m.get(col, "#94a3b8"), width=2.5),
                        marker=dict(size=8),
                        text=[f"{v:.1f}%" if pd.notna(v) else "" for v in vals],
                        textposition="top center",
                        textfont=dict(size=10, color="#f8fafc"),
                    )
                )
            fig_m.update_layout(
                xaxis_title="Ano",
                yaxis_title="Margem (%)",
                yaxis=dict(ticksuffix="%"),
                height=360,
                margin=dict(t=20, b=40, l=40, r=20),
            )
            apply_plotly_theme(fig_m)
            st.plotly_chart(fig_m, use_container_width=True)

    with tab_roe:
        if "ROE (%)" not in df_h.columns or df_h["ROE (%)"].dropna().empty:
            st.info("Dados de ROE histórico não disponíveis.")
        else:
            df_roe = df_h[["ROE (%)"]].dropna()
            fig_roe = go.Figure(
                go.Scatter(
                    x=df_roe.index.astype(str),
                    y=df_roe["ROE (%)"],
                    mode="lines+markers+text",
                    line=dict(color="#a855f7", width=2.5),
                    marker=dict(size=9, color="#a855f7"),
                    fill="tozeroy",
                    fillcolor="rgba(168,85,247,0.08)",
                    text=[
                        f"{v:.1f}%" if pd.notna(v) else "" for v in df_roe["ROE (%)"]
                    ],
                    textposition="top center",
                    textfont=dict(size=10, color="#f8fafc"),
                )
            )
            fig_roe.add_hline(
                y=15,
                line_dash="dash",
                line_color="#00ff87",
                line_width=1.5,
                annotation_text="Referência: 15%",
                annotation_position="top right",
                annotation_font=dict(color="#00ff87", size=10),
            )
            fig_roe.update_layout(
                xaxis_title="Ano",
                yaxis_title="ROE (%)",
                yaxis=dict(ticksuffix="%"),
                height=360,
                margin=dict(t=20, b=40, l=40, r=20),
            )
            apply_plotly_theme(fig_roe)
            st.plotly_chart(fig_roe, use_container_width=True)
