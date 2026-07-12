"""Pure Koller DCF calculation engine.

No Streamlit/yfinance I/O here on purpose — keeping this module free of
side effects is what makes NOPLAT/ROIC/DCF math unit-testable without
mocking network calls.
"""

CAIXA_OP_PCT = 0.015  # caixa operacional ≈ 1.5% da receita


def compute_year_metrics(
    revenue,
    ebit,
    pretax_income,
    tax_expense,
    da,
    capex,
    delta_wc,
    ppe,
    goodwill,
    current_assets,
    current_liabilities,
    cash,
    debt_short,
    debt_long,
    equity,
    interest_expense=None,
    cash_op_pct=CAIXA_OP_PCT,
):
    """NOPLAT, FCF e Invested Capital de um ano fiscal (metodologia Koller)."""
    if pretax_income and tax_expense and abs(pretax_income) > 0:
        t_rate = min(max(abs(tax_expense) / abs(pretax_income), 0.10), 0.40)
    else:
        t_rate = 0.34

    noplat = ebit * (1 - t_rate)
    da_v = da if da is not None else 0.0
    cap_v = abs(capex) if capex is not None else 0.0
    dwc_v = delta_wc if delta_wc is not None else 0.0
    # "Change In Working Capital" do yfinance já vem no sinal do efeito de caixa
    # (DFC indireto): negativo quando o WC aumenta e consome caixa. Deve ser somado,
    # não subtraído — subtrair inverteria o impacto do capital de giro no FCF.
    fcf = noplat + da_v - cap_v + dwc_v

    cash_op = (revenue * cash_op_pct) if revenue else 0.0
    # Current Liabilities do yfinance já inclui a dívida de curto prazo (Current Debt);
    # subtrair debt_short de novo contaria a dívida duas vezes no WCO.
    wco = (current_assets or 0) - (current_liabilities or 0) - max(
        (cash or 0) - cash_op, 0
    )
    ic = wco + (ppe or 0) + (goodwill or 0)

    return {
        "tax_rate": t_rate,
        "noplat": noplat,
        "da": da_v,
        "capex": cap_v,
        "delta_wc": dwc_v,
        "fcf": fcf,
        "wco": wco,
        "ppe": ppe or 0,
        "goodwill": goodwill or 0,
        "ic": ic,
        "debt": (debt_short or 0.0) + (debt_long or 0.0),
        "equity": equity or 0,
        "cash": cash or 0,
        "interest": interest_expense,
    }


def compute_roic_series(years):
    """Preenche `roic`/`roic_no_gw` em uma lista cronológica de anos (dicts com
    `noplat`/`ic`/`goodwill`), usando o IC do ano anterior como denominador.
    Modifica e retorna a mesma lista.
    """
    for i, y in enumerate(years):
        ic_prev = years[i - 1]["ic"] if i > 0 else None
        y["roic"] = (
            (y["noplat"] / ic_prev * 100) if (ic_prev and ic_prev > 1e4) else None
        )
        gw_prev = years[i - 1]["goodwill"] if i > 0 else None
        y["roic_no_gw"] = (
            y["noplat"] / (ic_prev - gw_prev) * 100
            if (ic_prev and gw_prev is not None and ic_prev - gw_prev > 1e4)
            else None
        )
    return years


def calc_cv(noplat_next, g_pct, roic_cv_pct, wacc_dec):
    """Continuing Value (Gordon Growth adaptado): CV = NOPLAT_{T+1} × (1 − g/ROIC_cv) / (WACC − g)."""
    g, r = g_pct / 100, wacc_dec
    if r <= g:
        return None
    roic_cv = roic_cv_pct / 100
    reinv = min(g / roic_cv, 0.99) if roic_cv > 0 else 0.0
    return noplat_next * max(1.0 - reinv, 0.01) / (r - g)


def calc_dcf(noplat0, g1_pct, g2_pct, gt_pct, wacc_dec, roic_cv_pct, roic_proj_pct=None):
    """Projeção de 10 anos + Continuing Value. Retorna (pv_exp, pv_cv, ev, rows, cv, noplat_11)."""
    g1, g2, gt = g1_pct / 100, g2_pct / 100, gt_pct / 100
    rows, pv_exp = [], 0.0
    # ROIC do período explícito (anos 1-10) pode diferir do ROIC na perpetuidade —
    # usar roic_cv para os dois conflita o reinvestimento atual com a premissa terminal.
    roic_proj = (roic_proj_pct if roic_proj_pct is not None else roic_cv_pct) / 100

    for t in range(1, 11):
        noplat_t = noplat0 * (1 + g1) ** min(t, 5) * (1 + g2) ** max(0, t - 5)
        g_t = g1 if t <= 5 else g2
        reinv = min(g_t / roic_proj, 0.95) if roic_proj > 0 else 0.0
        fcf_t = noplat_t * max(1.0 - reinv, 0.05)
        pv_t = fcf_t / (1 + wacc_dec) ** t
        pv_exp += pv_t
        rows.append(
            {
                "t": t,
                "g_pct": g_t * 100,
                "noplat": noplat_t,
                "reinv_pct": reinv * 100,
                "fcf": fcf_t,
                "pv": pv_t,
            }
        )

    noplat_11 = noplat0 * (1 + g1) ** 5 * (1 + g2) ** 5 * (1 + gt)
    cv = calc_cv(noplat_11, gt_pct, roic_cv_pct, wacc_dec)
    pv_cv = (cv / (1 + wacc_dec) ** 10) if cv is not None else 0.0
    return pv_exp, pv_cv, pv_exp + pv_cv, rows, cv, noplat_11
