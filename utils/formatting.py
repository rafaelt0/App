"""Pure formatting/lookup helpers shared by the Streamlit pages — no st.* calls,
so these are safe to unit-test without a Streamlit runtime.
"""

import pandas as pd


def get_ev_ebitda_context(setor: str):
    """Returns (alt_metric, reason) when EV/EBITDA doesn't apply, or None if it applies normally."""
    s = setor.lower() if setor else ""
    if any(k in s for k in ("banco", "crédito", "credito", "câmbio", "cambio")):
        return (
            "P/L · P/VP",
            "Bancos: resultado financeiro é a atividade-fim — EV/EBITDA não se aplica.",
        )
    if any(k in s for k in ("seguro", "previdência", "previdencia", "resseguro")):
        return (
            "P/L · P/VP",
            "Seguradoras: lucro atrelado ao resultado financeiro (float) — EV/EBITDA não se aplica.",
        )
    if any(k in s for k in ("holding", "participação", "participacao")):
        return (
            "Desconto sobre NAV",
            "Holdings: receita de equivalência patrimonial — EBITDA é quase nulo ou negativo.",
        )
    if any(k in s for k in ("tecnologia", "software", "internet")):
        return (
            "EV/Sales",
            "Tech em crescimento: EBITDA frequentemente negativo pelo reinvestimento agressivo.",
        )
    if any(
        k in s
        for k in ("exploração", "exploracao", "pré-operacional", "pre-operacional")
    ):
        return (
            "EV/Recursos · DCF",
            "Empresa pré-operacional: sem receita, EBITDA estruturalmente negativo.",
        )
    return None


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
                str(row[name]).replace(",", ".").strip("%").strip(), errors="coerce"
            )
            if not pd.isna(v):
                return v
    return None
