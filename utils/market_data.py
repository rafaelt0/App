"""Shared, cached access to bulk B3 market data.

Fetching the full Fundamentus table is expensive (scrapes & parses ~300+
tickers) and several pages need it independently — route them all through
this single cache instead of each page maintaining its own copy.
"""

import logging

import pandas as pd
import streamlit as st
from utils.formatting import normalize_numeric_text

logger = logging.getLogger(__name__)


def clean_numeric_column(col):
    """Parse Brazilian decimals and thousands into numeric values."""
    return pd.to_numeric(col.map(normalize_numeric_text), errors="coerce")


@st.cache_data(ttl=3600, show_spinner=False)
def get_full_market_data():
    """Fetch the full Fundamentus `resultado` table (all B3 tickers)."""
    import fundamentus.resultado as fzr

    return fzr.get_resultado_raw()


@st.cache_data(ttl=86400, show_spinner=False)
def get_sorted_tickers_by_liquidity(tickers_list):
    try:
        df = get_full_market_data()
        df = df.sort_values(by="Liq.2meses", ascending=False)
        sorted_all = df.index.tolist()
        sorted_filtered = [t for t in sorted_all if t in tickers_list]
        remaining = [t for t in tickers_list if t not in sorted_filtered]
        return sorted_filtered + remaining
    except Exception:
        logger.warning("get_sorted_tickers_by_liquidity failed, returning unsorted list", exc_info=True)
        return tickers_list
