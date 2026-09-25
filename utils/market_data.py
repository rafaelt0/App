"""Shared, cached access to bulk B3 market data.

Fetching the full Fundamentus table is expensive (scrapes & parses ~300+
tickers) and several pages need it independently — route them all through
this single cache instead of each page maintaining its own copy.
"""

import logging
import math
from pathlib import Path


import pandas as pd
import streamlit as st
from utils.formatting import normalize_numeric_text

logger = logging.getLogger(__name__)


def _normalize_listed_stocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the local stock-universe schema."""
    required = {"Ticker", "Setor"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"B3 universe is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("B3 universe is empty")
    frame = frame.copy()
    frame["Ticker"] = frame["Ticker"].astype(str).str.strip().str.upper()
    frame = frame[frame["Ticker"].ne("") & frame["Ticker"].ne("NAN")]
    if frame.empty:
        raise ValueError("B3 universe contains no valid tickers")
    return frame.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_listed_stocks() -> pd.DataFrame:
    """Load the shared local B3 universe once per hour."""
    frame = pd.read_csv(Path(__file__).resolve().parent.parent / "acoes-listadas-b3.csv")
    return _normalize_listed_stocks(frame)




def clean_numeric_column(col):
    """Parse Brazilian decimals and thousands into numeric values."""
    return pd.to_numeric(col.map(normalize_numeric_text), errors="coerce")


def compute_target_upside(current_price, target_price):
    """Return target-price upside in percent, or None for invalid prices."""
    try:
        current, target = float(current_price), float(target_price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(current) or not math.isfinite(target) or current <= 0 or target <= 0:
        return None
    return (target / current - 1) * 100


@st.cache_data(ttl=3600, show_spinner=False)
def get_full_market_data():
    """Fetch the full Fundamentus `resultado` table (all B3 tickers)."""
    import fundamentus.resultado as fzr

    return fzr.get_resultado_raw()


@st.cache_data(ttl=86400, show_spinner=False)
def get_sorted_tickers_by_liquidity(tickers_list):
    try:
        df = get_full_market_data()
        df = df.copy()
        df["_liquidity_sort"] = clean_numeric_column(df["Liq.2meses"])
        df = df.sort_values("_liquidity_sort", ascending=False, na_position="last")
        sorted_all = df.index.tolist()
        sorted_filtered = [t for t in sorted_all if t in tickers_list]
        remaining = [t for t in tickers_list if t not in sorted_filtered]
        return sorted_filtered + remaining
    except Exception:
        logger.warning("get_sorted_tickers_by_liquidity failed, returning unsorted list", exc_info=True)
        return tickers_list
