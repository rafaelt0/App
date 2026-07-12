"""Cached data-fetching + sector-comparison helpers shared by Main_Page.py
and pages/4_Valuation.py (fundamentus + yfinance).
"""

import time

import fundamentus
import pandas as pd
import streamlit as st
import yfinance as yf

from utils import db as _db
from utils.market_data import clean_numeric_column, get_full_market_data

# Mapa de renomeação de colunas do fundamentus para identificadores internos
FUNDAMENTUS_RENAME = {
    "P/L": "PL",
    "P/VP": "PVP",
    "EV/EBITDA": "EV_EBITDA",
    "EV/EBIT": "EV_EBIT",
    "PSR": "PSR",
    "ROE": "ROE",
    "ROIC": "ROIC",
    "Mrg Ebit": "Marg_EBIT",
    "Mrg. Líq.": "Marg_Liquida",
    "Div.Yield": "Div_Yield",
}

# Múltiplos a comparar na seção de peers: (coluna interna, nome display, menor=melhor?, categoria)
MULTIPLES_CFG = [
    ("PL", "P/L", True, "Valuation"),
    ("PVP", "P/VP", True, "Valuation"),
    ("EV_EBITDA", "EV/EBITDA", True, "Valuation"),
    ("EV_EBIT", "EV/EBIT", True, "Valuation"),
    ("PSR", "PSR", True, "Valuation"),
    ("ROE", "ROE (%)", False, "Rentabilidade"),
    ("ROIC", "ROIC (%)", False, "Rentabilidade"),
    ("Marg_EBIT", "Margem EBIT (%)", False, "Rentabilidade"),
    ("Marg_Liquida", "Marg. Líq. (%)", False, "Rentabilidade"),
    ("Div_Yield", "Div. Yield (%)", False, "Yield"),
]
COLS_NEEDED = [c[0] for c in MULTIPLES_CFG]


@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_yfinance_data(tickers_yf, start, interval):
    """Busca cotações do Yahoo Finance com retry automático."""
    import datetime

    today = datetime.date.today()
    for attempt in range(3):
        try:
            return yf.download(tickers_yf, start=start, end=today, interval=interval)[
                "Close"
            ]
        except OSError:
            if attempt < 2:
                time.sleep(1)
                continue
            raise


@st.cache_data(ttl=14400, show_spinner=False)
def get_hist_fundamentals(ticker_sa: str):
    """Fetch annual income statement + balance sheet from yfinance (4h cache)."""
    t_yf = yf.Ticker(ticker_sa)
    out = {}
    try:
        fin = t_yf.financials
        if fin is not None and not fin.empty:
            out["fin"] = fin.to_json()
    except Exception:
        pass
    try:
        bs = t_yf.balance_sheet
        if bs is not None and not bs.empty:
            out["bs"] = bs.to_json()
    except Exception:
        pass
    return out


def _hist_parse(json_str):
    """Parse a yfinance JSON financials/balance-sheet string into a year-indexed DataFrame."""
    df_p = pd.read_json(json_str)
    try:
        df_p.columns = pd.to_datetime(df_p.columns.astype("int64"), unit="ms").year
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


def build_hist_df(tkr: str):
    """Returns a DataFrame indexed by year with Receita, Lucro, Margens, ROE."""
    raw = get_hist_fundamentals(tkr + ".SA")
    if not raw:
        return None

    fin_df = bs_df = None
    if "fin" in raw:
        try:
            fin_df = _hist_parse(raw["fin"])
        except Exception:
            pass
    if "bs" in raw:
        try:
            bs_df = _hist_parse(raw["bs"])
        except Exception:
            pass

    records = {}

    if fin_df is not None:
        rev = _hist_row(fin_df, "Total Revenue", "Revenue")
        ni = _hist_row(
            fin_df,
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Applicable To Common Shares",
            "Net Income Including Noncontrolling Interests",
        )
        eb = _hist_row(fin_df, "EBIT", "Operating Income", "Ebit")

        for y in sorted(fin_df.columns.tolist()):
            y = int(y)
            rec = records.setdefault(y, {})
            r_v = (
                float(rev[y])
                if (rev is not None and y in rev.index and pd.notna(rev[y]))
                else None
            )
            n_v = (
                float(ni[y])
                if (ni is not None and y in ni.index and pd.notna(ni[y]))
                else None
            )
            e_v = (
                float(eb[y])
                if (eb is not None and y in eb.index and pd.notna(eb[y]))
                else None
            )
            if r_v is not None:
                rec["Receita"] = r_v
            if n_v is not None:
                rec["Lucro Líquido"] = n_v
            if r_v and n_v is not None and abs(r_v) > 0:
                rec["Margem Líquida (%)"] = n_v / r_v * 100
            if r_v and e_v is not None and abs(r_v) > 0:
                rec["Margem EBIT (%)"] = e_v / r_v * 100

    if bs_df is not None and fin_df is not None:
        eq = _hist_row(
            bs_df,
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Stockholder Equity",
            "Total Equity Gross Minority Interest",
        )
        ni = _hist_row(
            fin_df,
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income Applicable To Common Shares",
            "Net Income Including Noncontrolling Interests",
        )
        if eq is not None and ni is not None:
            for y in sorted(bs_df.columns.tolist()):
                y = int(y)
                e_v = float(eq[y]) if y in eq.index and pd.notna(eq[y]) else None
                n_v = float(ni[y]) if y in ni.index and pd.notna(ni[y]) else None
                if e_v and n_v is not None and abs(e_v) > 0:
                    records.setdefault(y, {})["ROE (%)"] = n_v / e_v * 100

    if not records:
        return None
    df_h = pd.DataFrame(records).T.sort_index()
    df_h.index.name = "Ano"
    return df_h


@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_peers(_setores):
    """Busca todos os tickers listados no fundamentus e mapeia colunas para identificadores internos.

    O argumento ``_setores`` é recebido (como tupla) apenas para que o Streamlit inclua o setor
    na chave de cache; os dados retornados cobrem toda a B3 para permitir filtragem posterior
    por setor individual sem chamadas extras à API.
    """
    try:
        raw = get_full_market_data()
        df2 = pd.DataFrame(index=raw.index)
        for src, dest in FUNDAMENTUS_RENAME.items():
            if src in raw.columns:
                df2[dest] = raw[src]
        df2 = df2.drop_duplicates(keep="first")
        return df2
    except Exception:
        return pd.DataFrame()


def compute_sector_ranking(peers_raw: pd.DataFrame, ticker: str, setor: str, b3_data: pd.DataFrame) -> pd.DataFrame:
    """Rank one ticker's multiples against its Fundamentus sector peers.

    `peers_raw` is the (uncleaned) output of `get_sector_peers` — raw string
    columns for the whole B3. `b3_data` is the acoes-listadas-b3.csv frame
    (columns Ticker/Setor), used to restrict peers to the same sector.

    Returns one row per available multiple in MULTIPLES_CFG: Múltiplo, Valor,
    Mediana Setor, Média Setor, Peers (n), Percentil, Veredicto. Percentil is
    the share of sector peers this ticker outperforms on that multiple
    (100 = best in sector). Empty DataFrame if the ticker has no usable data.
    """
    cols_available = [c for c in COLS_NEEDED if c in peers_raw.columns]
    peers_df = peers_raw[cols_available].copy()
    for col in cols_available:
        peers_df[col] = clean_numeric_column(peers_df[col])

    # Fundamentus usa 0 para indicar "não aplicável" em muitos múltiplos;
    # remove também outliers absurdos nos múltiplos onde menor = melhor.
    for mult, _name, lower_better, _categoria in MULTIPLES_CFG:
        if mult not in peers_df.columns:
            continue
        peers_df.loc[peers_df[mult] == 0, mult] = pd.NA
        if lower_better:
            peers_df.loc[
                (peers_df[mult].notna())
                & ((peers_df[mult] < 0) | (peers_df[mult] >= 500)),
                mult,
            ] = pd.NA

    peers_df = peers_df.dropna(how="all")
    if ticker not in peers_df.index:
        return pd.DataFrame()

    tickers_do_setor = b3_data[b3_data["Setor"] == setor]["Ticker"].tolist()
    if ticker not in tickers_do_setor:
        tickers_do_setor.append(ticker)

    rows = []
    for mult, name, lower_better, categoria in MULTIPLES_CFG:
        if mult not in peers_df.columns or pd.isna(peers_df.loc[ticker, mult]):
            continue
        val = peers_df.loc[ticker, mult]
        col_data = peers_df[mult].dropna()

        col_data_sector = col_data[col_data.index.isin(tickers_do_setor)]
        if col_data_sector.empty:
            col_data_sector = col_data

        n_peers = len(col_data_sector)
        pct = (
            (col_data_sector > val).sum() / n_peers * 100
            if lower_better
            else (col_data_sector < val).sum() / n_peers * 100
        )

        if pct >= 70:
            veredicto = "Favorável"
        elif pct >= 40:
            veredicto = "Neutro"
        else:
            veredicto = "Desfavorável"

        rows.append(
            {
                "Categoria": categoria,
                "Múltiplo": name,
                "Valor": round(val, 2),
                "Mediana Setor": round(col_data_sector.median(), 2),
                "Média Setor": round(col_data_sector.mean(), 2),
                "Peers (n)": n_peers,
                "Percentil": round(pct, 1),
                "Veredicto": veredicto,
            }
        )

    return pd.DataFrame(rows)
