"""Fundamentus sources and sector-relative fundamental calculations."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import fundamentus
import pandas as pd
from utils.cache import cache_data

from utils import db as _db
from utils.charts import CHART_ACCENT, CHART_DANGER, CHART_MUTED, CHART_POSITIVE
from utils.market_data import (
    FUNDAMENTUS_REQUEST_LOCK,
    clean_numeric_column,
    get_full_market_data,
)

logger = logging.getLogger(__name__)

def _fetch_one(ticker):
    try:
        with FUNDAMENTUS_REQUEST_LOCK:
            return fundamentus.get_papel(ticker)
    except Exception:
        logger.warning("fundamentus fetch failed for ticker=%s", ticker, exc_info=True)
        return None

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


def build_analyst_synthesis(
    peers_raw: pd.DataFrame,
    ticker: str,
    setor: str,
    b3_data: pd.DataFrame,
) -> dict:
    """Summarize a ticker's sector-relative fundamentals by category."""
    empty_result = {
        "pontos_positivos": [],
        "pontos_negativos": [],
        "alertas": [],
        "categorias": {},
        "indicadores_validos": 0,
        "categorias_validas": 0,
        "veredicto": "DADOS INSUFICIENTES",
        "cor_veredicto": CHART_MUTED,
        "fonte": "comparação setorial",
    }
    if (
        not isinstance(peers_raw, pd.DataFrame)
        or peers_raw.empty
        or not isinstance(b3_data, pd.DataFrame)
        or not {"Ticker", "Setor"}.issubset(b3_data.columns)
        or not isinstance(setor, str)
        or not setor.strip()
    ):
        return empty_result

    ranking = compute_sector_ranking(peers_raw, ticker, setor.strip(), b3_data)
    if ranking.empty:
        return empty_result

    ranking = ranking.copy()
    for column in ("Valor", "Percentil", "Peers (n)"):
        ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
    eligible = ranking[
        (ranking["Peers (n)"] >= 3)
        & ranking["Valor"].notna()
        & ranking["Percentil"].notna()
    ].copy()

    positive_points = []
    negative_points = []
    neutral_points = []
    categories = {}
    for category in dict.fromkeys(config[3] for config in MULTIPLES_CFG):
        category_rows = eligible[eligible["Categoria"] == category]
        if category_rows.empty:
            continue

        percentile = round(float(category_rows["Percentil"].median()), 1)
        if percentile >= 70:
            verdict = "Favorável"
        elif percentile >= 40:
            verdict = "Neutro"
        else:
            verdict = "Desfavorável"
        categories[category] = {
            "percentil": percentile,
            "veredicto": verdict,
            "indicadores_validos": len(category_rows),
        }

        for name, value, peers_count, metric_percentile in category_rows[
            ["Múltiplo", "Valor", "Peers (n)", "Percentil"]
        ].itertuples(index=False, name=None):
            name = str(name)
            value = float(value)
            peers_count = int(peers_count)
            metric_percentile = float(metric_percentile)
            is_percentage = name.endswith(" (%)")
            display_name = name[:-4] if is_percentage else name
            display_value = value * 100 if is_percentage else value
            formatted_value = (
                f"{display_value:g}%" if is_percentage else f"{display_value:g}"
            )
            label = (
                f"{display_name} {formatted_value} · "
                f"P{metric_percentile:.1f} · n={peers_count}"
            )
            tooltip = (
                f"{display_name}: valor {formatted_value}; "
                f"percentil {metric_percentile:.1f}; {peers_count} observações "
                "do setor, incluindo o ticker analisado."
            )
            signal = (label, tooltip)
            if metric_percentile >= 70:
                positive_points.append(signal)
            elif metric_percentile >= 40:
                neutral_points.append(signal)
            else:
                negative_points.append(signal)

    categories_valid = len(categories)
    positive_categories = sum(
        category["veredicto"] == "Favorável" for category in categories.values()
    )
    negative_categories = sum(
        category["veredicto"] == "Desfavorável" for category in categories.values()
    )
    if categories_valid < 2:
        verdict, color = "DADOS INSUFICIENTES", CHART_MUTED
    elif positive_categories >= 2 and positive_categories > negative_categories:
        verdict, color = "ATRATIVO", CHART_POSITIVE
    elif negative_categories >= 2 and negative_categories > positive_categories:
        verdict, color = "FRACO", CHART_DANGER
    else:
        verdict, color = "NEUTRO", CHART_ACCENT

    return {
        "pontos_positivos": positive_points,
        "pontos_negativos": negative_points,
        "alertas": neutral_points,
        "categorias": categories,
        "indicadores_validos": len(eligible),
        "categorias_validas": categories_valid,
        "veredicto": verdict,
        "cor_veredicto": color,
        "fonte": "comparação setorial",
    }


@cache_data(ttl=3600)
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
            logger.debug("get_fundamentus_data cache parse failed for key=%s", cache_key, exc_info=True)
    last_exc = None
    for attempt in range(3):
        try:
            with ThreadPoolExecutor(max_workers=min(len(tickers), 5)) as ex:
                raw = list(ex.map(_fetch_one, tickers))
            results = [r for r in raw if r is not None and not r.empty]
            if not results:
                raise RuntimeError(
                    f"Nenhum dado retornado pelo Fundamentus para: {', '.join(tickers)}. "
                    "Verifique se os tickers estão corretos."
                )
            result = pd.concat(results)
            if len(results) == len(tickers):
                _db.cache_set(cache_key, result.to_json())
            else:
                logger.warning(
                    "fundamentus returned partial data for %s; skipping cache",
                    tickers,
                )
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning("fundamentus fetch attempt %d/3 failed for %s: %s", attempt + 1, tickers, exc)
            if attempt < 2:
                time.sleep(1)
                continue
    raise last_exc


def clear_fundamentus_cache() -> int:
    """Clear process and persistent Fundamentus caches."""
    get_fundamentus_data.clear()
    get_full_market_data.clear()
    get_sector_peers.clear()
    return _db.cache_clear_prefix("fund_")



@cache_data(ttl=3600)
def get_sector_peers():
    """Fetch the whole-market snapshot; ranking applies sector membership."""
    try:
        raw = get_full_market_data()
        df2 = pd.DataFrame(index=raw.index)
        for src, dest in FUNDAMENTUS_RENAME.items():
            if src in raw.columns:
                df2[dest] = raw[src]
        df2 = df2.drop_duplicates(keep="first")
        df2 = df2[~df2.index.duplicated(keep="first")]
        return df2
    except Exception:
        logger.warning("get_sector_peers failed", exc_info=True)
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
