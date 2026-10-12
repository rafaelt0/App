"""Fundamentus sources and sector-relative fundamental calculations."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

import fundamentus
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
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
    "Liq.2meses": "Liq_2meses",
    "Patrim. Líq": "Patrim_Liq",
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
# Campos auxiliares (não são múltiplos): liquidez para escolher a classe de ação
# representativa do emissor e patrimônio líquido para detectar patrimônio negativo.
LIQUIDITY_COL = "Liq_2meses"
EQUITY_COL = "Patrim_Liq"
AUX_COLS = (LIQUIDITY_COL, EQUITY_COL)
# Com patrimônio líquido negativo, P/VP e ROE deixam de ter significado econômico.
NEGATIVE_EQUITY_NULLED = ("PVP", "ROE")
# Único múltiplo em que 0 é um valor real (não pagou dividendos) e não "não aplicável".
ZERO_IS_VALUE = {"Div_Yield"}
# Proporção cortada em cada extremo da média aparada do setor.
TRIM_PROPORTION = 0.1


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


def _issuer_of(ticker) -> str:
    """Issuer key shared by a company's share classes (PETR3, PETR4 -> PETR, TAEE11 -> TAEE)."""
    root = re.sub(r"\d+$", "", str(ticker).strip().upper())
    return root[:4]


def _one_class_per_issuer(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the most liquid share class of each issuer; ties keep the earlier row."""
    if frame.empty:
        return frame
    if LIQUIDITY_COL in frame.columns:
        liquidity = frame[LIQUIDITY_COL].to_numpy(dtype=float)
        # Missing liquidity sorts after every observed class.
        keys = np.where(np.isnan(liquidity), np.inf, -liquidity)
    else:
        keys = np.zeros(len(frame))
    ordered = frame.iloc[np.lexsort((np.arange(len(frame)), keys))]
    duplicated = (
        pd.Series([_issuer_of(t) for t in ordered.index]).duplicated(keep="first").to_numpy()
    )
    return ordered.iloc[~duplicated]


def compute_sector_ranking(peers_raw: pd.DataFrame, ticker: str, setor: str, b3_data: pd.DataFrame) -> pd.DataFrame:
    """Rank one ticker's multiples against its Fundamentus sector peers.

    `peers_raw` is the (uncleaned) output of `get_sector_peers` — raw string
    columns for the whole B3. `b3_data` is the acoes-listadas-b3.csv frame
    (columns Ticker/Setor), used to restrict peers to the same sector.

    Returns one row per available multiple in MULTIPLES_CFG: Múltiplo, Valor,
    Mediana Setor, Média Setor, Peers (n), Percentil, Veredicto.

    - Percentil is the share of sector peers this ticker outperforms on that
      multiple, with ties counted as half (midrank). The ticker's own issuer is
      excluded from the peers, so the best peer-relative value is 100.
    - Peers (n) counts the ticker plus its comparable peers (one share class per
      issuer, the most liquid one).
    - Mediana Setor and Média Setor (10% trimmed mean) describe the same sector
      observations, including the ticker.

    Empty DataFrame if the ticker has no usable data.
    """
    cols_available = [c for c in COLS_NEEDED if c in peers_raw.columns]
    aux_available = [c for c in AUX_COLS if c in peers_raw.columns]
    peers_df = peers_raw[cols_available + aux_available].copy()
    for col in cols_available + aux_available:
        peers_df[col] = clean_numeric_column(peers_df[col])

    # Patrimônio líquido negativo (P/VP < 0 ou Patrim. Líq. <= 0) torna P/VP e
    # ROE sem sentido; a linha sai dessas distribuições, ticker e peers incluídos.
    negative_equity = np.zeros(len(peers_df), dtype=bool)
    if "PVP" in peers_df.columns:
        negative_equity |= (peers_df["PVP"] < 0).to_numpy()
    if EQUITY_COL in peers_df.columns:
        negative_equity |= (peers_df[EQUITY_COL] <= 0).to_numpy()
    for mult in NEGATIVE_EQUITY_NULLED:
        if mult in peers_df.columns:
            peers_df.loc[negative_equity, mult] = pd.NA

    # Fundamentus usa 0 para indicar "não aplicável" na maioria dos múltiplos;
    # Div. Yield = 0 é um pagador nulo real e permanece na distribuição.
    # Remove também outliers absurdos nos múltiplos onde menor = melhor.
    for mult, _name, lower_better, _categoria in MULTIPLES_CFG:
        if mult not in peers_df.columns:
            continue
        if mult not in ZERO_IS_VALUE:
            peers_df.loc[peers_df[mult] == 0, mult] = pd.NA
        if lower_better:
            peers_df.loc[
                (peers_df[mult].notna())
                & ((peers_df[mult] < 0) | (peers_df[mult] >= 500)),
                mult,
            ] = pd.NA

    has_value = peers_df[cols_available].notna().any(axis=1).to_numpy()
    peers_df = peers_df[has_value]
    if ticker not in peers_df.index:
        return pd.DataFrame()

    tickers_do_setor = b3_data[b3_data["Setor"] == setor]["Ticker"].tolist()
    if ticker not in tickers_do_setor:
        tickers_do_setor.append(ticker)

    ticker_issuer = _issuer_of(ticker)
    other_issuer = np.array(
        [_issuer_of(t) != ticker_issuer for t in peers_df.index], dtype=bool
    )
    other_peers = peers_df[other_issuer]

    rows = []
    for mult, name, lower_better, categoria in MULTIPLES_CFG:
        if mult not in peers_df.columns or pd.isna(peers_df.loc[ticker, mult]):
            continue
        val = float(peers_df.loc[ticker, mult])
        # One share class per issuer, chosen among the classes reporting this multiple.
        with_multiple = other_peers[other_peers[mult].notna().to_numpy()]
        peer_values = _one_class_per_issuer(with_multiple)[mult]

        sector_values = peer_values[peer_values.index.isin(tickers_do_setor)]
        if sector_values.empty:
            sector_values = peer_values

        n_peers = len(sector_values)
        if n_peers == 0:
            # Sem peers comparáveis não há percentil; a linha não é emitida.
            continue
        if lower_better:
            beaten = (sector_values > val).sum()
        else:
            beaten = (sector_values < val).sum()
        ties = (sector_values == val).sum()
        pct = (beaten + 0.5 * ties) / n_peers * 100

        observed = np.append(sector_values.to_numpy(dtype=float), val)

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
                "Mediana Setor": round(float(np.median(observed)), 2),
                "Média Setor": round(float(trim_mean(observed, TRIM_PROPORTION)), 2),
                "Peers (n)": n_peers + 1,
                "Percentil": round(float(pct), 1),
                "Veredicto": veredicto,
            }
        )

    return pd.DataFrame(rows)
