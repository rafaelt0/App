"""Unified trailing and projected cash-flow engine for multi-asset portfolios.

Proventos are read from Yahoo Finance per listed asset and normalized to BRL. The
engine never fills a failed source with an estimate: an asset whose history or
current price cannot be read is listed in ``missing_tickers`` and kept out of the
totals. Modeled fixed income and non-coupon Tesouro Direto titles are reported
explicitly instead of being treated as dividend payers.

The forward schedule repeats the seasonality of the 12 completed calendar months
before the current month, applied to the share count implied by the current
position value. It is a historical-pace projection, never a guarantee.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from utils.assets import (
    ASSET_CLASS_LABELS,
    asset_currency,
    asset_metadata,
    is_synthetic_fixed_income,
    normalize_ticker,
    provider_symbol,
    supports_share_allocation,
)
from utils.cache import cache_data
from utils.charts import CHART_PRIMARY, CHART_PRIMARY_TRACE, apply_plotly_theme
from utils.market_data import YFINANCE_REQUEST_LOCK
from utils.portfolio_data import get_selic_rate
from utils.treasury_data import TREASURY_TITLES

logger = logging.getLogger(__name__)

MONTHS = ("Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez")
TRAILING_MONTHS = 12
JCP_IR_RATE = 0.15
US_WITHHOLDING_RATE = 0.30
CDI_IR_RATE = 0.15
TREASURY_NO_COUPON_NOTE = "Título sem cupons semestrais — fluxo concentrado no vencimento"
METHODOLOGY = (
    "Proventos por cota vêm do Yahoo Finance e são convertidos para reais pela taxa BRL=X "
    "na data de cada pagamento. A janela real cobre os 12 meses-calendário completos anteriores "
    "ao mês atual. A projeção de 12 meses repete a sazonalidade mensal observada nessa janela, "
    "aplicada à quantidade estimada pela posição atual (valor da posição ÷ preço de hoje). "
    "É um ritmo histórico, não uma garantia de pagamento."
)


def _timestamp(value=None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(dt.date.today()).normalize()
    return pd.Timestamp(value).normalize()


def _month_label(year: int, month: int) -> str:
    return f"{MONTHS[month - 1]}/{year}"


def _empty_history(ticker: str, *, status: str, note: str | None, currency: str = "BRL",
                   income_mode: str = "not_applicable", symbol: str | None = None,
                   source: str = "Yahoo Finance") -> dict[str, Any]:
    return {
        "ticker": ticker,
        "symbol": symbol,
        "currency": currency,
        "status": status,
        "income_mode": income_mode,
        "dividends_brl": pd.Series(dtype=float),
        "source": source,
        "note": note,
    }


@cache_data(ttl=3600)
def get_fx_brl_history(start_date) -> pd.Series:
    """USD/BRL closes from Yahoo Finance, with a margin before the first payment."""
    start = pd.Timestamp(start_date).normalize() - pd.Timedelta(days=10)
    end = pd.Timestamp(dt.date.today()) + pd.Timedelta(days=1)
    with YFINANCE_REQUEST_LOCK:
        frame = yf.download(
            "BRL=X",
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            progress=False,
            auto_adjust=False,
            timeout=15,
        )
    if frame is None or frame.empty or "Close" not in frame:
        raise ValueError("Yahoo Finance não retornou o câmbio USD/BRL.")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    close = close[np.isfinite(close) & (close > 0)]
    if close.empty:
        raise ValueError("Yahoo Finance retornou câmbio USD/BRL vazio.")
    return close.sort_index()


def _convert_usd_series(series: pd.Series, ticker: str) -> pd.Series:
    if series.empty:
        return series
    fx = get_fx_brl_history(series.index.min())
    rates = []
    for day, amount in series.items():
        available = fx.loc[:day]
        if available.empty:
            raise ValueError(f"Câmbio USD/BRL indisponível para {ticker} em {day:%d/%m/%Y}.")
        rates.append(float(amount) * float(available.iloc[-1]))
    return pd.Series(rates, index=series.index)


@cache_data(ttl=3600)
def get_asset_dividend_history(ticker: str, start_date) -> dict[str, Any]:
    """Trailing distributions per share in BRL for one asset.

    Returns ``status`` ``ok``, ``no_distributions``, ``missing``,
    ``no_intermediate_coupons`` or ``not_applicable`` and an explicit ``note``.
    A failed source returns an empty series; no yield is invented.
    """
    canonical = normalize_ticker(ticker)
    meta = asset_metadata(canonical)
    currency = str(meta.get("currency") or "BRL").upper()
    if canonical in TREASURY_TITLES:
        return _empty_history(
            canonical,
            status="no_intermediate_coupons",
            income_mode="no_intermediate_coupons",
            note=TREASURY_NO_COUPON_NOTE,
            source="Tesouro Direto",
        )
    if is_synthetic_fixed_income(canonical):
        return _empty_history(
            canonical,
            status="not_applicable",
            income_mode="not_applicable",
            note="Exposição de renda fixa modelada: sem provento de dividendos observável.",
            source="Modelo B3Lab",
        )
    symbol = meta.get("yahoo_symbol") or provider_symbol(canonical)
    if not symbol:
        return _empty_history(
            canonical,
            status="missing",
            note=f"Símbolo de mercado indisponível para {canonical}.",
            currency=currency,
        )
    currency = asset_currency(symbol)
    if currency not in {"BRL", "USD"}:
        return _empty_history(
            canonical,
            status="missing",
            note=f"Moeda {currency} sem conversão para BRL; proventos não incluídos.",
            currency=currency,
            symbol=symbol,
        )
    try:
        raw = yf.Ticker(symbol).dividends
    except Exception:
        logger.warning("Dividend history unavailable for %s", canonical, exc_info=True)
        return _empty_history(
            canonical,
            status="missing",
            note=f"Fonte de proventos indisponível para {canonical}.",
            currency=currency,
            symbol=symbol,
        )
    if raw is None:
        raw = pd.Series(dtype=float)
    raw_series = pd.Series(raw)
    if raw_series.empty:
        return _empty_history(
            canonical,
            status="no_distributions",
            income_mode="dividends",
            note="Sem distribuições nos 12 meses completos anteriores.",
            currency=currency,
            symbol=symbol,
        )
    index = pd.to_datetime(raw_series.index, errors="coerce")
    values = pd.to_numeric(raw_series.to_numpy(), errors="coerce").astype(float)
    valid = ~pd.isna(index) & np.isfinite(values) & (values > 0)
    series = pd.Series(values[valid], index=index[valid]).sort_index()
    if series.empty:
        return _empty_history(
            canonical,
            status="no_distributions",
            income_mode="dividends",
            note="Sem distribuições válidas no período.",
            currency=currency,
            symbol=symbol,
        )
    series = series.loc[series.index >= pd.Timestamp(start_date)]
    if series.empty:
        return _empty_history(
            canonical,
            status="no_distributions",
            income_mode="dividends",
            note="Sem distribuições nos 12 meses completos anteriores.",
            currency=currency,
            symbol=symbol,
        )
    if currency == "USD":
        try:
            series = _convert_usd_series(series, canonical)
        except Exception:
            logger.warning("USD/BRL conversion unavailable for %s dividends", canonical, exc_info=True)
            return _empty_history(
                canonical,
                status="missing",
                note=f"Câmbio USD/BRL indisponível para os proventos de {canonical}.",
                currency=currency,
                symbol=symbol,
            )
    return {
        "ticker": canonical,
        "symbol": symbol,
        "currency": currency,
        "status": "ok",
        "income_mode": "dividends",
        "dividends_brl": series,
        "source": "Yahoo Finance" + (" · BRL=X" if currency == "USD" else ""),
        "note": None,
    }


@cache_data(ttl=900)
def get_asset_price_brl(ticker: str) -> dict[str, Any]:
    """Latest unadjusted close in BRL, used only to scale per-share proventos."""
    canonical = normalize_ticker(ticker)
    meta = asset_metadata(canonical)
    currency = str(meta.get("currency") or "BRL").upper()
    if canonical in TREASURY_TITLES or is_synthetic_fixed_income(canonical):
        return {"ticker": canonical, "status": "not_applicable", "price_brl": None, "price_date": None,
                "currency": currency, "source": None, "note": TREASURY_NO_COUPON_NOTE}
    symbol = meta.get("yahoo_symbol") or provider_symbol(canonical)
    currency = asset_currency(symbol)
    if currency not in {"BRL", "USD"}:
        return {"ticker": canonical, "status": "missing", "price_brl": None, "price_date": None,
                "currency": currency, "source": "Yahoo Finance",
                "note": f"Moeda {currency} sem conversão para BRL."}
    try:
        history = yf.Ticker(symbol).history(period="10d", auto_adjust=False)
        close = history["Close"] if "Close" in history else pd.Series(dtype=float)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce")
        close = close[np.isfinite(close) & (close > 0)]
        if close.empty:
            raise ValueError("histórico de preços vazio")
        price = float(close.iloc[-1])
        price_date = close.index[-1]
        if currency == "USD":
            fx = get_fx_brl_history(price_date)
            available = fx.loc[:price_date]
            if available.empty:
                raise ValueError("câmbio USD/BRL indisponível")
            price *= float(available.iloc[-1])
        return {"ticker": canonical, "status": "ok", "price_brl": price,
                "price_date": pd.Timestamp(price_date).date().isoformat(), "currency": currency,
                "source": "Yahoo Finance" + (" · BRL=X" if currency == "USD" else ""), "note": None}
    except Exception:
        logger.warning("Current price unavailable for %s", canonical, exc_info=True)
        return {"ticker": canonical, "status": "missing", "price_brl": None, "price_date": None,
                "currency": currency, "source": "Yahoo Finance",
                "note": f"Preço atual indisponível para {canonical}; proventos não projetados."}


def income_tax_rate(asset_class: str, currency: str = "BRL", jcp_share: float = 0.0) -> tuple[float, str]:
    """Withholding rate and its plain-language reason for one asset class."""
    share = min(1.0, max(0.0, float(jcp_share or 0.0)))
    if asset_class == "fii":
        return 0.0, "FII: rendimentos isentos de IR para pessoa física."
    if asset_class == "acao_br":
        if share > 0:
            return JCP_IR_RATE * share, (
                f"Dividendos isentos; {share * 100:.0f}% estimados como JCP tributados em 15% na fonte."
            )
        return 0.0, "Dividendos isentos de IR para pessoa física."
    if asset_class in {"acao_us", "reit"} or str(currency).upper() == "USD":
        return US_WITHHOLDING_RATE, "Retenção de 30% nos EUA sobre dividendos."
    if asset_class == "etf":
        return 0.0, "ETF na B3: proventos considerados sem retenção; confirme o informe de rendimentos."
    return 0.0, "Sem retenção na fonte considerada."


def _cdi_annual_rate() -> float | None:
    try:
        daily = float(get_selic_rate())
        annual = (1 + daily) ** 252 - 1
    except Exception:
        logger.warning("Selic unavailable for cash-flow comparison", exc_info=True)
        return None
    return annual if math.isfinite(annual) and annual > 0 else None


def _month_totals(series: pd.Series) -> dict[int, float]:
    totals = {month: 0.0 for month in range(1, 13)}
    for day, amount in series.items():
        value = float(amount)
        if math.isfinite(value):
            totals[day.month] += value
    return totals


def _actual_month_totals(series: pd.Series) -> dict[tuple[int, int], float]:
    totals: dict[tuple[int, int], float] = {}
    for day, amount in series.items():
        value = float(amount)
        if math.isfinite(value):
            key = (day.year, day.month)
            totals[key] = totals.get(key, 0.0) + value
    return totals


def _base_asset(ticker: str, meta: Mapping[str, Any], weight: float, capital: float) -> dict[str, Any]:
    asset_class = str(meta.get("asset_class") or "outros")
    return {
        "ticker": ticker,
        "asset_class": asset_class,
        "asset_class_label": ASSET_CLASS_LABELS.get(asset_class, asset_class),
        "currency": str(meta.get("currency") or "BRL").upper(),
        "weight": weight,
        "position_value_brl": weight * capital,
        "income_mode": "not_applicable",
        "status": "not_applicable",
        "distributions_12m": 0,
        "dividend_per_share_brl": 0.0,
        "shares_estimate": None,
        "price_brl": None,
        "price_date": None,
        "annual_gross_brl": 0.0,
        "annual_net_brl": 0.0,
        "monthly_gross_brl": [0.0] * TRAILING_MONTHS,
        "monthly_net_brl": [0.0] * TRAILING_MONTHS,
        "tax_rate": 0.0,
        "tax_note": None,
        "yield_gross": 0.0,
        "yield_net": 0.0,
        "source": None,
        "note": None,
        "_monthly_by_calendar": {month: 0.0 for month in range(1, 13)},
        "_actual_by_month": {},
    }


def _public_asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in asset.items() if not key.startswith("_")}


def _modeled_asset(asset: dict[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    annual = spec.get("annual_gross_brl")
    try:
        annual_value = float(annual)
    except (TypeError, ValueError):
        annual_value = 0.0
    if not math.isfinite(annual_value) or annual_value < 0:
        annual_value = 0.0
    weights = spec.get("monthly_weights")
    if not isinstance(weights, (list, tuple)) or len(weights) != TRAILING_MONTHS:
        weights = [1 / TRAILING_MONTHS] * TRAILING_MONTHS
    cleaned: list[float] = []
    for value in weights:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        cleaned.append(number if math.isfinite(number) and number > 0 else 0.0)
    total = sum(cleaned)
    shares = [value / total for value in cleaned] if total > 0 else [1 / TRAILING_MONTHS] * TRAILING_MONTHS
    try:
        tax_rate = float(spec.get("tax_rate", 0.0))
    except (TypeError, ValueError):
        tax_rate = 0.0
    tax_rate = min(1.0, max(0.0, tax_rate)) if math.isfinite(tax_rate) else 0.0
    monthly = [annual_value * value for value in shares]
    asset.update({
        "income_mode": "modeled",
        "status": "modeled",
        "annual_gross_brl": annual_value,
        "annual_net_brl": annual_value * (1 - tax_rate),
        "monthly_gross_brl": monthly,
        "monthly_net_brl": [value * (1 - tax_rate) for value in monthly],
        "tax_rate": tax_rate,
        "tax_note": "Fluxo informado explicitamente como premissa de modelo.",
        "note": spec.get("note") or "Premissa de modelo informada pelo solicitante; não é provento observado.",
        "source": spec.get("source") or "Premissa do modelo",
    })
    return asset


def _project_asset(asset: dict[str, Any], history: Mapping[str, Any], price: Mapping[str, Any],
                   jcp_share: float) -> None:
    series = history.get("dividends_brl")
    if not isinstance(series, pd.Series) or series.empty:
        return
    price_value = price.get("price_brl")
    if price.get("status") != "ok" or not price_value:
        return
    price_value = float(price_value)
    if not math.isfinite(price_value) or price_value <= 0:
        return
    position = float(asset["position_value_brl"])
    shares = position / price_value
    month_totals = _month_totals(series)
    actual_totals = _actual_month_totals(series)
    dps_total = sum(month_totals.values())
    rate, tax_note = income_tax_rate(str(asset["asset_class"]), str(asset["currency"]), jcp_share)
    factor = 1 - rate
    monthly_gross = {month: shares * month_totals[month] for month in range(1, 13)}
    annual_gross = shares * dps_total
    asset.update({
        "income_mode": "dividends",
        "status": "ok",
        "distributions_12m": int(len(series)),
        "dividend_per_share_brl": dps_total,
        "shares_estimate": shares,
        "price_brl": price_value,
        "price_date": price.get("price_date"),
        "annual_gross_brl": annual_gross,
        "annual_net_brl": annual_gross * factor,
        "tax_rate": rate,
        "tax_note": tax_note,
        "yield_gross": annual_gross / position if position > 0 else 0.0,
        "yield_net": annual_gross * factor / position if position > 0 else 0.0,
        "source": history.get("source"),
        "note": history.get("note"),
        "_monthly_by_calendar": monthly_gross,
        "_actual_by_month": {key: shares * value for key, value in actual_totals.items()},
    })


def build_income_calendar(
    tickers,
    weights: Mapping[str, float],
    capital: float,
    *,
    cost_basis_brl: float | None = None,
    jcp_share: float = 0.0,
    today=None,
    modeled_income: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build trailing proventos, a 12-month projection, and portfolio metrics in BRL."""
    capital = float(capital)
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("Capital deve ser um valor positivo em reais.")
    jcp = float(jcp_share or 0.0)
    if not math.isfinite(jcp) or jcp < 0 or jcp > 1:
        raise ValueError("A parcela de JCP deve estar entre 0 e 1.")
    cost = None
    if cost_basis_brl is not None:
        cost = float(cost_basis_brl)
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("O custo de aquisição deve ser um valor positivo em reais.")

    normalized: dict[str, float] = {}
    for raw, raw_weight in (weights or {}).items():
        ticker = normalize_ticker(raw)
        weight = float(raw_weight or 0.0)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Peso inválido para {ticker}.")
        normalized[ticker] = normalized.get(ticker, 0.0) + weight

    now = _timestamp(today)
    current_month = now.replace(day=1)
    window_end = current_month
    window_start = current_month - pd.DateOffset(months=TRAILING_MONTHS)

    assets: list[dict[str, Any]] = []
    missing_tickers: list[str] = []
    no_distribution_tickers: list[str] = []
    not_applicable_tickers: list[str] = []
    seen = [normalize_ticker(ticker) for ticker in tickers]
    ordered = list(dict.fromkeys(seen))
    for ticker in normalized:
        if ticker not in ordered:
            ordered.append(ticker)

    for ticker in ordered:
        weight = float(normalized.get(ticker, 0.0))
        if weight <= 0:
            continue
        meta = asset_metadata(ticker)
        asset = _base_asset(ticker, meta, weight, capital)
        asset_class = str(asset["asset_class"])

        if ticker in TREASURY_TITLES:
            asset.update({
                "asset_class": "titulo",
                "asset_class_label": ASSET_CLASS_LABELS["titulo"],
                "currency": "BRL",
                "income_mode": "no_intermediate_coupons",
                "status": "no_intermediate_coupons",
                "note": TREASURY_NO_COUPON_NOTE,
                "source": "Tesouro Direto",
            })
            not_applicable_tickers.append(ticker)
            assets.append(asset)
            continue
        if is_synthetic_fixed_income(ticker):
            asset.update({
                "asset_class": "titulo",
                "asset_class_label": ASSET_CLASS_LABELS["titulo"],
                "note": "Exposição de renda fixa modelada: sem provento de dividendos observável.",
                "source": "Modelo B3Lab",
            })
            not_applicable_tickers.append(ticker)
            assets.append(asset)
            continue
        if asset_class == "titulo":
            asset.update({
                "note": "Título ou fundo de renda fixa: distribuições não tratadas como provento de dividendos.",
                "source": meta.get("yahoo_symbol") and "Yahoo Finance" or "Modelo B3Lab",
            })
            not_applicable_tickers.append(ticker)
            assets.append(asset)
            continue
        if not supports_share_allocation(ticker):
            asset.update({"note": "Instrumento cotado sem provento de dividendos próprio."})
            not_applicable_tickers.append(ticker)
            assets.append(asset)
            continue
        flagged = (modeled_income or {}).get(ticker)
        if flagged is not None:
            assets.append(_modeled_asset(asset, flagged))
            continue

        try:
            history = get_asset_dividend_history(ticker, window_start.date())
        except Exception:
            logger.warning("Dividend history failed for %s", ticker, exc_info=True)
            history = _empty_history(
                ticker,
                status="missing",
                note=f"Fonte de proventos indisponível para {ticker}.",
                currency=str(asset["currency"]),
            )
        if history.get("status") == "missing":
            asset.update({"status": "missing", "note": history.get("note"), "source": history.get("source")})
            missing_tickers.append(ticker)
            assets.append(asset)
            continue
        if history.get("status") == "no_intermediate_coupons":
            asset.update({
                "income_mode": "no_intermediate_coupons",
                "status": "no_intermediate_coupons",
                "note": history.get("note") or TREASURY_NO_COUPON_NOTE,
                "source": history.get("source"),
            })
            not_applicable_tickers.append(ticker)
            assets.append(asset)
            continue
        if history.get("status") != "ok":
            asset.update({
                "income_mode": "dividends",
                "status": history.get("status", "no_distributions"),
                "note": history.get("note") or "Sem distribuições no período.",
                "source": history.get("source"),
            })
            no_distribution_tickers.append(ticker)
            assets.append(asset)
            continue
        series = history.get("dividends_brl")
        series = series.loc[series.index < window_end] if isinstance(series, pd.Series) else pd.Series(dtype=float)
        if series.empty:
            asset.update({
                "income_mode": "dividends",
                "status": "no_distributions",
                "note": "Sem distribuições nos 12 meses completos anteriores.",
                "source": history.get("source"),
            })
            no_distribution_tickers.append(ticker)
            assets.append(asset)
            continue
        history = {**history, "dividends_brl": series}

        try:
            price = get_asset_price_brl(ticker)
        except Exception:
            logger.warning("Current price failed for %s", ticker, exc_info=True)
            price = {"status": "missing", "price_brl": None, "price_date": None, "note": None}
        quoted = price.get("price_brl")
        try:
            quoted = float(quoted)
        except (TypeError, ValueError):
            quoted = None
        if price.get("status") != "ok" or quoted is None or not math.isfinite(quoted) or quoted <= 0:
            asset.update({
                "status": "missing",
                "note": price.get("note") or f"Preço atual indisponível para {ticker}; proventos não projetados.",
                "source": history.get("source"),
            })
            missing_tickers.append(ticker)
            assets.append(asset)
            continue
        _project_asset(asset, history, price, jcp)
        if asset.get("status") != "ok":
            missing_tickers.append(ticker)
        assets.append(asset)

    calendar = []
    projection_start = current_month
    for offset in range(TRAILING_MONTHS):
        month_stamp = projection_start + pd.DateOffset(months=offset)
        month_index = month_stamp.month
        contributors = []
        gross = 0.0
        net = 0.0
        for asset in assets:
            for_month = asset.get("_monthly_by_calendar", {}).get(month_index, 0.0)
            if asset.get("income_mode") == "modeled":
                for_month = asset["monthly_gross_brl"][offset]
                asset_net = asset["monthly_net_brl"][offset]
            else:
                asset_net = for_month * (1 - float(asset.get("tax_rate", 0.0)))
            if for_month <= 0:
                continue
            gross += for_month
            net += asset_net
            contributors.append({
                "ticker": asset["ticker"],
                "gross_brl": for_month,
                "net_brl": asset_net,
            })
        contributors.sort(key=lambda item: item["gross_brl"], reverse=True)
        calendar.append({
            "month_label": _month_label(month_stamp.year, month_index),
            "month_index": month_index,
            "year": int(month_stamp.year),
            "gross_brl": gross,
            "net_brl": net,
            "contributors": contributors,
        })

    trailing_calendar = []
    cursor = window_start
    for _ in range(TRAILING_MONTHS):
        month_index = cursor.month
        gross = 0.0
        net = 0.0
        contributors = []
        for asset in assets:
            actual = asset.get("_actual_by_month", {}).get((cursor.year, month_index), 0.0)
            if actual <= 0:
                continue
            asset_net = actual * (1 - float(asset.get("tax_rate", 0.0)))
            gross += actual
            net += asset_net
            contributors.append({"ticker": asset["ticker"], "gross_brl": actual, "net_brl": asset_net})
        contributors.sort(key=lambda item: item["gross_brl"], reverse=True)
        trailing_calendar.append({
            "month_label": _month_label(cursor.year, month_index),
            "month_index": month_index,
            "year": int(cursor.year),
            "gross_brl": gross,
            "net_brl": net,
            "contributors": contributors,
        })
        cursor = cursor + pd.DateOffset(months=1)

    annual_gross = sum(float(asset.get("annual_gross_brl", 0.0)) for asset in assets)
    annual_net = sum(float(asset.get("annual_net_brl", 0.0)) for asset in assets)
    monthly_average_net = annual_net / TRAILING_MONTHS
    portfolio_yield_gross = annual_gross / capital
    portfolio_yield_net = annual_net / capital
    yield_on_cost = annual_net / cost if cost else None
    cdi_annual = _cdi_annual_rate()
    cdi_net = cdi_annual * (1 - CDI_IR_RATE) if cdi_annual is not None else None
    ratio = (
        portfolio_yield_net / cdi_net
        if cdi_net is not None and math.isfinite(cdi_net) and cdi_net > 0
        else None
    )

    public_assets = [_public_asset(asset) for asset in assets]
    projection_end = projection_start + pd.DateOffset(months=TRAILING_MONTHS) - pd.Timedelta(days=1)
    return {
        "as_of": now.date().isoformat(),
        "capital": capital,
        "cost_basis_brl": cost,
        "jcp_share": jcp,
        "currency": "BRL",
        "window": {
            "trailing_start": window_start.date().isoformat(),
            "trailing_end": (window_end - pd.Timedelta(days=1)).date().isoformat(),
            "projection_start": projection_start.date().isoformat(),
            "projection_end": projection_end.date().isoformat(),
        },
        "annual_gross_brl": annual_gross,
        "annual_net_brl": annual_net,
        "monthly_average_net_brl": monthly_average_net,
        "portfolio_yield_gross": portfolio_yield_gross,
        "portfolio_yield_net": portfolio_yield_net,
        "yield_on_cost_net": yield_on_cost,
        "cdi_annual_rate": cdi_annual,
        "cdi_net_rate": cdi_net,
        "cdi_ir_rate": CDI_IR_RATE,
        "yield_vs_cdi_ratio": ratio,
        "assets": public_assets,
        "calendar": calendar,
        "trailing_calendar": trailing_calendar,
        "missing_tickers": sorted(set(missing_tickers)),
        "no_distribution_tickers": sorted(set(no_distribution_tickers)),
        "not_applicable_tickers": sorted(set(not_applicable_tickers)),
        "chart": income_calendar_chart(calendar),
        "methodology": METHODOLOGY,
        "tax_notes": [
            "FIIs: rendimentos isentos de IR para pessoa física.",
            "Ações B3: dividendos isentos; parcela de JCP estimada tributada em 15% na fonte.",
            "Ações, REITs e ETFs em USD: retenção de 30% nos EUA.",
            "CDI líquido usa premissa de 15% de IR sobre a Selic anualizada do BCB.",
        ],
    }


def income_calendar_chart(calendar) -> go.Figure:
    """Themed bar figure of monthly net cash flow, ready for the shared frontend chart."""
    labels = [item["month_label"] for item in calendar]
    net = [item["net_brl"] for item in calendar]
    gross = [item["gross_brl"] for item in calendar]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=gross,
        name="Bruto",
        marker_color=CHART_PRIMARY_TRACE,
        hovertemplate="%{x}<br>Bruto: R$ %{y:,.2f}<extra></extra>",
    )
    fig.add_bar(
        x=labels,
        y=net,
        name="Líquido",
        marker_color=CHART_PRIMARY,
        hovertemplate="%{x}<br>Líquido: R$ %{y:,.2f}<extra></extra>",
    )
    fig.update_layout(
        title="Fluxo de caixa mensal projetado",
        barmode="overlay",
        bargap=0.32,
        xaxis_title="Mês",
        yaxis_title="R$ por mês",
        showlegend=True,
    )
    return apply_plotly_theme(fig)
