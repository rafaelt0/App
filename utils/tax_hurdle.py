"""After-tax and real-return hurdle engine (WP-1.1).

Pure, deterministic financial policy for the portfolio hurdle comparison. The
engine never fabricates a market series: a missing CDI, inflation, SOFR or CPI
series stays ``None`` and is reported through an explicit warning, while the
remaining metrics keep their real units.

Conventions
-----------
* Returns, rates and gains are decimal fractions (``0.12`` is 12%).
* ``pp`` values are already multiplied by 100 and carry the ``pp`` unit.
* Ratios (Sharpe) are unitless multiples.
* Real returns use the exact Fisher relation ``(1 + real) = (1 + nominal) / (1 + inflation)``.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

from utils.assets import ASSET_CLASS_LABELS, asset_metadata, normalize_ticker
from utils.cache import cache_data
from utils.charts import CHART_ACCENT, CHART_PRIMARY, apply_plotly_theme
from utils.portfolio_data import align_daily_rates
from utils.regime import (
    ANNUALIZATION_DAYS,
    HURDLE_METADATA,
    TAX_PROFILE_METADATA,
    default_hurdle_label,
    hurdle_metadata,
)
from utils.treasury_data import TREASURY_TITLES

logger = logging.getLogger(__name__)

#: Brazilian regressive individual income tax (IR) for fixed income, by holding days.
BR_IR_TIERS: tuple[tuple[int | None, float], ...] = (
    (180, 0.225),
    (360, 0.200),
    (720, 0.175),
    (None, 0.150),
)
FIXED_INCOME_EXEMPT_PREFIXES: tuple[str, ...] = ("LCI-", "LCA-", "CRI-", "CRA-")
FIXED_INCOME_EXEMPT_TICKERS: frozenset[str] = frozenset({"POUPANCA"})
FIXED_INCOME_TAXABLE_PREFIXES: tuple[str, ...] = ("TD-", "CDB-", "TESOURO-", "PRE-", "IPCA+")
FIXED_INCOME_TAXABLE_TICKERS: frozenset[str] = frozenset({"CDI", "SELIC"})
BR_EQUITY_RATE = 0.15
BR_FII_RATE = 0.20
BR_ETF_RATE = 0.15
BR_OFFSHORE_RATE = 0.15
US_LONG_TERM_RATE = 0.15
US_SHORT_TERM_RATE = 0.24
DEFAULT_FX_DRAG = 0.0038

BCB_SERIES_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
IPCA_SERIES = 433
IPCA15_SERIES = 13522
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SOFR_SERIES = "SOFR"
US_CPI_SERIES = "CPIAUCSL"
_REQUEST_TIMEOUT_SECONDS = 15


def _percent_text(rate: float) -> str:
    return f"{rate * 100:.1f}".replace(".", ",") + "%"


# ---------------------------------------------------------------------------
# Tax model
# ---------------------------------------------------------------------------


def regressive_ir_rate(holding_days: int) -> float:
    """Brazilian regressive IR rate for fixed income at a given holding period."""
    days = max(int(holding_days or 0), 0)
    for limit, rate in BR_IR_TIERS:
        if limit is None or days <= limit:
            return rate
    return BR_IR_TIERS[-1][1]


def _regressive_bucket(holding_days: int) -> str:
    days = max(int(holding_days or 0), 0)
    if days <= 180:
        return "até 180 dias"
    if days <= 360:
        return "181 a 360 dias"
    if days <= 720:
        return "361 a 720 dias"
    return "acima de 720 dias"


@dataclass(frozen=True)
class TaxAsset:
    """Asset classification used by the tax rules."""

    ticker: str
    asset_class: str
    category: str
    currency: str
    label: str


@dataclass(frozen=True)
class TaxRule:
    """One tax rule applied to an asset's positive cumulative gain."""

    code: str
    label: str
    rate: float


def classify_asset(ticker: str) -> TaxAsset:
    """Classify a ticker using the shared catalog plus the fixed-income prefixes.

    ``utils/assets`` exposes the canonical ``asset_metadata``/``ASSET_CLASSES``
    catalog; fixed income uses the same ticker prefixes and ``TREASURY_TITLES``
    that the providers understand (``LCI-``, ``LCA-``, ``CRI-``, ``CRA-``,
    ``POUPANCA``, ``CDB-``, ``TD-``, ``TESOURO-``, ``PRE-``, ``IPCA+``, ``CDI``,
    ``SELIC``).
    """
    canonical = normalize_ticker(ticker)
    upper = canonical.upper()
    fixed_income_label = "Renda fixa brasileira"
    if canonical in FIXED_INCOME_EXEMPT_TICKERS or upper.startswith(FIXED_INCOME_EXEMPT_PREFIXES):
        return TaxAsset(canonical, "titulo", "fixed_income_exempt", "BRL", fixed_income_label)
    if (
        canonical in FIXED_INCOME_TAXABLE_TICKERS
        or canonical in TREASURY_TITLES
        or upper.startswith(FIXED_INCOME_TAXABLE_PREFIXES)
    ):
        return TaxAsset(canonical, "titulo", "fixed_income_taxable", "BRL", fixed_income_label)

    meta = asset_metadata(canonical)
    asset_class = str(meta.get("asset_class") or "outros")
    currency = str(meta.get("currency") or "BRL").upper()
    label = str(meta.get("sector") or meta.get("company") or canonical)
    category = {
        "acao_br": "equity_br",
        "fii": "fii",
        "acao_us": "offshore",
        "reit": "offshore",
        "etf": "etf",
        "outros": "offshore",
        "titulo": "etf",  # fixed income ETFs/BDRs listed on B3 follow the 15% ETF rule
    }.get(asset_class, "offshore")
    return TaxAsset(canonical, asset_class, category, currency, label)


def tax_rule(
    asset: TaxAsset,
    tax_profile: str,
    holding_days: int,
    *,
    small_sale_exempt: bool = False,
) -> TaxRule:
    """Tax rule and rate for one asset under the chosen profile."""
    if tax_profile not in TAX_PROFILE_METADATA:
        raise ValueError(f"Regime tributário não suportado: {tax_profile!r}")

    if tax_profile == "NONE":
        return TaxRule("exempt", "Regime isento — 0% de IR", 0.0)

    if tax_profile == "US_TAXABLE":
        long_term = int(holding_days or 0) > 365
        rate = US_LONG_TERM_RATE if long_term else US_SHORT_TERM_RATE
        horizon = "longo prazo (>365 dias)" if long_term else "curto prazo (até 365 dias)"
        return TaxRule("us_capital_gains", f"Ganho de capital EUA — {horizon} ({_percent_text(rate)})", rate)

    # BR_PF
    if asset.category == "fixed_income_exempt":
        return TaxRule("br_fi_exempt", "Renda fixa isenta (LCI/LCA/CRI/CRA/Poupança) — 0% de IR", 0.0)
    if asset.category == "fixed_income_taxable":
        rate = regressive_ir_rate(holding_days)
        return TaxRule(
            "br_fi_regressive",
            f"IR regressivo de renda fixa ({_regressive_bucket(holding_days)}) — {_percent_text(rate)}",
            rate,
        )
    if asset.category == "equity_br":
        if small_sale_exempt:
            return TaxRule(
                "br_equity_small_sales",
                "Ações B3 — isenção para vendas de até R$ 20.000,00 no mês (0% de IR)",
                0.0,
            )
        return TaxRule("br_equity", "Ações B3 — ganho de capital (15% de IR)", BR_EQUITY_RATE)
    if asset.category == "fii":
        return TaxRule("br_fii", "FIIs — ganho de capital (20% de IR); dividendos isentos", BR_FII_RATE)
    if asset.category == "etf":
        return TaxRule("br_etf", "ETFs e BDRs na B3 — ganho de capital (15% de IR)", BR_ETF_RATE)
    return TaxRule(
        "br_offshore",
        "Ativos no exterior — Lei 14.754/2023, ganho de capital (15% de IR)",
        BR_OFFSHORE_RATE,
    )


@dataclass
class PortfolioTaxResult:
    """Aggregate tax drag and the per-asset evidence that produced it."""

    gross_return: float
    gross_gain: float
    tax_amount: float
    fx_drag: float
    total_drag: float
    net_return: float
    effective_tax_rate: float
    breakdown: list[dict[str, object]]
    assumptions: list[str]
    warnings: list[str]


def _assumption_lines(profile: str, small_sale_exempt: bool, breakdown: Sequence[Mapping[str, object]]) -> list[str]:
    categories = {str(row["tax_category"]) for row in breakdown}
    lines = [
        "Tributação aplicada ao ganho acumulado de cada ativo, ponderado pelo peso atual da carteira; vendas parciais, come-cotas e compensações entre classes não são modelados.",
        "O IR incide apenas sobre ganho positivo: prejuízo não gera imposto nem crédito.",
    ]
    if profile == "BR_PF":
        if "fixed_income_taxable" in categories:
            lines.append(
                "IR regressivo de renda fixa: 22,5% até 180 dias, 20,0% de 181 a 360 dias, 17,5% de 361 a 720 dias e 15,0% acima de 720 dias."
            )
        if "fixed_income_exempt" in categories:
            lines.append("LCI, LCA, CRI, CRA e Poupança são tratados como isentos de IR para pessoa física.")
        if "equity_br" in categories:
            lines.append(
                "Ações B3: 15% sobre o ganho positivo; a isenção de R$ 20.000,00 em vendas mensais só é aplicada quando o parâmetro small_sale_exempt é verdadeiro."
                if small_sale_exempt
                else "Ações B3: 15% sobre o ganho positivo; a isenção de R$ 20.000,00 em vendas mensais não foi aplicada nesta análise."
            )
        if "fii" in categories:
            lines.append("FIIs: 20% sobre o ganho de preço; dividendos de FIIs são isentos e não entram no cálculo.")
        if "etf" in categories:
            lines.append("ETFs e BDRs negociados na B3: 15% sobre o ganho positivo.")
        if "offshore" in categories:
            lines.append(
                "Ativos no exterior: 15% da Lei 14.754/2023 sobre o ganho positivo, com custo cambial/IOF estimado de 0,38% (ida e volta)."
            )
    elif profile == "US_TAXABLE":
        lines.append("Ganhos nos EUA: 15% para posições acima de 365 dias e 24% até 365 dias.")
    else:
        lines.append("Regime isento: nenhum imposto é aplicado ao portfólio.")
    return lines


def portfolio_tax_analysis(
    weights: Mapping[str, float],
    cumulative_returns: Mapping[str, float],
    *,
    tax_profile: str = "BR_PF",
    holding_days: int = 0,
    small_sale_exempt: bool = False,
    fx_drag: float = DEFAULT_FX_DRAG,
) -> PortfolioTaxResult:
    """Apply per-asset tax rules to weighted cumulative gains.

    ``weights`` may be keyed by provider symbol or canonical ticker; both are
    normalized before matching. Negative gains are never taxed.
    """
    if fx_drag < 0 or not math.isfinite(fx_drag):
        raise ValueError("O custo cambial/IOF deve ser um número finito e não negativo.")

    normalized_weights: dict[str, float] = {}
    for ticker, weight in weights.items():
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        canonical = normalize_ticker(ticker)
        normalized_weights[canonical] = normalized_weights.get(canonical, 0.0) + value

    normalized_returns = {normalize_ticker(ticker): float(value) for ticker, value in cumulative_returns.items()}
    total_weight = sum(normalized_weights.values())
    if total_weight <= 0:
        raise ValueError("A carteira precisa de ao menos um peso positivo.")

    warnings: list[str] = []
    breakdown: list[dict[str, object]] = []
    gross_return = 0.0
    tax_amount = 0.0
    fx_total = 0.0
    for ticker, weight in normalized_weights.items():
        if ticker not in normalized_returns:
            warnings.append(f"Sem retorno acumulado para {ticker}; o ativo não entra no cálculo tributário.")
            continue
        asset_return = normalized_returns[ticker]
        if not math.isfinite(asset_return):
            warnings.append(f"Retorno inválido para {ticker}; o ativo não entra no cálculo tributário.")
            continue
        normalized_weight = weight / total_weight
        asset = classify_asset(ticker)
        rule = tax_rule(asset, tax_profile, holding_days, small_sale_exempt=small_sale_exempt)
        contribution = normalized_weight * asset_return
        taxable_gain = max(contribution, 0.0)
        tax = taxable_gain * rule.rate
        asset_fx_drag = taxable_gain * fx_drag if asset.category == "offshore" and tax_profile != "NONE" else 0.0
        gross_return += contribution
        tax_amount += tax
        fx_total += asset_fx_drag
        breakdown.append(
            {
                "ticker": ticker,
                "asset_class": asset.asset_class,
                "asset_class_label": ASSET_CLASS_LABELS.get(asset.asset_class, asset.asset_class),
                "tax_category": asset.category,
                "currency": asset.currency,
                "tax_rule": rule.code,
                "tax_rule_label": rule.label,
                "applied_rate": rule.rate,
                "weight": normalized_weight,
                "gross_return": asset_return,
                "taxable_gain": taxable_gain,
                "tax_amount": tax,
                "fx_drag_amount": asset_fx_drag,
                "total_drag": tax + asset_fx_drag,
            }
        )

    breakdown.sort(key=lambda row: (-float(row["weight"]), str(row["ticker"])))
    gross_gain = gross_return
    total_drag = tax_amount + fx_total
    effective_rate = tax_amount / gross_gain if gross_gain > 0 else 0.0
    return PortfolioTaxResult(
        gross_return=gross_return,
        gross_gain=gross_gain,
        tax_amount=tax_amount,
        fx_drag=fx_total,
        total_drag=total_drag,
        net_return=gross_return - total_drag,
        effective_tax_rate=effective_rate,
        breakdown=breakdown,
        assumptions=_assumption_lines(tax_profile, small_sale_exempt, breakdown),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Inflation and hurdle series
# ---------------------------------------------------------------------------


def deflate_return(nominal_return: float | None, inflation_return: float | None) -> float | None:
    """Exact Fisher relation for cumulative returns."""
    if nominal_return is None or inflation_return is None:
        return None
    if not (math.isfinite(nominal_return) and math.isfinite(inflation_return)):
        return None
    if inflation_return <= -1:
        return None
    return (1 + nominal_return) / (1 + inflation_return) - 1


def deflate_series(nominal: pd.Series, inflation: pd.Series) -> pd.Series:
    """Deflate daily nominal returns with the exact Fisher relation."""
    common = nominal.index.intersection(inflation.index)
    return (1 + nominal.loc[common]) / (1 + inflation.loc[common]) - 1


def cumulative_return(series: pd.Series) -> float:
    """Compounded cumulative return of a daily fraction series."""
    values = pd.Series(series).dropna()
    if values.empty:
        return 0.0
    return float((1 + values).prod() - 1)


@cache_data(ttl=3600)
def get_bcb_monthly_inflation(series_code: int = IPCA_SERIES, start_date: dt.date | None = None) -> pd.Series | None:
    """Monthly BCB inflation as fractions per month, or ``None`` when the source fails."""
    start = pd.Timestamp(start_date or (dt.date.today() - dt.timedelta(days=365 * 6)))
    try:
        response = requests.get(
            BCB_SERIES_URL.format(code=series_code),
            params={"formato": "json", "dataInicial": f"{start:%d/%m/%Y}", "dataFinal": f"{dt.date.today():%d/%m/%Y}"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            raise ValueError("BCB retornou série de inflação vazia.")
        index = pd.to_datetime([row["data"] for row in rows], format="%d/%m/%Y")
        values = np.array([float(row["valor"]) for row in rows], dtype=float) / 100
        if not np.isfinite(values).all() or (values <= -1).any():
            raise ValueError("BCB retornou valores de inflação inválidos.")
        series = pd.Series(values, index=index.to_period("M").to_timestamp("M")).sort_index()
        return series[~series.index.duplicated(keep="last")]
    except Exception as exc:  # noqa: BLE001 - provider failure must stay a None, not a fake number
        logger.warning("Inflação BCB SGS %s indisponível: %s", series_code, exc)
        return None


def monthly_to_daily(monthly: pd.Series | None) -> pd.Series | None:
    """Spread each monthly fraction evenly across the calendar days of its month."""
    if monthly is None or monthly.empty:
        return None
    result: list[pd.Series] = []
    for stamp, rate in monthly.items():
        days = int(pd.Period(stamp, freq="M").days_in_month)
        daily = (1 + float(rate)) ** (1 / days) - 1
        days_index = pd.period_range(pd.Timestamp(stamp).replace(day=1), pd.Timestamp(stamp), freq="D").to_timestamp()
        result.append(pd.Series(daily, index=days_index))
    frame = pd.concat(result).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def inflation_daily_series(
    index: pd.DatetimeIndex,
    *,
    monthly: pd.Series | None = None,
    series_code: int = IPCA_SERIES,
    warnings: list[str] | None = None,
) -> pd.Series | None:
    """Daily inflation aligned to ``index``; returns ``None`` when unknown dates remain."""
    if len(index) == 0:
        return None
    if monthly is None:
        monthly = get_bcb_monthly_inflation(series_code, pd.Timestamp(index[0]).date())
    daily = monthly_to_daily(monthly)
    if daily is None or daily.empty:
        message = f"Série de inflação (BCB SGS {series_code}) indisponível; o retorno real não pode ser calculado."
        if warnings is not None and message not in warnings:
            warnings.append(message)
        return None
    aligned = daily.reindex(index)
    missing = int(aligned.isna().sum())
    if missing:
        message = f"A série de inflação não cobre {missing} data(s) da carteira; o retorno real não pode ser calculado."
        if warnings is not None and message not in warnings:
            warnings.append(message)
        return None
    return aligned


@cache_data(ttl=3600)
def get_fred_series(series_id: str = SOFR_SERIES) -> pd.Series | None:
    """FRED CSV series, or ``None`` on any provider failure (never a fabricated value)."""
    try:
        response = requests.get(FRED_CSV_URL, params={"id": series_id}, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if frame.empty or len(frame.columns) < 2:
            raise ValueError(f"FRED retornou série vazia para {series_id}.")
        dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
        values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
        series = pd.Series(values.to_numpy(), index=dates).dropna().sort_index()
        if series.empty:
            raise ValueError(f"FRED retornou série vazia para {series_id}.")
        return series
    except Exception as exc:  # noqa: BLE001
        logger.warning("Série FRED %s indisponível: %s", series_id, exc)
        return None


def _sofr_daily_series(index: pd.DatetimeIndex, warnings: list[str] | None = None) -> pd.Series | None:
    raw = get_fred_series(SOFR_SERIES)
    if raw is None or raw.empty:
        message = "SOFR indisponível na FRED; a comparação com esse hurdle não pode ser calculada."
        if warnings is not None and message not in warnings:
            warnings.append(message)
        return None
    rates = raw / 100
    try:
        return align_daily_rates(rates, index)
    except ValueError:
        message = "A série SOFR não cobre todas as datas da carteira."
        if warnings is not None and message not in warnings:
            warnings.append(message)
        return None


def _us_cpi_daily_series(index: pd.DatetimeIndex, warnings: list[str] | None = None) -> pd.Series | None:
    monthly = get_fred_series(US_CPI_SERIES)
    if monthly is None or monthly.empty:
        message = "CPI dos EUA indisponível na FRED; a comparação com esse hurdle não pode ser calculada."
        if warnings is not None and message not in warnings:
            warnings.append(message)
        return None
    cpi_monthly = monthly.pct_change().dropna()
    daily = monthly_to_daily(cpi_monthly)
    if daily is None:
        return None
    aligned = daily.reindex(index)
    if aligned.isna().any():
        message = "A série de CPI dos EUA não cobre todas as datas da carteira."
        if warnings is not None and message not in warnings:
            warnings.append(message)
        return None
    return aligned


def _daily_from_annual(rate: float, annualization: int = ANNUALIZATION_DAYS) -> float:
    return (1 + rate) ** (1 / annualization) - 1


def build_hurdle_series(
    hurdle: str,
    index: pd.DatetimeIndex,
    *,
    real_return: bool = False,
    cdi_rates: pd.Series | None = None,
    inflation_daily: pd.Series | None = None,
    sofr_daily: pd.Series | None = None,
    us_cpi_daily: pd.Series | None = None,
    annualization: int = ANNUALIZATION_DAYS,
    warnings: list[str] | None = None,
) -> tuple[pd.Series | None, str]:
    """Daily hurdle returns aligned to ``index``, in the requested regime.

    Returns ``(series, source)``. ``series`` is ``None`` when the required
    provider series is unavailable.
    """
    if hurdle not in HURDLE_METADATA:
        raise ValueError(f"Hurdle não suportado: {hurdle!r}")
    source = str(hurdle_metadata(hurdle)["source"])
    if len(index) == 0:
        return None, source

    if hurdle == "CDI":
        rates = cdi_rates
        if rates is None:
            from utils.portfolio_data import get_cdi_daily_rates

            try:
                rates = get_cdi_daily_rates(index[0])
            except Exception as exc:  # noqa: BLE001
                message = f"CDI diário indisponível no BCB ({exc}); o hurdle não pode ser calculado."
                if warnings is not None and message not in warnings:
                    warnings.append(message)
                return None, source
        try:
            nominal = align_daily_rates(rates, index)
        except ValueError:
            message = "A série de CDI não cobre todas as datas da carteira."
            if warnings is not None and message not in warnings:
                warnings.append(message)
            return None, source
        source = "BCB SGS 12 (CDI diário)"
        if real_return:
            inflation = inflation_daily if inflation_daily is not None else inflation_daily_series(index, warnings=warnings)
            if inflation is None:
                return None, source
            return deflate_series(nominal, inflation), source
        return nominal, source

    if hurdle == "IPCA_PLUS_6":
        real_daily = _daily_from_annual(0.06, annualization)
        real_series = pd.Series(real_daily, index=index)
        if real_return:
            return real_series, source
        inflation = inflation_daily if inflation_daily is not None else inflation_daily_series(index, warnings=warnings)
        if inflation is None:
            return None, source
        return (1 + real_series) * (1 + inflation) - 1, source

    if hurdle == "SOFR":
        nominal = sofr_daily if sofr_daily is not None else _sofr_daily_series(index, warnings)
        if nominal is None:
            return None, source
        if real_return:
            cpi = us_cpi_daily if us_cpi_daily is not None else _us_cpi_daily_series(index, warnings)
            if cpi is None:
                return None, source
            return deflate_series(nominal, cpi), source
        return nominal, source

    # US_CPI: the hurdle is inflation itself.
    cpi = us_cpi_daily if us_cpi_daily is not None else _us_cpi_daily_series(index, warnings)
    if cpi is None:
        return None, source
    if real_return:
        return pd.Series(0.0, index=index), source
    return cpi, source


# ---------------------------------------------------------------------------
# Hurdle analysis
# ---------------------------------------------------------------------------


def _annualized(total_return: float | None, observations: int, annualization: int) -> float | None:
    if total_return is None or observations <= 0 or total_return <= -1 or not math.isfinite(total_return):
        return None
    return (1 + total_return) ** (annualization / observations) - 1


def _volatility(daily: pd.Series, annualization: int) -> float | None:
    if len(daily) < 2:
        return None
    value = float(daily.std(ddof=1)) * math.sqrt(annualization)
    return value if math.isfinite(value) else None


def _net_hurdle(
    total: float,
    tax_profile: str,
    holding_days: int,
) -> float:
    """Net-of-tax hurdle total return under the selected profile."""
    if tax_profile == "NONE" or total <= 0:
        return total
    if tax_profile == "BR_PF":
        rate = regressive_ir_rate(holding_days)
    else:
        rate = US_LONG_TERM_RATE if int(holding_days) > 365 else US_SHORT_TERM_RATE
    return total - total * rate


def _metric(value: float | None, unit: str) -> dict[str, float | None | str]:
    return {"value": None if value is None else float(value), "unit": unit}


def _wealth_figure(
    portfolio_daily: pd.Series,
    hurdle_daily: pd.Series | None,
    *,
    hurdle_label: str,
    real_return: bool,
    net_of_tax: bool,
) -> go.Figure:
    base = pd.Timestamp(portfolio_daily.index[0]) - pd.Timedelta(days=1)
    portfolio_wealth = pd.concat(
        [pd.Series([100.0], index=pd.DatetimeIndex([base])), (1 + portfolio_daily).cumprod() * 100]
    )
    fig = go.Figure(
        go.Scatter(
            x=portfolio_wealth.index,
            y=portfolio_wealth.values,
            mode="lines",
            name="Portfólio",
            line=dict(color=CHART_PRIMARY, width=2),
            hovertemplate="%{x|%d/%m/%Y}<br>Portfólio: %{y:.2f}<extra></extra>",
        )
    )
    if hurdle_daily is not None and not hurdle_daily.empty:
        hurdle_wealth = pd.concat(
            [pd.Series([100.0], index=pd.DatetimeIndex([base])), (1 + hurdle_daily).cumprod() * 100]
        )
        fig.add_trace(
            go.Scatter(
                x=hurdle_wealth.index,
                y=hurdle_wealth.values,
                mode="lines",
                name=f"Hurdle · {hurdle_label}",
                line=dict(color=CHART_ACCENT, width=1.6, dash="dash"),
                hovertemplate=f"%{{x|%d/%m/%Y}}<br>{hurdle_label}: %{{y:.2f}}<extra></extra>",
            )
        )
    regime = "real (deflacionado)" if real_return else "nominal"
    tax = "líquida de IR" if net_of_tax else "bruta de IR"
    fig.update_layout(
        title=f"Riqueza acumulada base 100 — {regime}, {tax}",
        xaxis_title="Data",
        yaxis_title="Base 100",
        height=360,
        margin=dict(t=56),
        hovermode="x unified",
    )
    apply_plotly_theme(fig)
    return fig


def analyze_hurdle(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    hurdle: str = "CDI",
    tax_profile: str = "BR_PF",
    real_return: bool = False,
    net_of_tax: bool = True,
    small_sale_exempt: bool = False,
    base_currency: str = "BRL",
    holding_days: int | None = None,
    cdi_rates: pd.Series | None = None,
    inflation_daily: pd.Series | None = None,
    sofr_daily: pd.Series | None = None,
    us_cpi_daily: pd.Series | None = None,
    hurdle_series: pd.Series | None = None,
    fx_drag: float = DEFAULT_FX_DRAG,
    annualization: int = ANNUALIZATION_DAYS,
) -> dict[str, object]:
    """Compare portfolio returns (net/real as requested) against a hurdle.

    ``returns`` is a daily simple-return frame whose columns are provider
    symbols; ``weights`` is keyed by canonical ticker. Provider series may be
    injected for deterministic tests; otherwise real providers are consulted
    and a failure becomes an explicit warning.
    """
    if returns is None or returns.empty:
        raise ValueError("O histórico de retornos da carteira está vazio.")
    frame = returns.dropna(how="all")
    if frame.empty:
        raise ValueError("O histórico de retornos da carteira está vazio.")
    index = pd.DatetimeIndex(frame.index)
    warnings: list[str] = []

    if tax_profile not in TAX_PROFILE_METADATA:
        raise ValueError(f"Regime tributário não suportado: {tax_profile!r}")
    normalized = {}
    for ticker, weight in weights.items():
        value = float(weight)
        if math.isfinite(value) and value > 0:
            normalized[normalize_ticker(ticker)] = value
    active: dict[str, float] = {}
    for column in frame.columns:
        weight = normalized.get(normalize_ticker(column))
        if weight is None:
            warnings.append(f"Peso não encontrado para {normalize_ticker(column)}; o ativo foi excluído da comparação.")
            continue
        active[column] = weight
    if not active:
        raise ValueError("Nenhum ativo da carteira possui peso e histórico em comum.")
    weight_total = sum(active.values())
    if weight_total <= 0:
        raise ValueError("A carteira precisa de ao menos um peso positivo.")
    active = {column: weight / weight_total for column, weight in active.items()}

    daily_gross = frame.loc[:, list(active)].mul(pd.Series(active), axis=1).sum(axis=1)
    gross_total = cumulative_return(daily_gross)
    observations = len(daily_gross)
    if holding_days is None:
        holding_days = max((index[-1] - index[0]).days, 1)

    asset_cumulative = {column: cumulative_return(frame[column]) for column in active}
    taxes = portfolio_tax_analysis(
        active,
        asset_cumulative,
        tax_profile=tax_profile,
        holding_days=holding_days,
        small_sale_exempt=small_sale_exempt,
        fx_drag=fx_drag,
    )
    warnings.extend(taxes.warnings)

    tax_drag_applied = taxes.total_drag if net_of_tax else 0.0
    net_total_nominal = gross_total - tax_drag_applied
    denominator = 1 + gross_total
    tax_factor = (1 + net_total_nominal) / denominator if denominator > 0 else 1.0
    daily_effective = ((1 + daily_gross) * tax_factor - 1) if tax_drag_applied else daily_gross

    inflation_series: pd.Series | None = None
    total_return = net_total_nominal
    daily_final: pd.Series | None = daily_effective
    inflation_source: str | None = None
    if real_return:
        inflation_series = inflation_daily if inflation_daily is not None else inflation_daily_series(index, warnings=warnings)
        if inflation_series is None:
            total_return = None
            daily_final = None
        else:
            total_inflation = cumulative_return(inflation_series)
            total_return = deflate_return(net_total_nominal, total_inflation)
            daily_final = deflate_series(daily_effective, inflation_series)
            inflation_source = "BCB SGS 433 (IPCA mensal, distribuído por dia corrido)" if base_currency == "BRL" else "FRED CPIAUCSL (CPI EUA mensal, distribuído por dia corrido)"

    # Hurdle
    if hurdle_series is None:
        built, hurdle_source = build_hurdle_series(
            hurdle,
            index,
            real_return=real_return,
            cdi_rates=cdi_rates,
            inflation_daily=inflation_series,
            sofr_daily=sofr_daily,
            us_cpi_daily=us_cpi_daily,
            annualization=annualization,
            warnings=warnings,
        )
    else:
        built, hurdle_source = hurdle_series, str(hurdle_metadata(hurdle)["source"])
    hurdle_total_gross = cumulative_return(built) if built is not None else None
    hurdle_taxable = hurdle in {"CDI", "IPCA_PLUS_6"}
    hurdle_total = (
        _net_hurdle(hurdle_total_gross, tax_profile, holding_days)
        if hurdle_total_gross is not None and net_of_tax and hurdle_taxable
        else hurdle_total_gross
    )
    hurdle_daily: pd.Series | None = built
    if built is not None and net_of_tax and hurdle_taxable and hurdle_total_gross is not None and hurdle_total is not None:
        hurdle_factor = (1 + hurdle_total) / (1 + hurdle_total_gross) if (1 + hurdle_total_gross) > 0 else 1.0
        hurdle_daily = (1 + built) * hurdle_factor - 1

    cagr = _annualized(total_return, observations, annualization)
    hurdle_cagr = _annualized(hurdle_total, observations, annualization)
    excess_return = total_return - hurdle_total if total_return is not None and hurdle_total is not None else None
    excess_cagr = cagr - hurdle_cagr if cagr is not None and hurdle_cagr is not None else None
    volatility = _volatility(daily_final, annualization) if daily_final is not None else None
    excess_sharpe = excess_cagr / volatility if excess_cagr is not None and volatility not in (None, 0) else None

    metrics = {
        "total_return": _metric(total_return, "fraction"),
        "cagr": _metric(cagr, "fraction"),
        "volatility": _metric(volatility, "fraction"),
        "hurdle_total_return": _metric(hurdle_total, "fraction"),
        "hurdle_cagr": _metric(hurdle_cagr, "fraction"),
        "excess_return": _metric(excess_return, "fraction"),
        "excess_cagr": _metric(excess_cagr, "fraction"),
        "excess_sharpe": _metric(excess_sharpe, "ratio"),
        "effective_tax_rate": _metric(taxes.effective_tax_rate, "fraction"),
        "tax_drag_pp": _metric(taxes.total_drag * 100, "pp"),
    }

    if hurdle_total is None:
        warnings.append(f"Hurdle {default_hurdle_label(hurdle)} indisponível: as métricas de excesso ficam vazias.")

    assumptions = list(taxes.assumptions)
    if real_return:
        assumptions.append(
            "Retorno real pela equação de Fisher exata: (1 + real) = (1 + nominal) / (1 + inflação). A inflação mensal é distribuída por dia corrido."
        )
    if net_of_tax:
        assumptions.append(
            "O retorno líquido reduz a riqueza pelo IR estimado sobre o ganho acumulado; o gráfico distribui esse custo proporcionalmente ao longo do período."
        )
        if hurdle in {"CDI", "IPCA_PLUS_6"}:
            assumptions.append("O hurdle investível (CDI ou IPCA + 6%) recebe o mesmo IR regressivo de renda fixa do perfil escolhido.")
        else:
            assumptions.append("SOFR e CPI EUA são taxas de referência externas; nenhum IR brasileiro é aplicado a elas.")

    chart = _wealth_figure(
        daily_final if daily_final is not None else daily_effective,
        hurdle_daily,
        hurdle_label=default_hurdle_label(hurdle),
        real_return=bool(real_return and daily_final is not None),
        net_of_tax=net_of_tax,
    )

    regime = {
        "base_currency": base_currency,
        "hurdle": hurdle,
        "hurdle_label": default_hurdle_label(hurdle),
        "hurdle_kind": hurdle_metadata(hurdle)["kind"],
        "tax_profile": tax_profile,
        "tax_profile_label": TAX_PROFILE_METADATA[tax_profile]["label"],
        "real_return": real_return,
        "net_of_tax": net_of_tax,
        "holding_days": int(holding_days),
        "observations": observations,
        "annualization_days": annualization,
        "inflation_source": inflation_source,
    }

    return {
        "metrics": metrics,
        "asset_breakdown": taxes.breakdown,
        "assumptions": assumptions,
        "warnings": warnings,
        "chart": chart,
        "regime": regime,
        "hurdle": {
            "key": hurdle,
            "label": default_hurdle_label(hurdle),
            "source": hurdle_source,
            "total_return_gross": hurdle_total_gross,
            "total_return_net": hurdle_total,
            "tax_applied": bool(net_of_tax and hurdle in {"CDI", "IPCA_PLUS_6"}),
        },
        "sample": {
            "start": pd.Timestamp(index[0]).date().isoformat(),
            "end": pd.Timestamp(index[-1]).date().isoformat(),
            "observations": observations,
        },
        "tax": {
            "gross_return": taxes.gross_return,
            "tax_amount": taxes.tax_amount,
            "fx_drag": taxes.fx_drag,
            "total_drag": taxes.total_drag,
            "net_return": taxes.net_return,
        },
    }
