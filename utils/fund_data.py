"""Fund detail enrichment, fixed-income reference rates and Yahoo fund fields.

Sources, all read directly and never inferred:
- Fundamentus' FII detail page. The dedicated `fii_detalhes.php?papel=` endpoint
  currently answers HTTP 200 with a PHP fatal error, so the live
  `detalhes.php?papel=` page is used: it publishes the same FII block
  (vacância, cap rate, FFO yield, imóveis, segmento, gestão).
- Banco Central SGS series for the CDI annualized rate (4389) and the trailing
  12-month IPCA (13522), used only to describe modeled fixed-income terms.
- Yahoo Finance fields already present in the shared `info` payload.

Nothing here invents a value: a field the source omits stays None.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any

import pandas as pd
import requests

from utils.cache import cache_data

logger = logging.getLogger(__name__)

FUNDAMENTUS_DETAIL_URL = "https://www.fundamentus.com.br/detalhes.php"
FUNDAMENTUS_HEADERS = {"User-Agent": "Mozilla/5.0"}
FII_DETAIL_TTL = 6 * 3600

# Banco Central SGS: CDI anualizado (base 252) and IPCA acumulado em 12 meses.
BCB_CDI_SERIES = 4389
BCB_IPCA_SERIES = 13522
_BCB_TIMEOUT_SECONDS = 8
# Same defaults `build_synthetic_fixed_income_prices` applies when the BCB series is unavailable.
MODEL_CDI_FALLBACK = 13.75
MODEL_IPCA_FALLBACK = 4.62


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _number(value: Any) -> float | None:
    """Parse a Fundamentus cell: '10,5%' -> 10.5, '2.031.742' -> 2031742.0, '-' -> None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _finite(value)
    text = re.sub(r"<[^>]+>", "", str(value or "")).strip()
    text = text.replace("%", "").replace("\xa0", "").replace(" ", "")
    if text in {"", "-", "—", "N/A", "nan", "None"}:
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text):
        text = text.replace(".", "")
    return _finite(text)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"<[^>]+>", "", str(value)).strip()
    return None if text in {"", "-", "—", "N/A", "nan", "None"} else text


def _date(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    for pattern in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text[:10], pattern).date().isoformat()
        except ValueError:
            continue
    return None


def _epoch_date(value: Any) -> str | None:
    epoch = _finite(value)
    if epoch is None or epoch <= 0:
        return None
    try:
        return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return None


# ── FII / Fiagro / FI-Infra detail ─────────────────────────────────────────────


def _label_pairs(html: str) -> dict[str, str]:
    """Every Fundamentus label/value cell pair, keyed by the cleaned label text."""
    try:
        tables = pd.read_html(StringIO(html), decimal=",", thousands=".", flavor="lxml")
    except ValueError:
        return {}
    pairs: dict[str, str] = {}
    for table in tables:
        for row in table.itertuples(index=False, name=None):
            cells = ["" if pd.isna(cell) else str(cell) for cell in row]
            for index in range(0, len(cells) - 1, 2):
                label = cells[index].lstrip("?").strip()
                if label and label.lower() != "nan":
                    pairs[label] = cells[index + 1]
    return pairs


def parse_fii_detail_html(html: str) -> dict[str, Any] | None:
    """Map the Fundamentus FII detail block into B3Lab fields, or None when absent."""
    pairs = _label_pairs(html)
    if not _text(pairs.get("FII")) or not _text(pairs.get("Segmento")):
        return None
    detail = {
        "ticker": _text(pairs.get("FII")),
        "vacancy": _number(pairs.get("Vacância Média")),
        "cap_rate": _number(pairs.get("Cap Rate")),
        "ffo_yield": _number(pairs.get("FFO Yield")),
        "properties": _number(pairs.get("Qtd imóveis")),
        "properties_area_m2": _number(pairs.get("Área (m2)")),
        "units": _number(pairs.get("Qtd Unidades")),
        "segment": _text(pairs.get("Segmento")),
        "mandate": _text(pairs.get("Mandato")),
        "management": _text(pairs.get("Gestão")),
        # Fundamentus does not publish the manager today; kept ready for the label if it appears.
        "manager": _text(pairs.get("Gestora") or pairs.get("Gestor") or pairs.get("Administrador")),
        "nav_per_share": _number(pairs.get("VP/Cota")),
        "ffo_per_share": _number(pairs.get("FFO/Cota")),
        "dividend_per_share": _number(pairs.get("Dividendo/cota")),
        "market_value": _number(pairs.get("Valor de mercado")),
        "report_date": _date(pairs.get("Relatório")),
        "source": "Fundamentus · detalhes do FII",
    }
    if all(value is None for key, value in detail.items() if key != "source"):
        return None
    return detail


@cache_data(ttl=FII_DETAIL_TTL, stale=86400, cache_if=bool, max_entries=256)
def get_fii_detail(ticker: str) -> dict[str, Any] | None:
    """Fetch one FII detail page; returns None on any source or parse failure."""
    try:
        response = requests.get(
            FUNDAMENTUS_DETAIL_URL,
            params={"papel": ticker},
            headers=FUNDAMENTUS_HEADERS,
            timeout=8,
        )
        response.raise_for_status()
        detail = parse_fii_detail_html(response.content.decode("iso-8859-1"))
        if detail is not None and str(detail.get("ticker") or "").upper() != str(ticker).upper():
            return None
    except (requests.RequestException, ValueError, AttributeError, ImportError) as exc:
        logger.warning("Fundamentus FII detail unavailable for %s: %s", ticker, exc)
        return None
    return detail


# ── Fixed-income reference rates ───────────────────────────────────────────────


def _bcb_series(series: int, window_days: int) -> dict[str, Any] | None:
    """Latest published point of an SGS series, in percent, or None."""
    today = dt.date.today()
    start = today - dt.timedelta(days=window_days)
    try:
        response = requests.get(
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series}/dados",
            params={"formato": "json", "dataInicial": f"{start:%d/%m/%Y}", "dataFinal": f"{today:%d/%m/%Y}"},
            timeout=_BCB_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("formato inesperado")
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("BCB series %s unavailable: %s", series, exc)
        return None
    points = []
    for row in rows:
        try:
            date = dt.datetime.strptime(str(row["data"]), "%d/%m/%Y").date()
            value = _finite(str(row["valor"]).replace(",", "."))
        except (KeyError, TypeError, ValueError):
            continue
        if value is not None and date <= today:
            points.append((date, value))
    if not points:
        return None
    date, value = sorted(points)[-1]
    return {"percent": value, "date": date.isoformat(), "source": f"BCB SGS {series}"}


@cache_data(ttl=3600, stale=86400, max_entries=4)
def get_reference_rates() -> dict[str, dict[str, Any] | None]:
    """CDI annualized and trailing 12-month IPCA, each independently fail-soft."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        cdi = pool.submit(_bcb_series, BCB_CDI_SERIES, 30)
        ipca = pool.submit(_bcb_series, BCB_IPCA_SERIES, 200)
        return {"cdi": cdi.result(), "ipca": ipca.result()}


def synthetic_income_reference(spec: dict[str, Any]) -> dict[str, Any]:
    """Indexer and index rate behind a modeled fixed-income ticker, fallback included."""
    kind = str(spec.get("kind") or "")
    if kind in {"cdi_pct", "cdi_spread"}:
        reference = get_reference_rates().get("cdi")
        indexer, fallback_default = "CDI", MODEL_CDI_FALLBACK
    elif kind == "ipca_plus":
        reference = get_reference_rates().get("ipca")
        indexer, fallback_default = "IPCA", MODEL_IPCA_FALLBACK
    elif kind == "poupanca":
        return {
            "indexer": "Poupança (TR + 6,17% a.a.)", "index_rate": None, "index_rate_unit": None,
            "index_rate_date": None, "index_rate_source": "Premissa do modelo B3Lab", "index_rate_fallback": False,
        }
    elif kind == "usd_fixed":
        return {
            "indexer": "Prefixado (USD)", "index_rate": None, "index_rate_unit": None,
            "index_rate_date": None, "index_rate_source": "Premissa do modelo B3Lab", "index_rate_fallback": False,
        }
    else:
        return {
            "indexer": "Prefixado", "index_rate": None, "index_rate_unit": None,
            "index_rate_date": None, "index_rate_source": "Premissa do modelo B3Lab", "index_rate_fallback": False,
        }
    if reference is None:
        fallback_text = f"{fallback_default:.2f}".replace(".", ",")
        return {
            "indexer": indexer, "index_rate": fallback_default, "index_rate_unit": "% a.a.",
            "index_rate_date": None,
            "index_rate_source": f"Premissa do modelo B3Lab ({fallback_text}% a.a.)",
            "index_rate_fallback": True,
        }
    return {
        "indexer": indexer, "index_rate": reference["percent"], "index_rate_unit": "% a.a.",
        "index_rate_date": reference["date"], "index_rate_source": reference["source"],
        "index_rate_fallback": False,
    }


# ── Yahoo Finance fund profile ─────────────────────────────────────────────────


def yahoo_fund_block(
    info: dict[str, Any] | None,
    asset_class: str,
    *,
    distribution_yield: float | None = None,
) -> dict[str, Any] | None:
    """Yahoo fund/REIT fields, only those the payload actually carries."""
    if not isinstance(info, dict) or not info:
        return None
    quote_type = str(info.get("quoteType") or "").upper()
    fund_like = asset_class in {"etf", "reit", "fii"} or quote_type in {"ETF", "MUTUALFUND"}
    expense_ratio = _finite(info.get("netExpenseRatio"))
    if expense_ratio is None:
        expense_ratio = _finite(info.get("annualReportExpenseRatio"))
    total_assets = _finite(info.get("totalAssets")) or _finite(info.get("netAssets"))
    beta = _finite(info.get("beta3Year")) or _finite(info.get("beta"))
    leverage = None
    if asset_class == "reit":
        debt = _finite(info.get("totalDebt"))
        cash = _finite(info.get("totalCash"))
        ebitda = _finite(info.get("ebitda"))
        market_cap = _finite(info.get("marketCap"))
        net_debt_to_ebitda = None
        if debt is not None and ebitda is not None and ebitda > 0:
            net_debt_to_ebitda = (debt - (cash or 0.0)) / ebitda
        debt_to_market_cap = debt / market_cap if debt is not None and market_cap and market_cap > 0 else None
        if net_debt_to_ebitda is not None or debt_to_market_cap is not None:
            leverage = {"net_debt_to_ebitda": net_debt_to_ebitda, "debt_to_market_cap": debt_to_market_cap}
    block = {
        "fund_family": _text(info.get("fundFamily")),
        "category": _text(info.get("category")),
        "legal_type": _text(info.get("legalType")),
        "expense_ratio": expense_ratio,
        "total_assets": total_assets,
        "distribution_yield": distribution_yield,
        "beta": beta,
        "inception": _epoch_date(info.get("fundInceptionDate")),
        "nav_price": _finite(info.get("navPrice")),
        "leverage": leverage,
        # Yahoo does not publish BDR underlying or ratio; stay null instead of deriving them from price.
        "underlying": _text(info.get("underlyingSymbol") or info.get("underlyingExchangeSymbol")),
        "underlying_ratio": _finite(info.get("underlyingRatio")),
        "source": "Yahoo Finance",
    }
    if not fund_like and not any(
        block[key] is not None for key in ("fund_family", "category", "expense_ratio", "total_assets", "inception")
    ):
        return None
    if all(value is None for key, value in block.items() if key != "source"):
        return None
    return block
