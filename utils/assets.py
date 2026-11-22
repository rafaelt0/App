"""Supplemental portfolio instruments, keyed by the IDs stored in selections."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Asset:
    company: str
    asset_class: str
    market: str
    treasury_type: str = ""
    maturity: date | None = None


ASSETS = {
    "AAPL": Asset("Apple", "us_stock", "US"),
    "AMZN": Asset("Amazon", "us_stock", "US"),
    "BRK-B": Asset("Berkshire Hathaway Classe B", "us_stock", "US"),
    "GOOGL": Asset("Alphabet Classe A", "us_stock", "US"),
    "JNJ": Asset("Johnson & Johnson", "us_stock", "US"),
    "JPM": Asset("JPMorgan Chase", "us_stock", "US"),
    "META": Asset("Meta Platforms", "us_stock", "US"),
    "MSFT": Asset("Microsoft", "us_stock", "US"),
    "NVDA": Asset("Nvidia", "us_stock", "US"),
    "TSLA": Asset("Tesla", "us_stock", "US"),
    "V": Asset("Visa", "us_stock", "US"),
    "BTLG11": Asset("BTG Pactual Logística", "fii", "B3"),
    "CPTS11": Asset("Capitânia Securities II", "fii", "B3"),
    "HGLG11": Asset("Pátria Log", "fii", "B3"),
    "KNCR11": Asset("Kinea Rendimentos Imobiliários", "fii", "B3"),
    "KNRI11": Asset("Kinea Renda Imobiliária", "fii", "B3"),
    "MXRF11": Asset("Maxi Renda", "fii", "B3"),
    "VISC11": Asset("Vinci Shopping Centers", "fii", "B3"),
    "XPML11": Asset("XP Malls", "fii", "B3"),
    "BOVA11": Asset("iShares Ibovespa", "etf", "B3"),
    "GOLD11": Asset("Trend ETF LBMA Ouro", "etf", "B3"),
    "IVVB11": Asset("iShares S&P 500 Brasil", "etf", "B3"),
    "NASD11": Asset("Trend ETF Nasdaq 100", "etf", "B3"),
    "SMAL11": Asset("iShares Small Cap Brasil", "etf", "B3"),
    "AGG": Asset("iShares Core U.S. Aggregate Bond ETF", "etf", "US"),
    "QQQ": Asset("Invesco QQQ Trust", "etf", "US"),
    "SPY": Asset("SPDR S&P 500 ETF Trust", "etf", "US"),
    "VTI": Asset("Vanguard Total Stock Market ETF", "etf", "US"),
    "VOO": Asset("Vanguard S&P 500 ETF", "etf", "US"),
    "VNQ": Asset("Vanguard Real Estate ETF", "etf", "US"),
    "AMT": Asset("American Tower", "reit", "US"),
    "DLR": Asset("Digital Realty", "reit", "US"),
    "EQIX": Asset("Equinix", "reit", "US"),
    "O": Asset("Realty Income", "reit", "US"),
    "PLD": Asset("Prologis", "reit", "US"),
    "SPG": Asset("Simon Property Group", "reit", "US"),
    "TD-SELIC-2029": Asset("Tesouro Selic 2029", "treasury", "TD", "Tesouro Selic", date(2029, 3, 1)),
    "TD-SELIC-2031": Asset("Tesouro Selic 2031", "treasury", "TD", "Tesouro Selic", date(2031, 3, 1)),
    "TD-PREFIXADO-2029": Asset("Tesouro Prefixado 2029", "treasury", "TD", "Tesouro Prefixado", date(2029, 1, 1)),
    "TD-PREFIXADO-2032": Asset("Tesouro Prefixado 2032", "treasury", "TD", "Tesouro Prefixado", date(2032, 1, 1)),
    "TD-IPCA-2035": Asset("Tesouro IPCA+ 2035", "treasury", "TD", "Tesouro IPCA+", date(2035, 5, 15)),
    "TD-IPCA-2040": Asset("Tesouro IPCA+ 2040", "treasury", "TD", "Tesouro IPCA+", date(2040, 8, 15)),
}


def price_symbol(ticker: str) -> str:
    """Keep Yahoo close column names for stocks and stable IDs for Tesouro."""
    asset = ASSETS.get(ticker)
    return ticker if asset and asset.market != "B3" else f"{ticker}.SA"


def is_treasury(ticker: str) -> bool:
    asset = ASSETS.get(ticker)
    return asset is not None and asset.market == "TD"


def is_us_asset(ticker: str) -> bool:
    asset = ASSETS.get(ticker)
    return asset is not None and asset.market == "US"
