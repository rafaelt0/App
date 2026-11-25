"""Return regime configuration shared by the after-tax hurdle analysis (WP-1.1).

The regime is a small, immutable value object: base currency, hurdle benchmark,
tax profile and the two presentation switches (real return and net of tax).
It carries metadata only; every financial calculation lives in
``utils/tax_hurdle.py`` so the API and the React layer never re-derive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BaseCurrency = Literal["BRL", "USD", "EUR"]
Hurdle = Literal["CDI", "IPCA_PLUS_6", "SOFR", "US_CPI"]
TaxProfile = Literal["BR_PF", "US_TAXABLE", "NONE"]

ANNUALIZATION_DAYS = 252

#: Labels, units and real/nominal nature of each supported hurdle.
HURDLE_METADATA: dict[str, dict[str, str | bool]] = {
    "CDI": {
        "label": "CDI",
        "long_label": "CDI (pós-fixado, Brasil)",
        "unit": "fraction",
        "currency": "BRL",
        "kind": "nominal",
        "source": "BCB SGS 12 (CDI diário)",
    },
    "IPCA_PLUS_6": {
        "label": "IPCA + 6%",
        "long_label": "IPCA + 6% a.a. (rende 6% acima da inflação brasileira)",
        "unit": "fraction",
        "currency": "BRL",
        "kind": "real",
        "source": "BCB SGS 433 (IPCA mensal) + 6% a.a.",
    },
    "SOFR": {
        "label": "SOFR",
        "long_label": "SOFR (taxa de referência em dólar)",
        "unit": "fraction",
        "currency": "USD",
        "kind": "nominal",
        "source": "FRED SOFR (diário)",
    },
    "US_CPI": {
        "label": "CPI EUA",
        "long_label": "CPI dos EUA (inflação americana)",
        "unit": "fraction",
        "currency": "USD",
        "kind": "inflation",
        "source": "FRED CPIAUCSL (CPI EUA mensal)",
    },
}

CURRENCY_METADATA: dict[str, dict[str, str]] = {
    "BRL": {"symbol": "R$", "label": "Real brasileiro", "locale": "pt-BR", "inflation_index": "IPCA"},
    "USD": {"symbol": "US$", "label": "Dólar americano", "locale": "en-US", "inflation_index": "CPI EUA"},
    "EUR": {"symbol": "€", "label": "Euro", "locale": "de-DE", "inflation_index": "HICP"},
}

TAX_PROFILE_METADATA: dict[str, dict[str, str]] = {
    "BR_PF": {"label": "Brasil PF", "long_label": "Pessoa física residente no Brasil"},
    "US_TAXABLE": {"label": "EUA Tributável", "long_label": "Conta tributável nos Estados Unidos"},
    "NONE": {"label": "Isento", "long_label": "Sem tributação (conta isenta)"},
}


def hurdle_metadata(hurdle: str) -> dict[str, str | bool]:
    """Metadata for a supported hurdle; unknown keys fail loudly instead of guessing."""
    try:
        return dict(HURDLE_METADATA[hurdle])
    except KeyError as exc:
        raise ValueError(f"Hurdle não suportado: {hurdle!r}") from exc


def default_hurdle_label(hurdle: str) -> str:
    """Short pt-BR label used in captions and legends."""
    return str(hurdle_metadata(hurdle)["label"])


def currency_metadata(currency: str) -> dict[str, str]:
    """Symbol and inflation index for a supported base currency."""
    try:
        return dict(CURRENCY_METADATA[currency])
    except KeyError as exc:
        raise ValueError(f"Moeda não suportada: {currency!r}") from exc


@dataclass(frozen=True)
class RegimeConfig:
    """Chosen return regime for one hurdle comparison."""

    base_currency: BaseCurrency = "BRL"
    hurdle: Hurdle = "CDI"
    tax_profile: TaxProfile = "BR_PF"
    real_return: bool = False
    net_of_tax: bool = False

    def __post_init__(self) -> None:
        if self.base_currency not in CURRENCY_METADATA:
            raise ValueError(f"Moeda não suportada: {self.base_currency!r}")
        if self.hurdle not in HURDLE_METADATA:
            raise ValueError(f"Hurdle não suportado: {self.hurdle!r}")
        if self.tax_profile not in TAX_PROFILE_METADATA:
            raise ValueError(f"Regime tributário não suportado: {self.tax_profile!r}")

    @property
    def hurdle_label(self) -> str:
        return default_hurdle_label(self.hurdle)

    @property
    def hurdle_source(self) -> str:
        return str(hurdle_metadata(self.hurdle)["source"])

    @property
    def currency(self) -> dict[str, str]:
        return currency_metadata(self.base_currency)

    @property
    def tax_profile_label(self) -> str:
        return TAX_PROFILE_METADATA[self.tax_profile]["label"]

    def describe(self) -> dict[str, object]:
        """Serializable summary for API responses and UI captions."""
        return {
            "base_currency": self.base_currency,
            "currency": self.currency,
            "hurdle": self.hurdle,
            "hurdle_label": self.hurdle_label,
            "hurdle_source": self.hurdle_source,
            "hurdle_kind": hurdle_metadata(self.hurdle)["kind"],
            "tax_profile": self.tax_profile,
            "tax_profile_label": self.tax_profile_label,
            "real_return": self.real_return,
            "net_of_tax": self.net_of_tax,
            "annualization_days": ANNUALIZATION_DAYS,
        }
