import datetime as dt
from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest

from backend import home
from utils import assets, fund_data, treasury_data

FII_HTML = """
<html><body><table>
<tr><td>?FII</td><td>HGLG11</td><td>?Cotação</td><td>140,69</td></tr>
<tr><td>?Segmento</td><td>Multicategoria</td><td>?Gestão</td><td>Ativa</td></tr>
<tr><td>?Mandato</td><td>-</td><td>?Relatório</td><td>30/09/2026</td></tr>
<tr><td>?FFO Yield</td><td>8,14%</td><td>?FFO/Cota</td><td>11,45</td></tr>
<tr><td>?Div. Yield</td><td>9,5%</td><td>?VP/Cota</td><td>165,02</td></tr>
<tr><td>?Valor de mercado</td><td>6.527.800.000</td><td>?Qtd imóveis</td><td>36</td></tr>
<tr><td>?Área (m2)</td><td>2.031.742</td><td>?Cap Rate</td><td>10,5%</td></tr>
<tr><td>?Vacância Média</td><td>2,5%</td><td>?Qtd Unidades</td><td>36</td></tr>
</table></body></html>
"""


@pytest.fixture(autouse=True)
def reference_rates(monkeypatch):
    """Deterministic BCB rates; no test reaches the network."""
    monkeypatch.setattr(fund_data, "get_reference_rates", lambda: {
        "cdi": {"percent": 13.65, "date": "2026-11-19", "source": "BCB SGS 4389"},
        "ipca": {"percent": 4.63, "date": "2026-10-01", "source": "BCB SGS 13522"},
    })


def test_fii_detail_parser_reads_real_detail_fields():
    detail = fund_data.parse_fii_detail_html(FII_HTML)
    assert detail["ticker"] == "HGLG11"
    assert detail["vacancy"] == pytest.approx(2.5)
    assert detail["cap_rate"] == pytest.approx(10.5)
    assert detail["ffo_yield"] == pytest.approx(8.14)
    assert detail["properties"] == pytest.approx(36)
    assert detail["properties_area_m2"] == pytest.approx(2_031_742)
    assert detail["segment"] == "Multicategoria"
    assert detail["management"] == "Ativa"
    assert detail["mandate"] is None  # Fundamentus printed "-"
    assert detail["manager"] is None  # not published by the source
    assert detail["nav_per_share"] == pytest.approx(165.02)
    assert detail["ffo_per_share"] == pytest.approx(11.45)
    assert detail["market_value"] == pytest.approx(6_527_800_000)
    assert detail["report_date"] == "2026-09-30"
    assert detail["source"] == "Fundamentus · detalhes do FII"
    assert fund_data.parse_fii_detail_html("<html><body>sem tabela</body></html>") is None


def test_fii_detail_is_cached_per_ticker(monkeypatch):
    calls = []

    def fake_get(url, *, params, headers, timeout):
        calls.append(params["papel"])
        return SimpleNamespace(raise_for_status=lambda: None, content=FII_HTML.encode("iso-8859-1"))

    monkeypatch.setattr(fund_data.requests, "get", fake_get)
    fund_data.get_fii_detail.clear()
    try:
        assert fund_data.get_fii_detail("HGLG11")["vacancy"] == pytest.approx(2.5)
        assert fund_data.get_fii_detail("HGLG11")["cap_rate"] == pytest.approx(10.5)
        assert calls == ["HGLG11"]
    finally:
        fund_data.get_fii_detail.clear()


def test_fii_detail_keeps_summary_when_the_detail_page_fails(monkeypatch):
    table = pd.DataFrame({"price": [140.5], "pvp": [.89], "dy": [9.9], "liq2m": [715_027]}, index=["HGLG11"])
    monkeypatch.setattr(fund_data, "get_fii_detail", lambda ticker: None)
    data = assets.get_extra_asset_fundamentals("HGLG11", fii_table=table)
    assert data["source"] == "Fundamentus · FIIs"
    assert data["indicators"] == {"pvp": .89, "dy": 9.9}
    assert "fund" not in data
    assert data["price_unit"] == "money"


def test_fii_detail_enriches_vacancy_cap_rate_ffo_and_segment(monkeypatch):
    table = pd.DataFrame({"price": [140.5], "pvp": [.89], "dy": [9.9], "liq2m": [715_027]}, index=["HGLG11"])
    monkeypatch.setattr(fund_data, "get_fii_detail", lambda ticker: fund_data.parse_fii_detail_html(FII_HTML))
    data = assets.get_extra_asset_fundamentals("HGLG11", fii_table=table)
    assert data["fund"]["vacancy"] == pytest.approx(2.5)
    assert data["fund"]["cap_rate"] == pytest.approx(10.5)
    assert data["fund"]["ffo_yield"] == pytest.approx(8.14)
    assert data["fund"]["properties"] == pytest.approx(36)
    assert data["fund"]["segment"] == "Multicategoria"
    assert data["detail_source"] == "Fundamentus · detalhes do FII"
    assert data["indicators"] == {"pvp": .89, "dy": 9.9}


def _treasury_csv(today):
    return (
        "Tipo Titulo;Data Vencimento;Data Base;Taxa Compra Manha;Taxa Venda Manha;"
        "PU Compra Manha;PU Venda Manha;PU Base Manha\n"
        f"Tesouro Selic;01/03/2029;{today:%d/%m/%Y};0,05;0,07;15850,10;15842,30;15845,00\n"
        f"Tesouro IPCA+;15/05/2035;{today:%d/%m/%Y};6,40;6,55;2515,20;2510,00;2512,50\n"
    )


def test_treasury_detail_exposes_buy_sell_rates_maturity_and_quote_date(monkeypatch):
    today = dt.date.today()
    monkeypatch.setattr(treasury_data, "_download_treasury_csv", lambda: StringIO(_treasury_csv(today)))
    treasury_data.get_treasury_history.clear()
    try:
        detail = treasury_data.latest_treasury_detail("TD-SELIC-2029")
        assert detail["buy_price"] == pytest.approx(15850.10)
        assert detail["sell_price"] == pytest.approx(15842.30)
        assert detail["buy_rate"] == pytest.approx(0.05)
        assert detail["sell_rate"] == pytest.approx(0.07)
        assert detail["rate_unit"] == "% a.a."
        assert detail["maturity"] == "2029-03-01"
        assert detail["quote_date"] == today.isoformat()
        assert detail["source"].startswith("Tesouro Transparente")

        asset = assets.get_extra_asset_fundamentals("TD-SELIC-2029")
        assert asset["price"] == pytest.approx(15850.10)
        assert asset["source"] == "Tesouro Transparente · PU Compra Manhã"
        assert asset["treasury"]["sell_price"] == pytest.approx(15842.30)
        assert asset["treasury"]["maturity"] == "2029-03-01"
        assert asset["price_unit"] == "money"
    finally:
        treasury_data.get_treasury_history.clear()


def test_treasury_moves_keep_a_price_without_an_adjacent_prior(monkeypatch):
    index = pd.to_datetime(["2026-11-13", "2026-11-18", "2026-11-19"])
    frame = pd.DataFrame({
        "TD-IPCA-2035": [2500.0, 2542.31, 2526.53],
        "TD-SELIC-2029": [20300.0, None, 20320.65],
        "TD-PREFIXADO-2029": [746.01, None, None],
    }, index=index)
    monkeypatch.setattr(treasury_data, "treasury_prices", lambda tickers, start_date, adjusted: frame)
    moves = treasury_data.treasury_moves(("TD-IPCA-2035", "TD-SELIC-2029", "TD-PREFIXADO-2029"))
    assert moves["TD-IPCA-2035"]["price"] == pytest.approx(2526.53)
    assert moves["TD-IPCA-2035"]["change"] == pytest.approx(2526.53 / 2542.31 - 1)
    assert moves["TD-IPCA-2035"]["as_of"] == "2026-11-19"
    assert "Tesouro IPCA+ 15/05/2035" in moves["TD-IPCA-2035"]["source"]
    assert moves["TD-SELIC-2029"]["price"] == pytest.approx(20320.65)
    assert moves["TD-SELIC-2029"]["change"] is None
    assert moves["TD-PREFIXADO-2029"]["price"] == pytest.approx(746.01)
    assert moves["TD-PREFIXADO-2029"]["change"] is None
    assert treasury_data.resolve_treasury_ticker("TESOURO-IPCA-2035") == "TD-IPCA-2035"
    assert treasury_data.resolve_treasury_ticker("TESOURO-SELIC") == "TD-SELIC-2029"
    assert treasury_data.resolve_treasury_ticker("TESOURO-PRE") == "TD-PREFIXADO-2029"
    assert treasury_data.resolve_treasury_ticker("TESOURO-SELIC-2027") is None


def test_synthetic_fixed_income_surfaces_indexer_and_observed_rate(monkeypatch):
    cdi = assets.get_extra_asset_fundamentals("CDB-110CDI")
    assert cdi["fixed_income"]["indexer"] == "CDI"
    assert cdi["fixed_income"]["index_rate"] == pytest.approx(13.65)
    assert cdi["fixed_income"]["index_rate_unit"] == "% a.a."
    assert cdi["fixed_income"]["index_rate_source"] == "BCB SGS 4389"
    assert cdi["fixed_income"]["index_rate_fallback"] is False

    ipca = assets.get_extra_asset_fundamentals("IPCA+6")
    assert ipca["fixed_income"]["indexer"] == "IPCA"
    assert ipca["fixed_income"]["index_rate"] == pytest.approx(4.63)


def test_synthetic_fixed_income_marks_the_model_fallback(monkeypatch):
    monkeypatch.setattr(fund_data, "get_reference_rates", lambda: {"cdi": None, "ipca": None})
    cdi = assets.get_extra_asset_fundamentals("CDI")
    assert cdi["fixed_income"]["index_rate"] == pytest.approx(13.75)
    assert cdi["fixed_income"]["index_rate_fallback"] is True
    assert "Premissa do modelo" in cdi["fixed_income"]["index_rate_source"]

    ipca = assets.get_extra_asset_fundamentals("IPCA+7")
    assert ipca["fixed_income"]["index_rate"] == pytest.approx(4.62)
    assert ipca["fixed_income"]["index_rate_fallback"] is True


@pytest.mark.parametrize(("ticker", "symbol", "currency", "info", "expected"), [
    ("^BVSP", "^BVSP", "BRL", {"regularMarketVolume": 0}, {"price_unit": "points", "quote_pair": None}),
    ("BTC-USD", "BTC-USD", "USD", {"volume24Hr": 45_907_456_000, "regularMarketVolume": 45_907_456_000},
     {"price_unit": "money", "volume_24h": 45_907_456_000, "volume_window": "24h"}),
    ("USDBRL=X", "BRL=X", "BRL", {}, {"price_unit": "money", "quote_pair": "USD/BRL"}),
    ("GC=F", "GC=F", "USD", {"expireIsoDate": "2026-12-29T00:00:00Z", "underlyingSymbol": "GCZ26.CMX",
                             "openInterest": 107_337, "regularMarketVolume": 26_472},
     {"price_unit": "money", "session_volume": 26_472, "volume_window": "sessão"}),
])
def test_global_quote_units_pairs_volume_and_contract(monkeypatch, ticker, symbol, currency, info, expected):
    monkeypatch.setattr(assets, "get_yahoo_quotes", lambda symbols: {
        item: {"price": 100.0, "currency": currency, "source": "Yahoo Finance"} for item in symbols
    })
    monkeypatch.setattr(assets, "get_yahoo_info", lambda item: info if item == symbol else {})
    data = assets.get_extra_asset_fundamentals(ticker)
    for key, value in expected.items():
        assert data[key] == value, key
    if ticker == "GC=F":
        assert data["contract"]["expiry"] == "2026-12-29"
        assert data["contract"]["symbol"] == "GCZ26.CMX"
        assert data["contract"]["open_interest"] == pytest.approx(107_337)
        assert data["contract"]["unit"] is None  # not published by Yahoo


def test_yahoo_fund_fields_for_etf_reit_and_bdr(monkeypatch):
    etf = {
        "quoteType": "ETF", "currency": "USD", "regularMarketPrice": 690.0, "yield": 0.01,
        "fundFamily": "State Street Investment Management", "category": "Large Blend",
        "legalType": "Exchange Traded Fund", "netExpenseRatio": 0.0945,
        "totalAssets": 796_865_200_128, "beta3Year": 0.97,
        "fundInceptionDate": 727_660_800, "navPrice": 689.507,
    }
    monkeypatch.setattr(assets, "get_yahoo_info", lambda symbol: etf if symbol == "SPY" else {})
    data = assets.get_extra_asset_fundamentals("SPY")
    fund = data["fund"]
    assert fund["fund_family"] == "State Street Investment Management"
    assert fund["category"] == "Large Blend"
    assert fund["expense_ratio"] == pytest.approx(0.0945)
    assert fund["total_assets"] == pytest.approx(796_865_200_128)
    assert fund["distribution_yield"] == pytest.approx(1.0)
    assert fund["beta"] == pytest.approx(0.97)
    assert fund["inception"] == "1993-01-22"
    assert fund["underlying"] is None  # Yahoo does not publish a BDR underlying
    assert fund["source"] == "Yahoo Finance"

    reit = {
        "quoteType": "EQUITY", "currency": "USD", "regularMarketPrice": 57.5,
        "trailingAnnualDividendYield": 0.0558, "marketCap": 55_876_042_752,
        "totalDebt": 32_109_957_120, "totalCash": 2_000_000_000, "ebitda": 5_497_138_176, "beta": 0.649,
    }
    monkeypatch.setattr(assets, "get_yahoo_info", lambda symbol: reit if symbol == "O" else {})
    o = assets.get_extra_asset_fundamentals("O")
    assert o["fund"]["leverage"]["net_debt_to_ebitda"] == pytest.approx((32_109_957_120 - 2_000_000_000) / 5_497_138_176)
    assert o["fund"]["leverage"]["debt_to_market_cap"] == pytest.approx(32_109_957_120 / 55_876_042_752)

    bdr = {"quoteType": "EQUITY", "currency": "BRL", "regularMarketPrice": 65.0}
    monkeypatch.setattr(assets, "get_yahoo_info", lambda symbol: bdr if symbol == "AAPL34.SA" else {})
    aapl34 = assets.get_extra_asset_fundamentals("AAPL34")
    assert "fund" not in aapl34  # no fund field at all, nothing is invented


def test_foreign_equity_is_global_and_us_listing_is_american():
    toyota = assets._classify_yahoo_quote({
        "symbol": "7203.T", "quoteType": "EQUITY", "currency": "JPY", "exchange": "JPX",
        "longname": "Toyota Motor Corporation", "sector": "Consumer Cyclical",
    })
    assert toyota["asset_class"] == "outros"
    assert toyota["currency"] == "JPY"
    assert "(EUA)" not in toyota["sector"]

    apple = assets._classify_yahoo_quote({
        "symbol": "AAPL", "quoteType": "EQUITY", "currency": "USD", "exchange": "NMS",
        "longname": "Apple Inc.", "sector": "Technology",
    })
    assert apple["asset_class"] == "acao_us"
    assert assets.asset_metadata("7203.T")["asset_class"] == "outros"


def test_frame_forwards_fund_treasury_and_global_fields(monkeypatch):
    monkeypatch.setattr(fund_data, "get_fii_detail", lambda ticker: fund_data.parse_fii_detail_html(FII_HTML))
    monkeypatch.setattr(assets, "get_yahoo_quotes", lambda symbols: {
        symbol: {"price": 100.0, "currency": "USD", "source": "Yahoo Finance"} for symbol in symbols
    })
    monkeypatch.setattr(assets, "get_yahoo_info", lambda symbol: {
        "expireIsoDate": "2026-12-29T00:00:00Z", "underlyingSymbol": "GCZ26.CMX"
    } if symbol == "GC=F" else {})
    monkeypatch.setattr(home, "get_yahoo_quotes", lambda symbols: {
        symbol: {"price": 100.0, "currency": "USD", "source": "Yahoo Finance"} for symbol in symbols
    })
    monkeypatch.setattr(home, "get_yahoo_info", lambda symbol: {
        "expireIsoDate": "2026-12-29T00:00:00Z", "underlyingSymbol": "GCZ26.CMX"
    } if symbol == "GC=F" else {})
    monkeypatch.setattr(home, "load_fii_market_data", lambda: pd.DataFrame(
        {"price": [140.5], "pvp": [.89], "dy": [9.9], "liq2m": [715_027]}, index=["HGLG11"]))

    frame = home._extra_fundamentals_frame(["HGLG11", "GC=F"], {})
    assert frame.loc["HGLG11", "Unidade_cotacao"] == "money"
    assert frame.loc["HGLG11", "Fundo"]["vacancy"] == pytest.approx(2.5)
    assert frame.loc["GC=F", "Contrato"]["expiry"] == "2026-12-29"

    universe = pd.DataFrame({"Ticker": ["HGLG11", "GC=F"], "Setor": ["FII", "Global"]})
    result = home.build_analysis(frame, pd.DataFrame(), ["HGLG11", "GC=F"], universe)
    by_ticker = {company["ticker"]: company for company in result["companies"]}
    assert by_ticker["HGLG11"]["fund"]["cap_rate"] == pytest.approx(10.5)
    assert by_ticker["HGLG11"]["treasury"] is None
    assert by_ticker["GC=F"]["market"]["price_unit"] == "money"
    assert by_ticker["GC=F"]["market"]["contract"]["expiry"] == "2026-12-29"
    assert by_ticker["GC=F"]["fund"] is None
