"""Regression tests for sector-relative ranking: midrank percentiles, zero yield,
negative equity, share-class deduplication and trimmed sector means."""

from itertools import product

import pandas as pd
import pytest

from utils import home_data
from utils.home_data import build_analyst_synthesis, compute_sector_ranking


def _b3(rows):
    return pd.DataFrame(rows, columns=["Ticker", "Setor"])


def _bancos(*tickers):
    return _b3([(ticker, "Bancos") for ticker in tickers])


def _row(ranking, name):
    return ranking[ranking["Múltiplo"] == name].iloc[0]


def _names(ranking):
    return set() if ranking.empty else set(ranking["Múltiplo"])


def _distinct_tickers(count):
    """Tickers whose roots (letters before the class digit) are all distinct."""
    letters = [a + b for a, b in product("ABCDEFGHIJKLMNOPQRSTUVWXYZ", repeat=2)]
    return [f"{prefix}Q3" for prefix in letters[:count]]


# 1. Percentile ceiling: the ticker is excluded from its own denominator and
#    the best peer-relative value reaches 100.


def test_best_in_sector_reaches_100_with_minimum_observations():
    peers_raw = pd.DataFrame({"PL": ["5", "10", "20"]}, index=["AAA1", "BBB1", "CCC1"])

    ranking = compute_sector_ranking(peers_raw, "AAA1", "Bancos", _bancos("AAA1", "BBB1", "CCC1"))

    row = _row(ranking, "P/L")
    assert row["Peers (n)"] == 3
    assert row["Percentil"] == 100.0
    assert row["Veredicto"] == "Favorável"


def test_ties_count_as_half_against_peers_only():
    # Ticker 10 vs peers 10 (tie) and 20 (beaten): (1 + 0.5 * 1) / 2 peers.
    peers_raw = pd.DataFrame({"PL": ["10", "10", "20"]}, index=["AAA1", "BBB1", "CCC1"])

    ranking = compute_sector_ranking(peers_raw, "AAA1", "Bancos", _bancos("AAA1", "BBB1", "CCC1"))

    row = _row(ranking, "P/L")
    assert row["Peers (n)"] == 3
    assert row["Percentil"] == 75.0


def test_higher_is_better_best_in_sector_reaches_100_in_synthesis():
    peers_raw = pd.DataFrame(
        {"PL": ["5", "10", "20"], "ROE": ["30", "10", "5"], "Div_Yield": ["9", "3", "1"]},
        index=["AAA1", "BBB1", "CCC1"],
    )

    synthesis = build_analyst_synthesis(peers_raw, "AAA1", "Bancos", _bancos("AAA1", "BBB1", "CCC1"))

    assert synthesis["categorias"]["Valuation"]["percentil"] == 100.0
    assert synthesis["categorias"]["Rentabilidade"]["percentil"] == 100.0
    assert synthesis["categorias"]["Valuation"]["veredicto"] == "Favorável"
    assert synthesis["indicadores_validos"] == 3


# 2. Dividend yield of zero is a real observation; other multiples keep 0 -> NA.


def test_zero_dividend_yield_stays_in_distribution():
    peers_raw = pd.DataFrame(
        {"Div_Yield": ["5", "0", "2", "3"]}, index=["AAA1", "BBB1", "CCC1", "DDD1"]
    )

    ranking = compute_sector_ranking(
        peers_raw, "AAA1", "Bancos", _bancos("AAA1", "BBB1", "CCC1", "DDD1")
    )

    row = _row(ranking, "Div. Yield (%)")
    # The non-payer BBB1 is a peer: 3 of 3 peers are beaten by the 5% yield.
    assert row["Peers (n)"] == 4
    assert row["Percentil"] == 100.0


def test_zero_dividend_yield_ticker_is_ranked_not_dropped():
    peers_raw = pd.DataFrame(
        {"Div_Yield": ["5,00%", "0,00%", "2,00%", "3,00%"]},
        index=["AAA1", "BBB1", "CCC1", "DDD1"],
    )

    ranking = compute_sector_ranking(
        peers_raw, "BBB1", "Bancos", _bancos("AAA1", "BBB1", "CCC1", "DDD1")
    )

    row = _row(ranking, "Div. Yield (%)")
    assert row["Valor"] == 0.0
    assert row["Peers (n)"] == 4
    assert row["Percentil"] == 0.0


def test_zero_still_means_not_applicable_for_other_multiples():
    peers_raw = pd.DataFrame(
        {"PL": ["0", "10", "15", "20"], "Div_Yield": ["0", "1", "2", "3"]},
        index=["AAA1", "BBB1", "CCC1", "DDD1"],
    )

    ranking = compute_sector_ranking(
        peers_raw, "BBB1", "Bancos", _bancos("AAA1", "BBB1", "CCC1", "DDD1")
    )

    assert _row(ranking, "P/L")["Peers (n)"] == 3
    assert _row(ranking, "Div. Yield (%)")["Peers (n)"] == 4


# 3. Negative equity: P/VP and ROE are not meaningful for the ticker or peers.


def _telecom():
    return _b3(
        [
            ("OIBR3", "Telecom"),
            ("TELB4", "Telecom"),
            ("VIVT3", "Telecom"),
            ("TIMS3", "Telecom"),
            ("BRIT3", "Telecom"),
        ]
    )


TELECOM_TICKERS = ["OIBR3", "TELB4", "VIVT3", "TIMS3", "BRIT3"]


def test_negative_equity_ticker_has_no_roe_or_pvp():
    peers_raw = pd.DataFrame(
        {
            "ROE": ["69", "20", "15", "40", "5"],
            "PVP": ["-1,5", "1,2", "2", "-0,5", "1"],
            "Patrim_Liq": ["-52700000000", "5000", "9000", "-100", "3000"],
        },
        index=TELECOM_TICKERS,
    )

    ranking = compute_sector_ranking(peers_raw, "OIBR3", "Telecom", _telecom())

    assert "ROE (%)" not in _names(ranking)
    assert "P/VP" not in _names(ranking)


def test_negative_equity_peers_are_excluded_from_roe_and_pvp_distributions():
    peers_raw = pd.DataFrame(
        {
            "ROE": ["69", "20", "15", "40", "5"],
            "PVP": ["-1,5", "1,2", "2", "-0,5", "1"],
            "Patrim_Liq": ["-52700000000", "5000", "9000", "-100", "3000"],
        },
        index=TELECOM_TICKERS,
    )

    ranking = compute_sector_ranking(peers_raw, "TELB4", "Telecom", _telecom())

    # OIBR3 (ROE 69) and TIMS3 (ROE 40) have negative equity and drop out.
    # Only VIVT3 (15) and BRIT3 (5) remain; TELB4's 20 beats both.
    roe = _row(ranking, "ROE (%)")
    assert roe["Peers (n)"] == 3
    assert roe["Percentil"] == 100.0

    pvp = _row(ranking, "P/VP")
    # Peers with positive equity: VIVT3 (2.0, beaten by 1.2) and BRIT3 (1.0).
    assert pvp["Peers (n)"] == 3
    assert pvp["Percentil"] == 50.0


def test_negative_patrimonio_liquido_nulls_roe_even_without_pvp():
    peers_raw = pd.DataFrame(
        {"ROE": ["69", "15"], "PVP": [None, "2"], "Patrim_Liq": ["-100", "9000"]},
        index=["OIBR3", "VIVT3"],
    )
    b3 = _b3([("OIBR3", "Telecom"), ("VIVT3", "Telecom")])

    ranking = compute_sector_ranking(peers_raw, "OIBR3", "Telecom", b3)

    # OIBR3's only usable value was ROE, which is now null, so no row is left.
    assert ranking.empty


# 4. Share classes count once per issuer: the most liquid class represents it,
#    and the ticker's own issuer is not a peer.


def test_share_classes_collapse_to_most_liquid_representative():
    peers_raw = pd.DataFrame(
        {
            "PL": ["10", "1", "4", "6", "20"],
            "Liq_2meses": ["1000", "9000", "100", "500", "300"],
        },
        index=["AAA1", "AAA4", "BBB3", "BBB4", "CCC1"],
    )

    ranking = compute_sector_ranking(
        peers_raw, "AAA1", "Bancos", _bancos("AAA1", "AAA4", "BBB3", "BBB4", "CCC1")
    )

    row = _row(ranking, "P/L")
    # AAA4 is the ticker's own issuer and BBB3 loses to BBB4 on liquidity.
    # Observations: AAA1 (ticker), BBB4 (6), CCC1 (20).
    assert row["Peers (n)"] == 3
    assert row["Percentil"] == 50.0
    assert row["Mediana Setor"] == 10.0
    assert row["Média Setor"] == 12.0


def test_share_class_without_liquidity_column_keeps_first_listed_class():
    peers_raw = pd.DataFrame(
        {"PL": ["10", "4", "6", "20"]}, index=["AAA1", "BBB3", "BBB4", "CCC1"]
    )

    ranking = compute_sector_ranking(
        peers_raw, "AAA1", "Bancos", _bancos("AAA1", "BBB3", "BBB4", "CCC1")
    )

    row = _row(ranking, "P/L")
    assert row["Peers (n)"] == 3
    assert row["Média Setor"] == 11.33  # mean of 10, 4 (BBB3), 20


def test_ticker_own_issuer_is_not_a_peer_even_when_more_liquid():
    peers_raw = pd.DataFrame(
        {"PL": ["10", "1", "30"], "Liq_2meses": ["100", "9000", "100"]},
        index=["AAA3", "AAA4", "CCC1"],
    )

    ranking = compute_sector_ranking(peers_raw, "AAA3", "Bancos", _bancos("AAA3", "AAA4", "CCC1"))

    row = _row(ranking, "P/L")
    # Only CCC1 is a peer. AAA4 (P/L 1) would have made the ticker look expensive.
    assert row["Peers (n)"] == 2
    assert row["Percentil"] == 100.0


def test_get_sector_peers_carries_liquidity_and_equity_columns(monkeypatch):
    raw = pd.DataFrame(
        {
            "P/L": ["10"],
            "Liq.2meses": ["1.000.000"],
            "Patrim. Líq": ["-5.000"],
        },
        index=["AAA3"],
    )
    monkeypatch.setattr(home_data, "get_full_market_data", lambda: raw)

    # Bypass the process cache to exercise the renaming directly.
    peers = home_data.get_sector_peers.__wrapped__()

    assert {"PL", "Liq_2meses", "Patrim_Liq"}.issubset(peers.columns)
    assert peers.loc["AAA3", "Liq_2meses"] == "1.000.000"


# 5. Trimmed sector mean: outliers do not dominate, output names are unchanged.


def test_sector_mean_is_trimmed_and_median_unchanged():
    tickers = _distinct_tickers(12)
    ticker, peers = tickers[0], tickers[1:]
    # Ticker 5 among 12 observations; outliers -100 and 1000 are trimmed (10% each side).
    values = [-100, 1, 2, 2, 3, 4, 7, 8, 9, 1000, 6]
    peers_raw = pd.DataFrame(
        {"Marg_EBIT": [str(v) for v in [5] + values]},
        index=[ticker] + peers,
    )
    b3 = _bancos(*([ticker] + peers))

    ranking = compute_sector_ranking(peers_raw, ticker, "Bancos", b3)

    row = _row(ranking, "Margem EBIT (%)")
    assert row["Peers (n)"] == 12
    # Sorted observations: -100 1 2 2 3 4 5 6 7 8 9 1000; trimmed 10 middle values.
    assert row["Média Setor"] == 4.7
    assert row["Mediana Setor"] == 4.5
    assert list(ranking.columns) == [
        "Categoria",
        "Múltiplo",
        "Valor",
        "Mediana Setor",
        "Média Setor",
        "Peers (n)",
        "Percentil",
        "Veredicto",
    ]


def test_small_sector_mean_is_not_trimmed():
    peers_raw = pd.DataFrame({"Marg_EBIT": ["1", "2", "3", "100"]}, index=["AAA1", "BBB1", "CCC1", "DDD1"])

    ranking = compute_sector_ranking(
        peers_raw, "AAA1", "Bancos", _bancos("AAA1", "BBB1", "CCC1", "DDD1")
    )

    # With fewer than 10 observations trim_mean removes nothing: (1 + 2 + 3 + 100) / 4.
    assert _row(ranking, "Margem EBIT (%)")["Média Setor"] == 26.5


@pytest.mark.parametrize("ticker", ["TAEE11", "TAEE3", "TAEE4"])
def test_issuer_key_groups_share_classes_by_root(ticker):
    assert home_data._issuer_of(ticker) == "TAEE"
