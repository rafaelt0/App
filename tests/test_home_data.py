import pandas as pd
import pytest

from utils.home_data import compute_sector_ranking


def _b3_data():
    return pd.DataFrame(
        {
            "Ticker": ["AAA1", "BBB1", "CCC1", "DDD1", "ZZZ1"],
            "Setor": ["Bancos", "Bancos", "Bancos", "Bancos", "Varejo"],
        }
    )


def test_compute_sector_ranking_favoravel_for_cheap_low_multiple():
    # AAA1 has the lowest P/L (lower is better) among its sector peers.
    peers_raw = pd.DataFrame(
        {"PL": ["5", "10", "15", "20"]}, index=["AAA1", "BBB1", "CCC1", "DDD1"]
    )
    rank = compute_sector_ranking(peers_raw, "AAA1", "Bancos", _b3_data())
    row = rank[rank["Múltiplo"] == "P/L"].iloc[0]
    assert row["Percentil"] == 75.0
    assert row["Veredicto"] == "Favorável"
    assert row["Peers (n)"] == 4


def test_compute_sector_ranking_desfavoravel_for_expensive_low_multiple():
    peers_raw = pd.DataFrame(
        {"PL": ["5", "10", "15", "20"]}, index=["AAA1", "BBB1", "CCC1", "DDD1"]
    )
    rank = compute_sector_ranking(peers_raw, "DDD1", "Bancos", _b3_data())
    row = rank[rank["Múltiplo"] == "P/L"].iloc[0]
    assert row["Percentil"] == 0.0
    assert row["Veredicto"] == "Desfavorável"


def test_compute_sector_ranking_higher_better_metric_inverts_direction():
    # ROE: higher is better, so the highest ROE gets the best percentile.
    peers_raw = pd.DataFrame(
        {"ROE": ["5", "10", "15", "20"]}, index=["AAA1", "BBB1", "CCC1", "DDD1"]
    )
    rank = compute_sector_ranking(peers_raw, "DDD1", "Bancos", _b3_data())
    row = rank[rank["Múltiplo"] == "ROE (%)"].iloc[0]
    assert row["Percentil"] == 75.0
    assert row["Veredicto"] == "Favorável"


def test_compute_sector_ranking_restricts_to_same_sector():
    peers_raw = pd.DataFrame(
        {"PL": ["5", "100", "15", "20"]},
        index=["AAA1", "ZZZ1", "CCC1", "DDD1"],  # ZZZ1 is a Varejo outlier
    )
    rank = compute_sector_ranking(peers_raw, "AAA1", "Bancos", _b3_data())
    row = rank[rank["Múltiplo"] == "P/L"].iloc[0]
    # Only the 3 Bancos peers (AAA1, CCC1, DDD1) should count — ZZZ1 excluded.
    assert row["Peers (n)"] == 3


def test_compute_sector_ranking_treats_zero_as_not_applicable():
    peers_raw = pd.DataFrame(
        {"PL": ["0", "10", "15", "20"]}, index=["AAA1", "BBB1", "CCC1", "DDD1"]
    )
    rank = compute_sector_ranking(peers_raw, "BBB1", "Bancos", _b3_data())
    row = rank[rank["Múltiplo"] == "P/L"].iloc[0]
    # AAA1's zero P/L is dropped, leaving 3 peers (BBB1, CCC1, DDD1).
    assert row["Peers (n)"] == 3


def test_compute_sector_ranking_ticker_not_in_peers_returns_empty():
    peers_raw = pd.DataFrame({"PL": ["5", "10"]}, index=["BBB1", "CCC1"])
    rank = compute_sector_ranking(peers_raw, "AAA1", "Bancos", _b3_data())
    assert rank.empty
