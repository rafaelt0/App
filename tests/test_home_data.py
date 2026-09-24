import pandas as pd

from utils.home_data import build_analyst_synthesis, compute_sector_ranking


def _b3_data():
    return pd.DataFrame(
        {
            "Ticker": ["AAA1", "BBB1", "CCC1", "DDD1", "ZZZ1"],
            "Setor": ["Bancos", "Bancos", "Bancos", "Bancos", "Varejo"],
        }
    )


def _sector_peers(valuation_values, profitability_values, yield_values):
    return pd.DataFrame(
        {
            "PL": valuation_values,
            "PVP": valuation_values,
            "EV_EBITDA": valuation_values,
            "EV_EBIT": valuation_values,
            "PSR": valuation_values,
            "ROE": profitability_values,
            "ROIC": profitability_values,
            "Marg_EBIT": profitability_values,
            "Marg_Liquida": profitability_values,
            "Div_Yield": yield_values,
        },
        index=["AAA1", "BBB1", "CCC1", "DDD1"],
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


def test_build_analyst_synthesis_marks_three_favorable_categories_attractive():
    peers_raw = _sector_peers(
        [1, 2, 3, 4],
        [0.4, 0.3, 0.2, 0.1],
        [0.06, 0.05, 0.04, 0.03],
    )

    synthesis = build_analyst_synthesis(peers_raw, "AAA1", "Bancos", _b3_data())

    assert synthesis["veredicto"] == "ATRATIVO"
    assert synthesis["cor_veredicto"] == "#00ff87"
    assert synthesis["categorias_validas"] == 3
    assert all(
        category["veredicto"] == "Favorável"
        for category in synthesis["categorias"].values()
    )
    pl_signal = next(
        signal
        for signal in synthesis["pontos_positivos"]
        if signal[0].startswith("P/L ")
    )
    assert pl_signal[0] == "P/L 1 · P75.0 · n=4"
    assert "percentil 75.0" in pl_signal[1]
    roe_signal = next(
        signal
        for signal in synthesis["pontos_positivos"]
        if signal[0].startswith("ROE ")
    )
    assert roe_signal[0] == "ROE 40% · P75.0 · n=4"
    assert "valor 40%" in roe_signal[1]


def test_build_analyst_synthesis_marks_three_unfavorable_categories_weak():
    peers_raw = _sector_peers(
        [4, 3, 2, 1],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    )

    synthesis = build_analyst_synthesis(peers_raw, "AAA1", "Bancos", _b3_data())

    assert synthesis["veredicto"] == "FRACO"
    assert synthesis["cor_veredicto"] == "#ff3d5a"
    assert synthesis["categorias_validas"] == 3
    assert all(
        category["veredicto"] == "Desfavorável"
        for category in synthesis["categorias"].values()
    )


def test_build_analyst_synthesis_returns_insufficient_without_peers():
    synthesis = build_analyst_synthesis(
        pd.DataFrame(),
        "AAA1",
        "Bancos",
        _b3_data(),
    )

    assert synthesis["veredicto"] == "DADOS INSUFICIENTES"
    assert synthesis["cor_veredicto"] == "#94a3b8"
    assert synthesis["indicadores_validos"] == 0
    assert synthesis["categorias_validas"] == 0


def test_build_analyst_synthesis_requires_two_available_categories():
    peers_raw = pd.DataFrame(
        {
            "PL": [1, 2, 3, 4],
            "PVP": [1, 2, 3, 4],
            "EV_EBITDA": [1, 2, 3, 4],
            "EV_EBIT": [1, 2, 3, 4],
            "PSR": [1, 2, 3, 4],
        },
        index=["AAA1", "BBB1", "CCC1", "DDD1"],
    )

    synthesis = build_analyst_synthesis(peers_raw, "AAA1", "Bancos", _b3_data())

    assert synthesis["veredicto"] == "DADOS INSUFICIENTES"
    assert synthesis["indicadores_validos"] == 5
    assert synthesis["categorias_validas"] == 1
    assert synthesis["categorias"]["Valuation"]["veredicto"] == "Favorável"


def test_build_analyst_synthesis_excludes_indicators_with_fewer_than_three_peers():
    peers_raw = pd.DataFrame(
        {
            "PL": [1, 2, None, None],
            "PVP": [1, 2, 3, None],
            "ROE": [1, 2, 3, 4],
            "Div_Yield": [1, 2, 3, 4],
        },
        index=["AAA1", "BBB1", "CCC1", "DDD1"],
    )

    synthesis = build_analyst_synthesis(peers_raw, "AAA1", "Bancos", _b3_data())

    assert synthesis["veredicto"] == "FRACO"
    assert synthesis["indicadores_validos"] == 3
    assert synthesis["categorias_validas"] == 3
    assert synthesis["categorias"]["Valuation"] == {
        "percentil": 66.7,
        "veredicto": "Neutro",
        "indicadores_validos": 1,
    }
    assert not any(
        label.startswith("P/L ")
        for group in (
            synthesis["pontos_positivos"],
            synthesis["pontos_negativos"],
            synthesis["alertas"],
        )
        for label, _ in group
    )
    assert any(
        label == "P/VP 1 · P66.7 · n=3"
        for label, _ in synthesis["alertas"]
    )


def test_build_analyst_synthesis_weights_categories_equally():
    peers_raw = _sector_peers(
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [2, 1, 1.5, 3],
    )

    synthesis = build_analyst_synthesis(peers_raw, "AAA1", "Bancos", _b3_data())

    assert synthesis["categorias"] == {
        "Valuation": {
            "percentil": 75.0,
            "veredicto": "Favorável",
            "indicadores_validos": 5,
        },
        "Rentabilidade": {
            "percentil": 0.0,
            "veredicto": "Desfavorável",
            "indicadores_validos": 4,
        },
        "Yield": {
            "percentil": 50.0,
            "veredicto": "Neutro",
            "indicadores_validos": 1,
        },
    }
    assert synthesis["veredicto"] == "NEUTRO"
