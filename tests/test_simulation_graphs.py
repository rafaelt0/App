import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def test_simulation_graphs_keep_trajectory_and_quartile_labels_readable():
    returns = pd.DataFrame(
        np.random.default_rng(3).normal(0.0003, 0.01, (40, 2)),
        columns=["AAA3.SA", "BBB4.SA"],
    )
    app = AppTest.from_file("pages/2_Simulação.py")
    for key, value in {
        "selected_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded_tickers": ["AAA3", "BBB4"],
        "portfolio_analysis_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded": True,
        "modo": "Alocação Manual",
        "returns": returns,
        "pesos_manuais": {"AAA3.SA": 0.5, "BBB4.SA": 0.5},
        "peso_manual_df": pd.DataFrame(
            {"Peso": [0.5, 0.5]}, index=["AAA3.SA", "BBB4.SA"]
        ),
        "sim_n_simulations_input": 10,
        "sim_years_input": 1,
    }.items():
        app.session_state[key] = value

    def fake_simulation(*args):
        days, simulations, initial_value, start_date = args[3:]
        paths = np.linspace(initial_value, initial_value * 1.2, days + 1)[:, None]
        paths = paths * np.linspace(0.8, 1.2, simulations)[None, :]
        return pd.DataFrame(
            paths,
            index=pd.date_range(start_date, periods=days + 1, freq="B"),
        )

    with patch("streamlit.page_link", lambda *args, **kwargs: None), patch(
        "utils.simulation.simulate_portfolio", fake_simulation
    ):
        app.run()
        next(button for button in app.button if button.label == "Rodar Simulação").click().run()

    assert not app.exception
    charts = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    individual = next(
        chart for chart in charts
        if chart["layout"]["title"]["text"].startswith("Exemplos de Trajetórias")
    )
    assert individual["layout"]["showlegend"] is False
    assert individual["layout"]["margin"]["b"] == 55

    histogram = next(
        chart for chart in charts
        if chart["layout"]["title"]["text"].startswith("Distribuição dos Valores")
    )
    percentile_annotations = {
        annotation["text"]: annotation["yshift"]
        for annotation in histogram["layout"]["annotations"]
        if annotation["text"] in {"Q1 (25%)", "Mediana (50%)", "Q3 (75%)"}
    }
    assert sorted(percentile_annotations.values()) == [0, 18, 36]
