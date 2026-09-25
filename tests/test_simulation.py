import pandas as pd

from utils.simulation import simulate_portfolio


def test_trajectory_display_count_is_not_a_simulation_input(monkeypatch):
    import utils.simulation as simulation

    import streamlit as st

    monkeypatch.setattr(st, "session_state", {})
    calls = 0
    original = simulation.np.random.multivariate_normal

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation.np.random, "multivariate_normal", counted)
    args = ((0.001,), ((0.01,),), (1.0,), 20, 10, 1000.0, "2026-01-01")
    first = simulate_portfolio(*args)
    # The displayed trajectory count is a rendering-only setting, so reruns
    # with the same simulation inputs reuse paths instead of generating them.
    second = simulate_portfolio(*args)
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (21, 10)
    assert calls == 1


def test_page_keeps_simulation_metrics_when_display_count_changes():
    import numpy as np
    from unittest.mock import patch

    from streamlit.testing.v1 import AppTest

    returns = pd.DataFrame(
        np.random.default_rng(3).normal(0.0003, 0.01, (80, 2)),
        index=pd.date_range("2024-01-01", periods=80, freq="B"),
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
        "pesos_manuais": {"AAA3.SA": 0.6, "BBB4.SA": 0.4},
        "peso_manual_df": pd.DataFrame(
            {"Peso": [0.6, 0.4]}, index=["AAA3.SA", "BBB4.SA"]
        ),
    }.items():
        app.session_state[key] = value

    # AppTest treats this multipage file as a standalone app, so stub navigation.
    with patch("streamlit.page_link", lambda *args, **kwargs: None):
        app.run()
        submit = next(button for button in app.button if button.label == "Rodar Simulação")
        submit.click().run()
        assert not app.exception
        metrics_before = [(metric.label, metric.value) for metric in app.metric]

        trajectories = next(
            widget
            for widget in app.number_input
            if widget.label == "Número de trajetórias exibidas"
        )
        trajectories.set_value(10).run()

    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric] == metrics_before
