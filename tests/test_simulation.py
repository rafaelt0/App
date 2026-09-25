import pandas as pd

from utils.simulation import simulate_portfolio


def test_trajectory_display_count_is_not_a_simulation_input(monkeypatch):
    import utils.simulation as simulation

    import streamlit as st

    monkeypatch.setattr(st, "session_state", {})
    calls = 0
    original = simulation.np.random.default_rng

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation.np.random, "default_rng", counted)
    args = ((0.001,), ((0.01,),), (1.0,), 20, 10, 1000.0, "2026-01-01")
    first = simulate_portfolio(*args)
    # The displayed trajectory count is a rendering-only setting, so reruns
    # with the same simulation inputs reuse paths instead of generating them.
    second = simulate_portfolio(*args)
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (21, 10)
    assert calls == 1


def test_simulation_daily_rebalances_and_does_not_change_global_rng(monkeypatch):
    import numpy as np
    import streamlit as st
    import utils.simulation as simulation

    monkeypatch.setattr(st, "session_state", {})
    original = np.random.get_state()
    paths = simulate_portfolio(
        (np.log(1.1), np.log(0.9)), ((0.0, 0.0), (0.0, 0.0)),
        (0.5, 0.5), 2, 3, 100.0, "2026-01-01"
    )
    after = np.random.get_state()
    assert all((a == b).all() if hasattr(a, "shape") else a == b for a, b in zip(original, after))
    assert paths.shape == (3, 3)
    # Equal-weight daily returns cancel; buy-and-hold would end at R$101.
    assert (paths.iloc[-1] == 100).all()


def test_simulation_chunks_and_cache_are_deterministic(monkeypatch):
    import streamlit as st
    import utils.simulation as simulation

    monkeypatch.setattr(st, "session_state", {})
    calls = []
    original = simulation.np.random.default_rng

    class TrackingRng:
        def __init__(self, rng):
            self.rng = rng

        def multivariate_normal(self, *args, **kwargs):
            calls.append(kwargs.get("size"))
            return self.rng.multivariate_normal(*args, **kwargs)

    monkeypatch.setattr(simulation.np.random, "default_rng", lambda *args: TrackingRng(original(*args)))
    args = ((0.001,), ((0.01,),), (1.0,), 3, 7, 1000.0, "2026-01-01")
    first = simulate_portfolio(*args, chunk_size=3)
    assert len(calls) == 3
    second = simulate_portfolio(*args, chunk_size=3)
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 3
    simulate_portfolio(*args, chunk_size=2)
    assert len(calls) == 7  # Chunk size changes the generated paths/cache key.


def test_page_uses_displayed_allocation_without_manual_weight_state():
    import numpy as np
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest

    returns = pd.DataFrame(np.zeros((40, 2)), columns=["AAA3.SA", "BBB4.SA"])
    app = AppTest.from_file("pages/2_Simulação.py")
    for key, value in {
        "selected_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded_tickers": ["AAA3", "BBB4"],
        "portfolio_analysis_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded": True,
        "modo": "Otimização de Markowitz",
        "returns": returns,
        "peso_manual_df": pd.DataFrame({"Peso": [0.8, 0.2]}, index=["AAA3.SA", "BBB4.SA"]),
    }.items():
        app.session_state[key] = value
    with patch("streamlit.page_link", lambda *args, **kwargs: None):
        app.run()
        assert not app.exception
        assert "pesos_manuais" not in app.session_state


def test_page_rejects_incomplete_return_history():
    import numpy as np
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest

    returns = pd.DataFrame(
        np.zeros((40, 2)), columns=["AAA3.SA", "BBB4.SA"]
    )
    returns.iloc[3, 0] = np.nan
    app = AppTest.from_file("pages/2_Simulação.py")
    for key, value in {
        "selected_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded_tickers": ["AAA3", "BBB4"],
        "portfolio_analysis_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded": True,
        "modo": "Alocação Manual",
        "returns": returns,
        "peso_manual_df": pd.DataFrame(
            {"Peso": [0.5, 0.5]}, index=["AAA3.SA", "BBB4.SA"]
        ),
    }.items():
        app.session_state[key] = value

    with patch("streamlit.page_link", lambda *args, **kwargs: None):
        app.run()
        submit = next(button for button in app.button if button.label == "Rodar Simulação")
        submit.click().run()

    assert not app.exception
    assert any("Retornos históricos inválidos" in error.value for error in app.error)


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
        "pesos_manuais": {"AAA3.SA": 0.1, "BBB4.SA": 0.9},
        "peso_manual_df": pd.DataFrame(
            {"Peso": [0.0000001, 0.9999999]}, index=["AAA3.SA", "BBB4.SA"]
        ),
    }.items():
        app.session_state[key] = value

    import utils.simulation as simulation

    # AppTest treats this multipage file as a standalone app, so stub navigation.
    observed_weights = []
    original_simulate = simulation.simulate_portfolio

    def capture_weights(*args, **kwargs):
        observed_weights.append(args[2])
        return original_simulate(*args, **kwargs)

    with patch("streamlit.page_link", lambda *args, **kwargs: None), patch(
        "utils.simulation.simulate_portfolio", capture_weights
    ):
        app.run()
        submit = next(button for button in app.button if button.label == "Rodar Simulação")
        submit.click().run()
        assert not app.exception
        assert observed_weights == [(0.0000001, 0.9999999)]
        metrics_before = [(metric.label, metric.value) for metric in app.metric]

        trajectories = next(
            widget
            for widget in app.number_input
            if widget.label == "Número de trajetórias exibidas"
        )
        trajectories.set_value(10).run()

    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric] == metrics_before
