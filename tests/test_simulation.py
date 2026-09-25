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


def test_annualized_log_return_stats_uses_more_daily_observations_to_reduce_iid_se():
    import numpy as np
    from utils.simulation import annualized_log_return_stats

    returns = np.array([[0.01], [-0.02], [0.03], [0.0]])
    annual_mean, standard_error = annualized_log_return_stats(returns, [1.0])
    daily_log = np.log1p(returns[:, 0])
    assert np.isclose(annual_mean, 252 * daily_log.mean())
    assert np.isclose(standard_error, 252 * daily_log.std(ddof=1) / np.sqrt(4))

    repeated_returns = np.tile(returns, (4, 1))
    _, longer_standard_error = annualized_log_return_stats(repeated_returns, [1.0])
    repeated_log = np.log1p(repeated_returns[:, 0])
    expected = 252 * repeated_log.std(ddof=1) / np.sqrt(len(repeated_log))
    assert np.isclose(longer_standard_error, expected)
    assert longer_standard_error < standard_error


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


def test_page_drops_incomplete_days_before_calibration():
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
        "sim_n_simulations_input": 10,
    }.items():
        app.session_state[key] = value

    with patch("streamlit.page_link", lambda *args, **kwargs: None):
        app.run()
        submit = next(button for button in app.button if button.label == "Rodar Simulação")
        submit.click().run()

    assert not app.exception
    assert not any("Retornos históricos inválidos" in error.value for error in app.error)
    assert any("39 retornos diários completos" in caption.value for caption in app.caption)


def test_longer_simulation_calibration_uses_more_rows_and_fixed_horizon():
    import numpy as np
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest

    days = np.arange(1300)
    prices = pd.DataFrame(
        {
            "AAA3.SA": 100 * np.exp(0.0004 * days + 0.01 * np.sin(days / 20)),
            "BBB4.SA": np.nan,
        },
        index=pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(days)),
    )
    returns = pd.DataFrame(
        np.zeros((40, 2)),
        index=pd.date_range(end=pd.Timestamp.today().normalize(), periods=40, freq="B"),
        columns=["AAA3.SA", "BBB4.SA"],
    )
    simulation_calls = []
    bootstrap_rows = []

    def simulate(mu, covariance, weights, days, simulations, initial_value, start_date):
        simulation_calls.append((days, tuple(weights)))
        index = pd.date_range(start=start_date, periods=days + 1, freq="B")
        return pd.DataFrame(np.full((days + 1, simulations), initial_value), index=index)

    def bootstrap(asset_returns, weights, days, simulations, initial_value):
        bootstrap_rows.append(len(asset_returns))
        return np.full(simulations, initial_value)

    app = AppTest.from_file("pages/2_Simulação.py")
    for key, value in {
        "selected_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded_tickers": ["AAA3", "BBB4"],
        "portfolio_analysis_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded": True,
        "modo": "Alocação Manual",
        "returns": returns,
        "peso_manual_df": pd.DataFrame(
            {"Peso": [1.0, 0.0]}, index=["AAA3.SA", "BBB4.SA"]
        ),
        "sim_n_simulations_input": 10,
        "sim_years_input": 1,
    }.items():
        app.session_state[key] = value

    with (
        patch("streamlit.page_link", lambda *args, **kwargs: None),
        patch("utils.portfolio_data.get_portfolio_prices", return_value=prices) as fetch_prices,
        patch("utils.simulation.simulate_portfolio", side_effect=simulate),
        patch("utils.simulation.bootstrap_terminal_values", side_effect=bootstrap),
    ):
        app.run()
        assert not fetch_prices.called
        app.selectbox(key="sim_calibration_window").set_value("5 Anos")
        next(button for button in app.button if button.label == "Rodar Simulação").click().run()

    assert not app.exception
    assert fetch_prices.call_count == 1
    assert fetch_prices.call_args.args[0] == ("AAA3.SA",)
    assert simulation_calls == [(252, (1.0,))]
    assert bootstrap_rows == [len(prices.index) - 1]
    assert any(
        f"{len(prices.index) - 1} retornos diários completos" in caption.value
        for caption in app.caption
    )
    comparison = next(
        element.value for element in app.dataframe
        if "Retornos completos" in element.value.columns
    )
    assert comparison["Janela"].tolist() == ["2 anos", "3 anos", "5 anos"]
    counts = comparison["Retornos completos"].tolist()
    assert counts[0] < counts[1] < counts[2]


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
        stats_cards = next(
            block.value for block in app.markdown
            if "Valor Esperado Final" in block.value
        )
        assert stats_cards.count('class="mcard-label"') == 6
        assert all(
            label not in stats_cards
            for label in (
                "Probabilidade de Ganho",
                "Retorno Anual Esperado",
                "Retorno final P5",
            )
        )

        trajectories = next(
            widget
            for widget in app.number_input
            if widget.label == "Número de trajetórias exibidas"
        )
        trajectories.set_value(10).run()

    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric] == metrics_before


def test_bootstrap_resamples_joint_days_and_invalidates_changed_history(monkeypatch):
    import numpy as np
    import streamlit as st
    from utils.simulation import bootstrap_terminal_values

    monkeypatch.setattr(st, "session_state", {})
    # Independently drawing assets would create +/-20% days; joint rows cancel exactly.
    returns = np.array([[0.2, -0.2], [-0.2, 0.2]])
    args = ((0.5, 0.5), 5, 100, 100.0)
    first = bootstrap_terminal_values(returns, *args, chunk_size=7)
    np.testing.assert_allclose(first, 100.0)
    assert bootstrap_terminal_values(returns, *args, chunk_size=7) is first

    # A changed middle observation must invalidate the cache, even with the same shape.
    changed = np.array([[0.2, -0.2], [-0.2, 0.0]])
    second = bootstrap_terminal_values(changed, *args, chunk_size=7)
    assert second is not first
    assert (second < 100).any()
    crash_days = np.array([[-0.3], [0.0]])
    crash_paths = bootstrap_terminal_values(crash_days, (1.0,), 1, 100, 100.0)
    assert set(np.round(crash_paths, 2)) == {70.0, 100.0}
    with np.testing.assert_raises(ValueError):
        bootstrap_terminal_values(np.array([[-1.0, 0.0], [0.0, 0.0]]), *args)


def test_page_discloses_sample_and_compares_model_tails():
    import numpy as np
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest

    returns = pd.DataFrame(
        np.zeros((40, 2)),
        index=pd.date_range("2024-01-01", periods=40, freq="B"),
        columns=["AAA3.SA", "BBB4.SA"],
    )
    returns.iloc[20] = [-0.3, 0.1]
    app = AppTest.from_file("pages/2_Simulação.py")
    for key, value in {
        "selected_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded_tickers": ["AAA3", "BBB4"],
        "portfolio_analysis_tickers": ["AAA3", "BBB4"],
        "portfolio_loaded": True,
        "modo": "Alocação Manual",
        "returns": returns,
        "peso_manual_df": pd.DataFrame({"Peso": [0.5, 0.5]}, index=["AAA3", "BBB4"]),
        "sim_n_simulations_input": 100,
        "sim_years_input": 3,
    }.items():
        app.session_state[key] = value

    with patch("streamlit.page_link", lambda *args, **kwargs: None):
        app.run()
        next(button for button in app.button if button.label == "Rodar Simulação").click().run()

    assert not app.exception
    assert len(app.warning) == 2
    assert any("01/01/2024" in caption.value and "erro-padrão" in caption.value for caption in app.caption)
    assert any("Comparação de modelos" in block.value for block in app.markdown)
    assert any("Amplitude entre P5 e P95" in block.value for block in app.markdown)
    assert any("crises ausentes" in caption.value for caption in app.caption)
    assert {metric.label for metric in app.metric} == {
        "CAGR do valor final médio", "Probabilidade de Ganho", "Retorno final P5"
    }
