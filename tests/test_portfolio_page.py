from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_portfolio_loads_quotes_only_on_click_and_names_missing_ticker():
    dates = pd.bdate_range("2025-01-01", periods=40)
    prices = pd.DataFrame(
        {"PETR4.SA": [float("nan")] * 40, "VALE3.SA": [100.0] * 40},
        index=dates,
    )
    calls = []

    def get_prices(*args):
        calls.append(args)
        return prices.copy()

    stocks = pd.DataFrame({"Ticker": ["PETR4", "VALE3"], "Setor": ["Energia", "Mineração"],
                           "Empresa": ["Petrobras", "Vale"],
                           "RazaoSocial": ["Petróleo Brasileiro", "Vale S.A."]})
    with (
        patch("utils.market_data.get_listed_stocks", return_value=stocks),
        patch("utils.portfolio_data.get_portfolio_prices", get_prices),
        patch("utils.portfolio_data.get_selic_rate", lambda: 0.0005),
        patch("utils.db.portfolio_get", lambda uid: ([], {})),
        patch("utils.identity.get_browser_uid", lambda: "test-visitor"),
    ):
        app = AppTest.from_file("pages/1_Portfolio.py", default_timeout=30)
        app.session_state["selected_tickers"] = ["PETR4", "VALE3"]
        app.run()
        assert not app.exception
        assert not calls
        selector = app.multiselect(key="selected_tickers")
        assert selector.options == ["PETR4  ·  Petrobras", "VALE3  ·  Vale"]
        assert selector.value == ["PETR4", "VALE3"]
        assert not any(widget.key == "portfolio_stock_search" for widget in app.text_input)
        assert not any(button.label == "Buscar" for button in app.button)

        selector.set_value(["PETR4"]).run()
        assert app.session_state["selected_tickers"] == ["PETR4"]
        assert not calls
        app.multiselect(key="selected_tickers").set_value(["PETR4", "VALE3"]).run()
        assert app.session_state["selected_tickers"] == ["PETR4", "VALE3"]
        assert not calls

        next(button for button in app.button if button.label == "Carregar portfólio").click().run()
        assert not app.exception
        assert len(calls) == 1
        assert any("PETR4 (sem cotações)" in error.value for error in app.error)


def test_ibov_outage_still_passes_portfolio_to_simulation():
    import numpy as np

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=70)
    days = np.arange(len(dates))
    prices = pd.DataFrame(
        {
            "PETR4.SA": 100 + days + 2 * np.sin(days),
            "VALE3.SA": 90 + 0.8 * days + np.cos(days),
        },
        index=dates,
    )
    raw_prices = pd.DataFrame(
        {"PETR4.SA": [10.0] * len(dates), "VALE3.SA": [20.0] * len(dates)},
        index=dates,
    )
    with (
        patch("utils.portfolio_data.get_portfolio_prices", lambda *args: prices.copy()),
        patch("utils.portfolio_data.get_portfolio_trade_prices", lambda *args: raw_prices.copy()),
        patch("utils.portfolio_data.get_benchmark_prices", side_effect=OSError("offline")),
        patch("utils.portfolio_data.get_selic_rate", lambda: 0.0005),
        patch("utils.db.portfolio_get", lambda uid: ([], {})),
        patch("utils.db.portfolio_save", lambda *args: None),
        patch("utils.identity.get_browser_uid", lambda: "test-visitor"),
        patch("streamlit.page_link", lambda *args, **kwargs: None),
    ):
        app = AppTest.from_file("pages/1_Portfolio.py", default_timeout=30)
        app.session_state["selected_tickers"] = ["PETR4", "VALE3"]
        app.run()
        app.radio[0].set_value("Alocação Manual").run()
        next(button for button in app.button if button.label == "Carregar portfólio").click().run()

        assert not app.exception
        assert app.session_state["portfolio_analysis_tickers"] == ["PETR4", "VALE3"]
        assert app.session_state["pesos_manuais"] == {"PETR4.SA": 0.5, "VALE3.SA": 0.5}
        assert app.session_state["retorno_bench"] is None
        pd.testing.assert_frame_equal(
            app.session_state["returns"], prices.pct_change(fill_method=None).dropna()
        )
        trade_table = next(
            element.value for element in app.dataframe
            if "Preço Unitário" in element.value.columns
        )
        assert trade_table.loc[0, "Preço Unitário"] == "R$ 10.00"
        assert trade_table.loc[1, "Preço Unitário"] == "R$ 20.00"
        assert trade_table.loc[0, "Cotas a Comprar"] == "500"
        assert trade_table.loc[1, "Cotas a Comprar"] == "250"
        assert any("69 retornos diários completos" in item.value for item in app.caption)
        assert any("A alocação manual descreve" in item.value for item in app.caption)


def test_zero_weight_manual_asset_does_not_limit_return_sample():
    import numpy as np

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=70)
    days = np.arange(len(dates))
    prices = pd.DataFrame(
        {
            "PETR4.SA": 100 + days + np.sin(days),
            "VALE3.SA": [float("nan")] * len(dates),
        },
        index=dates,
    )
    raw_prices = pd.DataFrame({"PETR4.SA": [10.0] * len(dates)}, index=dates)
    with (
        patch("utils.portfolio_data.get_portfolio_prices", lambda *args: prices.copy()),
        patch("utils.portfolio_data.get_portfolio_trade_prices", lambda *args: raw_prices.copy()),
        patch("utils.portfolio_data.get_benchmark_prices", side_effect=OSError("offline")),
        patch("utils.portfolio_data.get_selic_rate", lambda: 0.0005),
        patch("utils.db.portfolio_get", lambda uid: ([], {})),
        patch("utils.db.portfolio_save", lambda *args: None),
        patch("utils.identity.get_browser_uid", lambda: "test-visitor"),
        patch("streamlit.page_link", lambda *args, **kwargs: None),
    ):
        app = AppTest.from_file("pages/1_Portfolio.py", default_timeout=30)
        app.session_state["selected_tickers"] = ["PETR4", "VALE3"]
        app.run()
        app.radio[0].set_value("Alocação Manual").run()
        app.number_input(key="peso_manual_PETR4").set_value(100)
        app.number_input(key="peso_manual_VALE3").set_value(0)
        app.run()
        next(button for button in app.button if button.label == "Carregar portfólio").click().run()

        assert not app.exception
        assert list(app.session_state["returns"].columns) == ["PETR4.SA"]
        assert app.session_state["pesos_manuais"] == {"PETR4.SA": 1.0, "VALE3.SA": 0.0}
        assert any("Ativos com peso zero foram excluídos" in item.value for item in app.caption)
