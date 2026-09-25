from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_market_consensus_waits_for_analysis_and_uses_selected_tickers():
    stocks = pd.DataFrame(
        {"Ticker": ["PETR4", "VALE3"], "Setor": ["Energia", "Mineração"],
         "Empresa": ["Petrobras", "Vale"], "RazaoSocial": ["Petróleo Brasileiro", "Vale S.A."]}
    )
    fundamentals = pd.DataFrame(
        {
            "Empresa": ["Petrobras", "Vale"],
            "Setor": ["Energia", "Mineração"],
            "Subsetor": ["Petróleo", "Mineração"],
            "Cotacao": [100, 200],
            "Min_52_sem": [90, 180],
            "Max_52_sem": [110, 220],
            "Vol_med_2m": [1_000_000, 2_000_000],
            "Valor_de_mercado": [1_000_000_000, 2_000_000_000],
            "Data_ult_cot": ["2026-09-01", "2026-09-01"],
            "Marg_Liquida": [0.1, 0.2],
            "Marg_EBIT": [0.2, 0.3],
            "ROE": [0.15, 0.16],
            "ROIC": [0.12, 0.13],
            "Div_Yield": [0.05, 0.04],
            "Cres_Rec_5a": [0.1, 0.15],
            "PL": [1000, 1200],
            "EV_EBITDA": [800, 900],
            "PVP": [150, 200],
        },
        index=["PETR4", "VALE3"],
    )

    def target_data(ticker):
        price = 100 if ticker == "PETR4" else 200
        return {
            "company_name": ticker,
            "currency": "BRL",
            "price": price,
            "target_low": price * 0.9,
            "target_mean": price * 1.1,
            "target_median": price * 1.08,
            "target_high": price * 1.2,
            "analyst_count": 5,
            "recommendation": "buy",
            "recommendation_mean": 2,
        }

    with (
        patch("utils.market_data.get_listed_stocks", return_value=stocks),
        patch(
            "utils.market_data.get_sorted_tickers_by_liquidity",
            side_effect=lambda tickers: tickers,
        ),
        patch(
            "utils.market_data.get_market_target_data", side_effect=target_data
        ) as fetch_targets,
        patch("utils.home_data.get_fundamentus_data", return_value=fundamentals) as fetch_fundamentals,
        patch("utils.home_data.get_sector_peers", return_value=pd.DataFrame()),
        patch("utils.home_data.clear_fundamentus_cache"),
        patch("utils.db.portfolio_get", return_value=([], None)),
        patch("utils.db.wl_get", return_value=[]),
        patch("utils.db.wl_has", return_value=False),
        patch("utils.identity.get_browser_uid", return_value="test"),
    ):
        app = AppTest.from_file("Main_Page.py").run()
        assert not fetch_targets.called
        selector = app.multiselect(key="selected_tickers")
        assert selector.options == [
            "PETR4  ·  Petrobras  ·  Energia",
            "VALE3  ·  Vale  ·  Mineração",
        ]
        assert selector.value == []
        assert len(app.multiselect) == 2  # Home selector and sidebar sector filter
        assert not app.text_input
        assert not app.selectbox
        assert not any(button.label == "Buscar" for button in app.button)

        selector.set_value(["PETR4"]).run()
        assert app.multiselect(key="selected_tickers").value == ["PETR4"]
        assert app.session_state["selected_tickers"] == ["PETR4"]
        assert not fetch_targets.called
        assert not fetch_fundamentals.called
        app.multiselect(key="selected_tickers").set_value(["PETR4", "VALE3"]).run()
        assert not fetch_targets.called
        assert not fetch_fundamentals.called

        app.button(key="btn_analisar").click().run()

        assert not app.exception
        fetch_fundamentals.assert_called_once_with(["PETR4", "VALE3"])
        assert fetch_targets.call_args_list[0].args == ("PETR4",)
        assert fetch_targets.call_args_list[1].args == ("VALE3",)
        target_table = next(
            element.value
            for element in app.dataframe
            if "Alvo médio" in element.value.columns
        )
        assert target_table["Ticker"].tolist() == ["PETR4", "VALE3"]
        assert target_table["Potencial"].tolist() == ["+10.0%", "+10.0%"]
