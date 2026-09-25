from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_main_page_market_target_accepts_handoff_and_typed_changes():
    class FakeTicker:
        @property
        def info(self):
            return {
                "longName": "Teste",
                "currency": "BRL",
                "currentPrice": 100,
                "targetMeanPrice": 110,
                "targetLowPrice": 90,
                "targetMedianPrice": 108,
                "targetHighPrice": 120,
                "numberOfAnalystOpinions": 5,
                "recommendationKey": "strongSell",
            }

    stocks = pd.DataFrame(
        {"Ticker": ["PETR4", "VALE3"], "Setor": ["Energia", "Mineração"]}
    )
    with (
        patch("utils.market_data.get_listed_stocks", return_value=stocks),
        patch(
            "utils.market_data.get_sorted_tickers_by_liquidity",
            side_effect=lambda tickers: tickers,
        ),
        patch("yfinance.Ticker", side_effect=lambda _: FakeTicker()),
        patch("utils.db.portfolio_get", return_value=([], None)),
        patch("utils.db.wl_get", return_value=[]),
        patch("utils.db.wl_has", return_value=False),
        patch("utils.identity.get_browser_uid", return_value="test"),
    ):
        app = AppTest.from_file("Main_Page.py")
        app.session_state["_market_target_handoff_ticker"] = "PETR4"
        app.run()

        assert app.text_input(key="market_target_ticker").value == "PETR4"
        assert any(metric.label == "Preço-alvo médio" for metric in app.metric)
        assert any(metric.value == "Venda forte" for metric in app.metric)

        app.text_input(key="market_target_ticker").set_value("VALE3").run()
        assert app.text_input(key="market_target_ticker").value == "VALE3"
        assert app.session_state["market_target_ticker"] == "VALE3"
