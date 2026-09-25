from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_market_page_keeps_user_ticker_change_when_query_has_old_ticker():
    class FakeTicker:
        @property
        def info(self):
            return {
                "longName": "Teste",
                "currency": "BRL",
                "currentPrice": 100,
                "targetMeanPrice": 110,
                "targetLowPrice": 90,
                "targetHighPrice": 120,
                "numberOfAnalystOpinions": 5,
            }

    with (
        patch(
            "utils.market_data.get_listed_stocks",
            return_value=pd.DataFrame({"Ticker": ["PETR4", "VALE3"]}),
        ),
        patch("yfinance.Ticker", side_effect=lambda _: FakeTicker()),
        patch("utils.db.portfolio_get", return_value=([], None)),
        patch("utils.db.wl_has", return_value=False),
    ):
        app = AppTest.from_file("pages/4_Visão_de_mercado.py")
        app.run()

        assert app.text_input(key="valuation_ticker").value == ""
        app.button(key="valuation_quick_PETR4").click().run()
        assert app.session_state["valuation_ticker"] == "PETR4"

        app.text_input(key="valuation_ticker").set_value("VALE3").run()

        assert app.text_input(key="valuation_ticker").value == "VALE3"
        assert app.session_state["valuation_ticker"] == "VALE3"
