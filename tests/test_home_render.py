from unittest.mock import patch

import pandas as pd

from utils.home_render import render_analyst_synthesis, render_star_button


def test_favorite_button_names_the_ticker():
    with (
        patch("utils.home_render._db.wl_has", return_value=False),
        patch("utils.home_render.st.button", return_value=False) as button,
    ):
        render_star_button("VALE3", "test-user")

    assert button.call_args.args[0] == "Favoritar · VALE3"


def test_analyst_synthesis_shows_simple_category_summary():
    synthesis = {
        "categorias": {
            "Valuation": {
                "veredicto": "Favorável",
                "percentil": 75,
                "indicadores_validos": 5,
            },
            "Rentabilidade": {
                "veredicto": "Favorável",
                "percentil": 80,
                "indicadores_validos": 4,
            },
            "Yield": {
                "veredicto": "Neutro",
                "percentil": 50,
                "indicadores_validos": 1,
            },
        },
        "categorias_validas": 3,
        "indicadores_validos": 10,
        "veredicto": "ATRATIVO",
        "pontos_positivos": [("P/L 1 · P75.0 · n=4", "tooltip")],
        "alertas": [],
        "pontos_negativos": [],
    }
    with (
        patch("utils.home_render.analyst_synthesis_header"),
        patch("utils.home_render.build_analyst_synthesis", return_value=synthesis),
        patch("utils.home_render.st.markdown") as markdown,
    ):
        render_analyst_synthesis(
            pd.DataFrame(index=["AAA1"]),
            pd.DataFrame({"Empresa": ["Empresa A"]}, index=["AAA1"]),
            ["AAA1"],
            pd.DataFrame(),
            pd.DataFrame({"Ticker": ["AAA1"], "Setor": ["Bancos"]}),
        )

    rendered = "\n".join(call.args[0] for call in markdown.call_args_list)
    assert "2/3" in rendered
    assert "Dividendos" in rendered
    assert 'class="analyst-synthesis-card"' in rendered
    assert '<details class="analyst-synthesis-details">' in rendered
