from unittest.mock import patch

from utils.home_render import render_star_button


def test_favorite_button_names_the_ticker():
    with (
        patch("utils.home_render._db.wl_has", return_value=False),
        patch("utils.home_render.st.button", return_value=False) as button,
    ):
        render_star_button("VALE3", "test-user")

    assert button.call_args.args[0] == "Favoritar · VALE3"
