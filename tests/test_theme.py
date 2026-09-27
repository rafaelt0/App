import re
import tomllib
from pathlib import Path

from utils.charts import CHART_PRIMARY, CHART_SURFACE, CHART_TEXT


def test_streamlit_defaults_to_the_shared_dark_chart_palette():
    project_root = Path(__file__).resolve().parents[1]
    theme = tomllib.loads(
        (project_root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )["theme"]
    css_root = (
        (project_root / "style.css")
        .read_text(encoding="utf-8")
        .split("}", 1)[0]
    )
    tokens = dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", css_root))

    assert theme["base"] == "dark"
    assert (
        theme["primaryColor"].lower()
        == CHART_PRIMARY
        == tokens["--brand-primary"]
    )
    assert theme["backgroundColor"].lower() == tokens["--surface-canvas"]
    assert (
        theme["secondaryBackgroundColor"].lower()
        == CHART_SURFACE
        == tokens["--surface-panel"]
    )
    assert (
        theme["textColor"].lower() == CHART_TEXT == tokens["--text-primary"]
    )
