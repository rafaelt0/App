"""Pure screener presets and filtering, separate from Streamlit rendering."""

import pandas as pd


PRESET_FILTERS = {
    "Explorar B3": {"liq2m_min": 1_000_000},
    "Lucro a preço moderado": {
        "pl_min": 0,
        "pl_min_exclusive": True,
        "pl_max": 15,
        "roe_min": 0.12,
        "liq2m_min": 1_000_000,
    },
    "Renda atual": {
        "dy_min": 0.05,
        "roe_min": 0.10,
        "liq2m_min": 1_000_000,
    },
    "Qualidade rentável": {
        "pl_min": 0,
        "pl_min_exclusive": True,
        "pl_max": 20,
        "roe_min": 0.18,
        "liq2m_min": 2_000_000,
    },
    "Alta liquidez": {"liq2m_min": 10_000_000},
    "Crescimento com lucro": {
        "c5y_min": 0.10,
        "roe_min": 0.12,
        "liq2m_min": 1_000_000,
    },
    "Personalizado": {},
}

_FILTER_COLUMNS = {
    "pl_min": "pl",
    "pl_max": "pl",
    "roe_min": "roe",
    "dy_min": "dy",
    "c5y_min": "c5y",
    "liq2m_min": "liq2m",
}


def prepare_export(frame: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Return every result with headers and percentage-point values ready for CSV."""
    columns = [column for column in column_map if column in frame.columns]
    exported = frame[columns].copy()
    for column in ("dy", "roe", "roic", "mrgebit", "mrgliq", "c5y"):
        if column in exported.columns:
            exported[column] *= 100
    exported = exported.rename(columns=column_map)
    exported.index.name = "Papel"
    return exported


def filter_stocks(frame: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """Apply only active criteria; an active criterion needs a present value."""
    mask = pd.Series(True, index=frame.index)
    for criterion, value in criteria.items():
        if criterion == "pl_min_exclusive":
            continue
        column = _FILTER_COLUMNS.get(criterion)
        if column is None:
            raise ValueError(f"Filtro desconhecido: {criterion}")
        if column not in frame.columns:
            raise ValueError(f"Coluna necessária para o filtro indisponível: {column}")

        values = frame[column]
        valid = values.notna()
        if criterion == "pl_min":
            if criteria.get("pl_min_exclusive", False):
                valid &= values > value
            else:
                valid &= values >= value
        elif criterion.endswith("_min"):
            valid &= values >= value
        else:
            valid &= values <= value
        mask &= valid

    return frame.loc[mask].copy()
