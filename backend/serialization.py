import datetime as dt
import json
import math
from collections.abc import Mapping
from decimal import Decimal

import numpy as np
import pandas as pd
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def clean_json(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, BaseModel):
        return clean_json(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        return [clean_json(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return clean_json(value.to_dict(orient="records"))
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, (float, Decimal)):
        return float(value) if math.isfinite(value) else None
    return value


def figure(fig):
    return clean_json(json.loads(fig.to_json()))


def csv_response(frame, filename, index=True):
    return Response(
        frame.to_csv(index=index).encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"},
    )
