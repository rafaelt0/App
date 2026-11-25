"""After-tax and real-return hurdle endpoint (WP-1.1).

Financial policy lives in ``utils/tax_hurdle.py``; this module only validates
the owned portfolio context and serializes the result.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from backend.serialization import StrictModel, clean_json, figure
from backend.state import require_portfolio, session
from utils.regime import HURDLE_METADATA, TAX_PROFILE_METADATA
from utils.tax_hurdle import analyze_hurdle

router = APIRouter(prefix="/api/portfolio", tags=["hurdle"])

Hurdle = Literal["CDI", "IPCA_PLUS_6", "SOFR", "US_CPI"]
TaxProfile = Literal["BR_PF", "US_TAXABLE", "NONE"]


class HurdleRequest(StrictModel):
    revision: int = Field(ge=0)
    portfolio_id: str = Field(min_length=1)
    real_return: bool = False
    net_of_tax: bool = True
    hurdle: Hurdle = "CDI"
    tax_profile: TaxProfile = "BR_PF"
    small_sale_exempt: bool = False


@router.post("/hurdle")
def hurdle_analysis(body: HurdleRequest, request: Request):
    """Compare the owned portfolio against the selected after-tax hurdle."""
    ctx = session(request)
    portfolio = require_portfolio(ctx, returns=True)
    if portfolio.id != body.portfolio_id or portfolio.revision != body.revision:
        raise HTTPException(409, "O portfólio mudou desde o cálculo. Atualize o resultado e tente novamente.")

    try:
        result = analyze_hurdle(
            portfolio.returns,
            portfolio.weights,
            hurdle=body.hurdle,
            tax_profile=body.tax_profile,
            real_return=body.real_return,
            net_of_tax=body.net_of_tax,
            small_sale_exempt=body.small_sale_exempt,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    chart = result.pop("chart")
    payload = {
        **result,
        "chart": figure(chart),
        "labels": {
            "hurdles": {key: str(value["label"]) for key, value in HURDLE_METADATA.items()},
            "tax_profiles": {key: str(value["label"]) for key, value in TAX_PROFILE_METADATA.items()},
        },
    }
    return clean_json(payload)
