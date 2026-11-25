"""Owned cross-asset macro shock endpoint; calculations stay in utils.macro_shock."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from backend.serialization import StrictModel, clean_json, figure
from backend.state import require_portfolio, session
from utils.macro_shock import (
    MACRO_PRESETS,
    MacroShockInput,
    simulate_macro_shock,
)

router = APIRouter(prefix="/api/stress", tags=["stress"])


class MacroShockRequest(StrictModel):
    revision: int = Field(ge=0)
    portfolio_id: str = Field(min_length=1)
    capital: float = Field(default=10000, ge=100, le=10_000_000, allow_inf_nan=False)
    selic_bps: float = Field(default=0, ge=-600, le=600, allow_inf_nan=False)
    ipca_bps: float = Field(default=0, ge=-500, le=500, allow_inf_nan=False)
    usd_brl_pct: float = Field(default=0, ge=-40, le=60, allow_inf_nan=False)
    us10y_bps: float = Field(default=0, ge=-300, le=300, allow_inf_nan=False)
    sp500_pct: float = Field(default=0, ge=-50, le=50, allow_inf_nan=False)
    dxy_pct: float = Field(default=0, ge=-30, le=30, allow_inf_nan=False)


@router.get("/presets")
def presets():
    """The built-in Brazil and global scenarios; no portfolio is required."""
    return clean_json(list(MACRO_PRESETS))


@router.post("/macro")
def macro(payload: MacroShockRequest, request: Request):
    ctx = session(request)
    with ctx.lock:
        portfolio = require_portfolio(ctx, returns=True)
        if portfolio.id != payload.portfolio_id:
            raise HTTPException(409, "A carteira mudou. Carregue a análise atual antes de simular.")
        if portfolio.revision != payload.revision:
            raise HTTPException(409, "A seleção mudou. Atualize a página e tente novamente.")
    shock = MacroShockInput(
        selic_bps=payload.selic_bps,
        ipca_bps=payload.ipca_bps,
        usd_brl_pct=payload.usd_brl_pct,
        us10y_bps=payload.us10y_bps,
        sp500_pct=payload.sp500_pct,
        dxy_pct=payload.dxy_pct,
    )
    try:
        result = simulate_macro_shock(
            portfolio.returns,
            portfolio.weights,
            payload.capital,
            shock,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Não foi possível simular o choque macroeconômico: {exc}") from exc
    chart = result.pop("chart", None)
    result["chart"] = figure(chart) if chart is not None else None
    result.update({"portfolio_id": portfolio.id, "revision": payload.revision})
    return clean_json(result)
