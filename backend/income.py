"""Owner-bound proventos calendar endpoint; financial policy stays in utils.income."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from backend.serialization import StrictModel, clean_json, figure
from backend.state import require_portfolio, session
from utils.income import build_income_calendar

router = APIRouter(prefix="/api/income", tags=["income"])


class IncomeCalendarRequest(StrictModel):
    revision: int = Field(ge=0)
    portfolio_id: str = Field(min_length=1)
    capital: float = Field(default=10000, ge=100, le=10_000_000, allow_inf_nan=False)
    cost_basis_brl: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    jcp_share: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)


@router.post("/calendar")
def calendar(payload: IncomeCalendarRequest, request: Request):
    ctx = session(request)
    portfolio = require_portfolio(ctx)
    if payload.portfolio_id != portfolio.id:
        raise HTTPException(409, "O portfólio mudou. Carregue a carteira e atualize o calendário.")
    if payload.revision != portfolio.revision:
        raise HTTPException(409, "A carteira foi recalculada. Atualize o calendário de proventos.")
    try:
        result = build_income_calendar(
            portfolio.tickers,
            portfolio.weights,
            payload.capital,
            cost_basis_brl=payload.cost_basis_brl,
            jcp_share=payload.jcp_share,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Não foi possível montar o calendário de proventos: {exc}") from exc
    result.update({"revision": portfolio.revision, "portfolio_id": portfolio.id})
    if result.get("chart") is not None:
        result["chart"] = figure(result["chart"])
    return clean_json(result)
