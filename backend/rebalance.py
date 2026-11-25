"""Monthly contribution API: import broker positions and plan a buy-only ticket."""

from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from backend.serialization import StrictModel, clean_json
from backend.state import require_portfolio, session
from utils.assets import normalize_ticker, provider_symbol
from utils.importers import parse_positions
from utils.market_data import get_listed_stocks
from utils.portfolio_data import finite_or_none, get_portfolio_trade_prices
from utils.rebalance import plan_buy_only_rebalance

router = APIRouter(prefix="/api/rebalance", tags=["rebalance"])


class ImportRequest(StrictModel):
    content: str = Field(min_length=1, max_length=200_000)


class PlanRequest(StrictModel):
    revision: int = Field(ge=0)
    portfolio_id: str = Field(min_length=1)
    contribution: float = Field(gt=0, le=10_000_000, allow_inf_nan=False)
    current_quantities: dict[str, float] = Field(default_factory=dict)
    allow_fractional_us: bool = False


@router.post("/import")
def import_positions(body: ImportRequest):
    try:
        return clean_json(parse_positions(body.content))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _trade_prices(tickers: list[str]) -> tuple[dict[str, float], str | None]:
    """Latest unadjusted purchase references keyed by canonical ticker."""
    if not tickers:
        return {}, None
    try:
        universe = set(get_listed_stocks()["Ticker"])
    except Exception:
        universe = None
    symbols = [provider_symbol(ticker, universe) for ticker in tickers]
    try:
        frame = get_portfolio_trade_prices(tuple(symbols))
    except Exception as exc:
        raise HTTPException(
            422,
            f"Não foi possível buscar os preços de negociação: {exc}. Tente novamente.",
        ) from exc
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=symbols[0])
    prices: dict[str, float] = {}
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        for symbol in symbols:
            if symbol not in frame.columns:
                continue
            values = pd.to_numeric(frame[symbol], errors="coerce").replace([float("inf"), float("-inf")], float("nan")).dropna()
            if values.empty:
                continue
            price = finite_or_none(values.iloc[-1])
            if price is not None and price > 0:
                prices[normalize_ticker(symbol)] = float(price)
    return prices, "Preços de compra não ajustados (Yahoo Finance e Tesouro Direto)"


@router.post("/plan")
def plan(body: PlanRequest, request: Request):
    ctx = session(request)
    portfolio = require_portfolio(ctx)
    if body.revision != portfolio.revision or body.portfolio_id != portfolio.id:
        raise HTTPException(409, "O portfólio mudou. Calcule a carteira novamente antes de planejar o aporte.")

    quantities: dict[str, float] = {}
    for raw, value in body.current_quantities.items():
        ticker = normalize_ticker(raw)
        if not ticker:
            raise HTTPException(422, "Informe tickers válidos para as quantidades atuais.")
        if not math.isfinite(value) or value < 0:
            raise HTTPException(422, "As quantidades atuais devem ser finitas e não negativas.")
        quantities[ticker] = quantities.get(ticker, 0.0) + value

    wanted = sorted(set(portfolio.tickers) | set(quantities))
    trade_prices, price_source = _trade_prices(wanted)

    try:
        result = plan_buy_only_rebalance(
            target_weights=portfolio.weights,
            current_holdings=quantities,
            trade_prices=trade_prices,
            contribution=body.contribution,
            allow_fractional_us=body.allow_fractional_us,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    result["portfolio_id"] = portfolio.id
    result["revision"] = portfolio.revision
    result["price_source"] = price_source
    return clean_json(result)
