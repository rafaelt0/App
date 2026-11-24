"""Read-only market strip and delayed quotes, no owner state."""
from fastapi import APIRouter, HTTPException

from backend.serialization import clean_json
from utils.assets import normalize_ticker
from utils.market_data import monitor_quotes
from utils.market_strip import market_strip

router = APIRouter(prefix="/api/market", tags=["market"])
_MAX_MONITOR_QUOTES = 100


@router.get("/strip")
def read_market_strip():
    return clean_json(market_strip())


@router.get("/quotes")
def read_market_quotes(tickers: str = ""):
    """Delayed last price and daily change for the home monitor's quick list."""
    requested = []
    for part in tickers.split(","):
        ticker = normalize_ticker(part)
        if ticker and ticker not in requested:
            requested.append(ticker)
    if not requested:
        raise HTTPException(422, "Informe os ativos da cotação.")
    if len(requested) > _MAX_MONITOR_QUOTES:
        raise HTTPException(422, "A consulta aceita no máximo 100 ativos.")
    return clean_json(monitor_quotes(tuple(requested)))
