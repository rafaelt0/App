"""Read-only market strip: delayed quotes and BCB macro indicators, no owner state."""
from fastapi import APIRouter

from backend.serialization import clean_json
from utils.market_strip import market_strip

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/strip")
def read_market_strip():
    return clean_json(market_strip())
