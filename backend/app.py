import asyncio
import logging
import os
import sqlite3
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field, field_validator

from backend.serialization import StrictModel, clean_json
from backend.state import SESSION_COOKIE, release_session, replace_selection, resolve_session, session
from utils import db
from utils.market_data import get_listed_stocks


logger = logging.getLogger(__name__)
app = FastAPI(title="B3Lab", version="1.0.0")
_capacity = asyncio.Semaphore(2)


def allowed_origin(origin: str | None) -> bool:
    configured = os.environ.get("B3LAB_ORIGIN", "http://localhost:3000").rstrip("/")
    if origin == configured:
        return True
    if os.environ.get("B3LAB_ENV") != "production" and origin:
        parsed = urlsplit(origin)
        return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"} and parsed.port == 3000
    return False


@app.middleware("http")
async def owned_requests(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        if request.headers.get("x-b3lab-request") != "1" or not allowed_origin(request.headers.get("origin")):
            return JSONResponse({"detail": "Origem da solicitação não autorizada."}, status_code=403)
    acquired = False
    ctx = None
    try:
        ctx, issued = await asyncio.to_thread(resolve_session, request.cookies.get(SESSION_COOKIE))
        request.state.session = ctx
        expensive = request.url.path.endswith(("/analyze", "/run", "/restore", "/stress")) or request.url.path == "/api/news/refresh"
        if request.method == "POST" and expensive:
            try:
                await asyncio.wait_for(_capacity.acquire(), timeout=1)
                acquired = True
            except TimeoutError:
                return JSONResponse({"detail": "Duas análises já estão em execução. Tente novamente após a conclusão."}, status_code=503)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        if issued:
            response.set_cookie(SESSION_COOKIE, issued, httponly=True, samesite="lax", secure=os.environ.get("B3LAB_ENV") == "production", path="/")
        return response
    except sqlite3.Error:
        logger.exception("Session persistence failed")
        return JSONResponse({"detail": "Não foi possível acessar os dados desta sessão."}, status_code=503)
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    finally:
        if acquired:
            _capacity.release()
        if ctx is not None:
            release_session(ctx)


VALIDATION_MESSAGES = {
    "missing": "Campo obrigatório ausente.",
    "too_long": "A lista excede o limite de itens permitido.",
    "string_too_long": "Texto longo demais.",
    "string_too_short": "Texto curto demais.",
    "greater_than_equal": "O valor informado está abaixo do mínimo permitido.",
    "int_parsing": "Informe um número inteiro válido.",
    "int_type": "Informe um número inteiro válido.",
    "list_type": "Envie uma lista de ativos.",
    "json_invalid": "Corpo da solicitação inválido.",
}


def portuguese_validation_message(error: dict) -> str:
    context = error.get("ctx") or {}
    if error.get("type") == "value_error":
        return str(context.get("error", "Confira os dados informados."))
    return VALIDATION_MESSAGES.get(error.get("type"), "Confira os dados informados.")


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    messages = dict.fromkeys(portuguese_validation_message(error) for error in exc.errors())
    return JSONResponse({"detail": " ".join(messages)}, status_code=422)


@app.get("/health")
def health():
    return {"status": "ok"}


def session_payload(ctx):
    portfolio = ctx.portfolio
    return {
        "selection": ctx.selection,
        "revision": ctx.revision,
        "watchlist": db.wl_get(ctx.uid),
        "portfolio_ready": portfolio is not None and portfolio.revision == ctx.revision,
        "simulation_ready": portfolio is not None and portfolio.revision == ctx.revision and not portfolio.returns.empty,
        "portfolio_id": portfolio.id if portfolio else None,
        "restored": ctx.restored,
        "restoration": ctx.restoration if ctx.restored else None,
    }


@app.get("/api/session")
def read_session(request: Request):
    ctx = session(request)
    with ctx.lock:
        return session_payload(ctx)


class SelectionRequest(StrictModel):
    tickers: list[str] = Field(max_length=20)
    revision: int = Field(ge=0)

    @field_validator("tickers")
    @classmethod
    def normalized_tickers(cls, values):
        normalized = [value.strip().upper().removesuffix(".SA") for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Selecione cada ativo apenas uma vez.")
        return normalized


@app.put("/api/session/selection")
def select_stocks(body: SelectionRequest, request: Request):
    universe = set(get_listed_stocks()["Ticker"])
    if any(ticker not in universe for ticker in body.tickers):
        raise HTTPException(422, "A seleção contém ativos fora da lista da B3.")
    ctx = session(request)
    replace_selection(ctx, body.tickers, body.revision)
    with ctx.lock:
        return session_payload(ctx)


@app.get("/api/watchlist")
def read_watchlist(request: Request):
    return {"watchlist": db.wl_get(session(request).uid)}


@app.put("/api/watchlist/{ticker}")
def add_favorite(ticker: str, request: Request):
    ticker = ticker.strip().upper().removesuffix(".SA")
    if ticker not in set(get_listed_stocks()["Ticker"]):
        raise HTTPException(422, "Ativo indisponível na lista da B3.")
    ctx = session(request)
    if not db.wl_add(ctx.uid, ticker):
        raise HTTPException(503, "Não foi possível salvar o favorito.")
    return {"watchlist": db.wl_get(ctx.uid)}


@app.delete("/api/watchlist/{ticker}")
def remove_favorite(ticker: str, request: Request):
    ctx = session(request)
    if not db.wl_remove(ctx.uid, ticker.strip().upper().removesuffix(".SA")):
        raise HTTPException(503, "Não foi possível remover o favorito.")
    return {"watchlist": db.wl_get(ctx.uid)}


@app.get("/api/portfolio")
def read_portfolio(request: Request):
    ctx = session(request)
    with ctx.lock:
        saved_tickers, saved_weights = db.portfolio_get(ctx.uid)
        portfolio = ctx.portfolio
        return clean_json({
            "id": portfolio.id if portfolio else None,
            "tickers": portfolio.tickers if portfolio else ctx.selection,
            "weights": portfolio.weights if portfolio else {},
            "mode": portfolio.mode if portfolio else None,
            "revision": ctx.revision,
            "saved": {"tickers": saved_tickers, "weights": saved_weights},
            "portfolio_ready": portfolio is not None,
            "simulation_ready": portfolio is not None and not portfolio.returns.empty,
        })


@app.delete("/api/portfolio")
def clear_portfolio(request: Request):
    ctx = session(request)
    with ctx.lock:
        if not db.portfolio_clear(ctx.uid):
            raise HTTPException(503, "Não foi possível limpar o portfólio salvo.")
        replace_selection(ctx, [], ctx.revision)
        ctx.portfolio = None
        ctx.artifacts.clear()
        ctx.restored = False
        return session_payload(ctx)


from backend import home, market, news, portfolio, screener, simulation

app.include_router(home.router)
app.include_router(home.catalog_router)
app.include_router(screener.router)
app.include_router(portfolio.router)
app.include_router(simulation.router)
app.include_router(news.router)
app.include_router(market.router)
