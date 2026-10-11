import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field

import pandas as pd
from fastapi import HTTPException, Request

from utils import db


@dataclass
class PortfolioSnapshot:
    id: str
    revision: int
    tickers: list[str]
    weights: dict[str, float]
    mode: str
    returns: pd.DataFrame
    prices: pd.DataFrame
    lookback_days: int
    result: dict


@dataclass
class SessionState:
    uid: str
    selection: list[str] = field(default_factory=list)
    revision: int = 0
    restored: bool = False
    restoration: dict[str, int] = field(default_factory=dict)
    lock: object = field(default_factory=threading.RLock)
    generations: dict[str, int] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)
    portfolio: PortfolioSnapshot | None = None
    model_retry_at: float = 0.0
    last_seen: float = field(default_factory=time.time)
    active_requests: int = 0


_states: dict[str, SessionState] = {}
_states_lock = threading.RLock()
SESSION_COOKIE = "b3lab_session"
_SESSION_LIFETIME = 30 * 86400
_ANALYSIS_IDLE = 2 * 3600
_MAX_ACTIVE_SESSIONS = 32


def session(request: Request) -> SessionState:
    return request.state.session


def resolve_session(token: str | None) -> tuple[SessionState, str | None]:
    now = time.time()
    digest = hashlib.sha256(token.encode()).hexdigest() if token and len(token) == 43 else None
    issued = None
    with db._conn() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS browser_sessions (token_hash TEXT PRIMARY KEY, uid TEXT NOT NULL, expires_at REAL NOT NULL)")
        row = connection.execute("SELECT uid FROM browser_sessions WHERE token_hash = ? AND expires_at > ?", (digest, now)).fetchone() if digest else None
        if row:
            uid = row[0]
            connection.execute("UPDATE browser_sessions SET expires_at = ? WHERE token_hash = ?", (now + _SESSION_LIFETIME, digest))
        else:
            issued = secrets.token_urlsafe(32)
            uid = secrets.token_hex(16)
            connection.execute("INSERT INTO browser_sessions VALUES (?, ?, ?)", (hashlib.sha256(issued.encode()).hexdigest(), uid, now + _SESSION_LIFETIME))
        connection.execute("DELETE FROM browser_sessions WHERE expires_at <= ?", (now,))
    with _states_lock:
        for owner in list(_states):
            if _states[owner].active_requests == 0 and now - _states[owner].last_seen > _ANALYSIS_IDLE:
                del _states[owner]
        state = _states.get(uid)
        if state is None:
            if len(_states) >= _MAX_ACTIVE_SESSIONS:
                idle = [key for key, value in _states.items() if value.active_requests == 0]
                if not idle:
                    raise HTTPException(503, "Todas as sessões estão ocupadas. Tente novamente após a conclusão das análises.")
                del _states[min(idle, key=lambda key: _states[key].last_seen)]
            tickers, _ = db.portfolio_get(uid)
            from utils.market_data import get_listed_stocks
            universe = set(get_listed_stocks()["Ticker"])
            candidates = list(dict.fromkeys(str(t).strip().upper().removesuffix(".SA") for t in tickers))
            valid = [ticker for ticker in candidates if ticker in universe]
            selection = valid[:20]
            state = SessionState(
                uid=uid, selection=selection, restored=bool(selection),
                restoration={"original_count": len(tickers), "invalid_count": len(candidates) - len(valid), "truncated_count": max(0, len(valid) - 20)},
            )
            _states[uid] = state
        state.last_seen = now
        state.active_requests += 1
    return state, issued


def release_session(ctx: SessionState):
    with _states_lock:
        ctx.active_requests -= 1


def begin(ctx: SessionState, domain: str, revision: int) -> int:
    with ctx.lock:
        if revision != ctx.revision:
            raise HTTPException(409, "A seleção mudou. Atualize a página e execute a análise novamente.")
        generation = ctx.generations.get(domain, 0) + 1
        ctx.generations[domain] = generation
        if domain == "portfolio":
            ctx.portfolio = None
            for dependent in ("portfolio", "simulation", "news"):
                ctx.artifacts.pop(dependent, None)
                if dependent != domain:
                    ctx.generations[dependent] = ctx.generations.get(dependent, 0) + 1
        return generation


def publish(ctx: SessionState, domain: str, revision: int, generation: int, value: object, portfolio_id: str | None = None):
    with ctx.lock:
        if revision != ctx.revision or generation != ctx.generations.get(domain):
            raise HTTPException(409, "Esta análise ficou desatualizada. Execute novamente para a seleção atual.")
        if portfolio_id is not None and (ctx.portfolio is None or ctx.portfolio.id != portfolio_id):
            raise HTTPException(409, "O portfólio mudou durante a análise. Execute novamente.")
        ctx.artifacts[domain] = value
        if domain == "portfolio":
            if not isinstance(value, PortfolioSnapshot):
                raise TypeError("Portfolio publication requires a PortfolioSnapshot")
            ctx.portfolio = value
            ctx.restored = False
        return value


def artifact(ctx: SessionState, domain: str, identifier: str):
    with ctx.lock:
        value = ctx.artifacts.get(domain)
        value_id = value.get("id") if isinstance(value, dict) else getattr(value, "id", None)
        if value is None or value_id != identifier:
            raise HTTPException(404, "Resultado indisponível nesta sessão. Execute a análise novamente.")
        return value


def require_portfolio(ctx: SessionState, returns: bool = False) -> PortfolioSnapshot:
    with ctx.lock:
        portfolio = ctx.portfolio
        if portfolio is None:
            raise HTTPException(409, "Carregue o portfólio antes de continuar.")
        if portfolio.revision != ctx.revision or portfolio.tickers != ctx.selection:
            raise HTTPException(409, "A seleção mudou. Atualize o portfólio antes de continuar.")
        if returns and portfolio.returns.empty:
            raise HTTPException(409, "Restaure ou carregue o histórico do portfólio antes de simular.")
        return portfolio


def replace_selection(ctx: SessionState, tickers: list[str], revision: int):
    with ctx.lock:
        if revision != ctx.revision:
            raise HTTPException(409, "A seleção foi alterada em outra página. Atualize antes de continuar.")
        if tickers != ctx.selection:
            ctx.selection = tickers
            ctx.revision += 1
            ctx.restored = False
            ctx.portfolio = None
            ctx.artifacts.clear()
            for domain in list(ctx.generations):
                ctx.generations[domain] += 1
