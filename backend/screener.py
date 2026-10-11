"""Full-market screening retains source fractions; only presentation/export scales percentages."""
import datetime as dt
import uuid
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, model_validator

from backend.serialization import StrictModel, clean_json, csv_response
from backend.state import artifact, begin, publish, session
from utils.market_data import clean_numeric_column, get_full_market_data
from utils.screener import PRESET_FILTERS, filter_stocks, prepare_export

router = APIRouter(prefix="/api/screener", tags=["screener"])
RENAMES = {"Cotação": "cotacao", "P/L": "pl", "P/VP": "pvp", "PSR": "psr", "Div.Yield": "dy",
           "P/Ativo": "pa", "P/Cap.Giro": "pcg", "P/EBIT": "pebit", "P/Ativ Circ.Liq": "pacl",
           "EV/EBIT": "evebit", "EV/EBITDA": "evebitda", "Mrg Ebit": "mrgebit", "Mrg. Líq.": "mrgliq",
           "ROIC": "roic", "ROE": "roe", "Liq. Corr.": "liqc", "Liq.2meses": "liq2m",
           "Patrim. Líq": "patrliq", "Dív.Líq/ Patrim.": "divbpatr", "Cresc. Rec.5a": "c5y"}
COLUMN_MAP = {"cotacao": "Cotação (R$)", "pl": "P/L", "pvp": "P/VP", "dy": "Div. Yield (%)", "roe": "ROE (%)",
              "roic": "ROIC (%)", "evebitda": "EV/EBITDA", "evebit": "EV/EBIT", "mrgebit": "Mrg. EBIT (%)",
              "mrgliq": "Mrg. Líq. (%)", "liqc": "Liq. Corrente", "liq2m": "Liq. 2m (R$)",
              "divbpatr": "Dív. Líq./Patrim.", "c5y": "Cresc. Rec. 5a (%)", "patrliq": "Patrim. Líq. (R$)", "setor": "Setor"}
PERCENT_FIELDS = {"dy", "roe", "roic", "mrgebit", "mrgliq", "c5y"}
DEFAULT_COLUMNS = ["cotacao", "pl", "pvp", "dy", "roe", "roic", "evebitda", "liq2m"]
DESCRIPTIONS = {
    "Explorar B3": "Liquidez média em 2 meses de pelo menos R$ 1 milhão.",
    "Lucro a preço moderado": "P/L > 0 e ≤ 15, ROE ≥ 12% e liquidez ≥ R$ 1 milhão.",
    "Renda atual": "Dividend yield ≥ 5%, ROE ≥ 10% e liquidez ≥ R$ 1 milhão. DY passado não garante dividendos futuros.",
    "Qualidade rentável": "ROE ≥ 18%, 0 < P/L ≤ 20 e liquidez ≥ R$ 2 milhões.",
    "Alta liquidez": "Liquidez média em 2 meses ≥ R$ 10 milhões; sem outros critérios.",
    "Crescimento com lucro": "Receita em 5 anos ≥ 10%, ROE ≥ 12% e liquidez ≥ R$ 1 milhão.",
    "Personalizado": "Escolha explicitamente quais critérios aplicar.",
}
SORTS = [{"field": "liq2m", "direction": "desc", "label": "Liquidez 2m"},
         {"field": "pl", "direction": "asc", "label": "P/L (menor primeiro)"},
         {"field": "roe", "direction": "desc", "label": "ROE (maior primeiro)"},
         {"field": "dy", "direction": "desc", "label": "DY (maior primeiro)"}]


class Criteria(StrictModel):
    pl_min: float | None = Field(default=None, ge=-50, le=200)
    pl_max: float | None = Field(default=None, ge=-50, le=200)
    pl_min_exclusive: bool = False
    roe_min: float | None = Field(default=None, ge=0, le=.8)
    dy_min: float | None = Field(default=None, ge=0, le=.3)
    c5y_min: float | None = Field(default=None, ge=0, le=1)
    liq2m_min: float | None = Field(default=None, ge=0, le=100_000_000)

    @model_validator(mode="after")
    def valid_range(self):
        if self.pl_min is not None and self.pl_max is not None and self.pl_min > self.pl_max:
            raise ValueError("O P/L mínimo não pode ser maior que o P/L máximo.")
        if self.pl_min_exclusive and self.pl_min is None:
            raise ValueError("P/L exclusivo requer um mínimo ativo.")
        return self

    def active(self):
        values = self.model_dump(exclude_none=True)
        if self.pl_min is None:
            values.pop("pl_min_exclusive", None)
        return values


class Sort(StrictModel):
    field: Literal["liq2m", "pl", "roe", "dy"] = "liq2m"
    direction: Literal["asc", "desc"] = "desc"

    @model_validator(mode="after")
    def supported(self):
        if (self.field == "pl") != (self.direction == "asc"):
            raise ValueError("Ordenação não suportada.")
        return self


class Query(StrictModel):
    criteria: Criteria = Field(default_factory=Criteria)
    sort: Sort = Field(default_factory=Sort)


def columns():
    return [{"key": key, "label": label,
             "unit": "fraction" if key in PERCENT_FIELDS else "money" if key in ("cotacao", "liq2m", "patrliq") else "number" if key == "setor" else "multiple" if key in ("pl", "pvp", "evebitda", "evebit") else "ratio",
             "digits": 0 if key in ("liq2m", "patrliq") else 2} for key, label in COLUMN_MAP.items()]


def normalize_market(raw):
    renames = dict(RENAMES)
    for column in raw.columns:
        normalized = str(column).lower().replace(" ", "").replace(".", "")
        if "patrim" in normalized and any(term in normalized for term in ("dív", "div")) and any(term in normalized for term in ("líq", "liq")):
            renames[column] = "divbpatr"
    result = raw.rename(columns=renames).copy()
    for column in result.columns:
        result[column] = clean_numeric_column(result[column])
    result.attrs.update(raw.attrs)
    return result


def pt(value, decimals=None):
    return (f"{value:g}" if decimals is None else f"{value:,.{decimals}f}").translate(str.maketrans(".,", ",."))


def active_summary(criteria):
    result = []
    if "liq2m_min" in criteria:
        result.append(f"Liquidez 2m ≥ R$ {pt(criteria['liq2m_min'], 0)}")
    if "pl_min" in criteria:
        result.append(f"P/L {'>' if criteria.get('pl_min_exclusive') else '≥'} {pt(criteria['pl_min'])}" + (f" e ≤ {pt(criteria['pl_max'])}" if "pl_max" in criteria else ""))
    elif "pl_max" in criteria:
        result.append(f"P/L ≤ {pt(criteria['pl_max'])}")
    for key, label in [("roe_min", "ROE"), ("dy_min", "DY"), ("c5y_min", "Cresc. Rec. 5a")]:
        if key in criteria:
            result.append(f"{label} ≥ {pt(criteria[key] * 100)}%")
    return result


def source_time(value):
    try:
        parsed = dt.datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return {"fetched_at": None, "age_minutes": None}
        age = max(0, int((dt.datetime.now(dt.timezone.utc) - parsed.astimezone(dt.timezone.utc)).total_seconds() // 60))
        return {"fetched_at": parsed.isoformat(), "age_minutes": age}
    except (ValueError, TypeError):
        return {"fetched_at": None, "age_minutes": None}


def screen(raw, body):
    if raw is None or raw.empty:
        raise HTTPException(503, "A Fundamentus não retornou dados; isso é diferente de uma busca sem resultados.")
    frame = normalize_market(raw)
    criteria = body.criteria.active()
    try:
        matched = filter_stocks(frame, criteria)
    except ValueError as exc:
        raise HTTPException(422, f"Os dados recebidos não atendem ao esquema do filtro selecionado: {exc}") from exc
    if body.sort.field not in matched:
        raise HTTPException(422, f"Não é possível ordenar: a coluna {body.sort.field} não está disponível.")
    matched = matched.sort_values(body.sort.field, ascending=body.sort.direction == "asc", na_position="last", kind="stable")
    available = [column for column in columns() if column["key"] in matched]
    display_frame = matched[[column["key"] for column in available]]
    rows = [{"ticker": str(ticker), **row} for ticker, row in zip(display_frame.index, display_frame.to_dict("records"))]
    result = {"total_count": len(frame), "matched_count": len(matched), "criteria": criteria,
              "active_filters": active_summary(criteria), "source": "Fundamentus", **source_time(raw.attrs.get("fetched_at")),
              "metrics": {"mean_dy": matched["dy"].mean() * 100 if "dy" in matched and not matched.empty else None,
                          "mean_roe": matched["roe"].mean() * 100 if "roe" in matched and not matched.empty else None},
              "available_columns": available, "default_columns": [key for key in DEFAULT_COLUMNS if key in matched], "rows": rows}
    return clean_json(result), prepare_export(matched, COLUMN_MAP)


@router.get("/config")
def config():
    return {"presets": [{"name": name, "description": DESCRIPTIONS[name], "criteria": criteria} for name, criteria in PRESET_FILTERS.items()],
            "sorts": SORTS, "columns": columns(), "default_columns": DEFAULT_COLUMNS,
            "bounds": {"pl": [-50, 200, .5], "roe": [0, 80, 1], "growth": [0, 100, 1], "dy": [0, 30, .5], "liquidity": [0, 100_000_000, 100_000]}}


@router.post("/query")
def query(body: Query, request: Request):
    ctx = session(request)
    with ctx.lock:
        revision = ctx.revision
    generation = begin(ctx, "screener", revision)
    try:
        result, exported = screen(get_full_market_data(), body)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Não foi possível carregar os dados da Fundamentus: {exc}") from exc
    result["id"] = uuid.uuid4().hex
    publish(ctx, "screener", revision, generation, {"id": result["id"], "result": result, "export": exported})
    return result


@router.get("/{id}/results.csv")
def results_csv(id: str, request: Request):
    saved = artifact(session(request), "screener", id)
    return csv_response(saved["export"], f"screener_b3_{dt.date.today():%Y%m%d}.csv")
