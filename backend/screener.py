"""Full-market screening retains source fractions; only presentation/export scales percentages."""
import datetime as dt
import uuid
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, model_validator

from backend.serialization import StrictModel, clean_json, csv_response
from backend.state import artifact, begin, publish, session
from utils.assets import ASSET_CLASSES, ASSET_CLASS_LABELS, ASSET_INDICATORS, EXTRA_ASSETS_BY_TICKER, catalog_screener_rows
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
SORTABLE_FIELDS = tuple(key for key in COLUMN_MAP if key != "setor")
_HOME_TO_SCREEN = {"ev_ebitda": "evebitda", "growth": "c5y", "net_margin": "mrgliq", "ebit_margin": "mrgebit"}
SCREEN_FIELDS = {category: {"cotacao", "liq2m", "setor"} | {_HOME_TO_SCREEN.get(key, key) for key in indicators}
                 for category, indicators in ASSET_INDICATORS.items()}
SCREEN_FIELDS["acao_br"] = set(COLUMN_MAP)
SCREEN_FIELDS["b3"] = SCREEN_FIELDS["all"] = SCREEN_FIELDS["acao_br"]
CRITERION_FIELDS = {"pl_min": "pl", "pl_max": "pl", "pl_min_exclusive": "pl", "roe_min": "roe",
                    "dy_min": "dy", "c5y_min": "c5y", "liq2m_min": "liq2m"}
# Plain-language definitions for each column; shown in the Screener glossary, not used in calculations.
GLOSSARY = {
    "cotacao": "Último preço negociado da ação, em reais.",
    "pl": "Preço dividido pelo lucro por ação dos últimos 12 meses. Negativo indica prejuízo.",
    "pvp": "Preço dividido pelo valor patrimonial por ação. Abaixo de 1 pode indicar desconto contábil.",
    "dy": "Proventos pagos nos últimos 12 meses divididos pelo preço atual. Retorno passado; não garante proventos futuros.",
    "roe": "Lucro dos últimos 12 meses dividido pelo patrimônio líquido.",
    "roic": "Retorno sobre o capital investido, somando patrimônio e dívida.",
    "evebitda": "Valor da empresa, com dívida líquida, dividido pelo EBITDA dos últimos 12 meses.",
    "evebit": "Valor da empresa, com dívida líquida, dividido pelo EBIT dos últimos 12 meses.",
    "mrgebit": "EBIT dividido pela receita líquida.",
    "mrgliq": "Lucro líquido dividido pela receita líquida.",
    "liqc": "Ativo circulante dividido pelo passivo circulante. Abaixo de 1 indica mais obrigações de curto prazo que ativos de curto prazo.",
    "liq2m": "Média do volume negociado por dia nos últimos 2 meses, em reais.",
    "divbpatr": "Dívida líquida dividida pelo patrimônio líquido. Negativo indica caixa líquido.",
    "c5y": "Taxa anual composta de crescimento da receita nos últimos 5 anos.",
    "patrliq": "Patrimônio líquido contábil, em reais.",
}
# Fundamentus prints 0.00 for these when a bank or insurer has no applicable value; a real 0 is not expected here.
ZERO_MEANS_MISSING = ("psr", "evebit", "evebitda", "mrgebit", "mrgliq", "roic", "liqc", "divbpatr")
DEFAULT_COLUMNS = ["cotacao", "pl", "pvp", "dy", "roe", "roic", "evebitda", "liq2m"]
DESCRIPTIONS = {
    "Explorar B3": "Liquidez média em 2 meses de pelo menos R$ 1 milhão.",
    "Lucro a preço moderado": "P/L > 0 e ≤ 15, ROE ≥ 12% e liquidez ≥ R$ 1 milhão.",
    "Renda atual (12m)": "DY dos últimos 12 meses ≥ 6%, 0 < P/L ≤ 15, ROE ≥ 10% e liquidez ≥ R$ 1 milhão. O DY pode incluir proventos extraordinários; o filtro não verifica payout, fluxo de caixa, dívida nem histórico de dividendos. O teto de P/L e o ROE evitam yield inflado por queda de preço ou lucro fraco; DY passado não garante dividendos futuros.",
    "Valor descontado": "0 < P/L ≤ 8, ROE ≥ 12% e liquidez ≥ R$ 2 milhões. Empresas cíclicas costumam aparecer com P/L baixo no pico do ciclo; confira a sustentabilidade do lucro.",
    "Qualidade rentável": "ROE ≥ 18%, 0 < P/L ≤ 20 e liquidez ≥ R$ 2 milhões. ROE alto pode refletir alavancagem.",
    "Crescimento a preço razoável": "Receita em 5 anos ≥ 10%, ROE ≥ 15%, 0 < P/L ≤ 20 e liquidez ≥ R$ 2 milhões. Mede crescimento de receita, não de lucro.",
    "Grandes e líquidas": "Liquidez média em 2 meses ≥ R$ 20 milhões, ROE ≥ 10% e 0 < P/L ≤ 25: papéis de grande negociação com lucro positivo.",
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
    field: str = "liq2m"
    direction: Literal["asc", "desc"] = "desc"

    @model_validator(mode="after")
    def supported(self):
        if self.field not in SORTABLE_FIELDS:
            raise ValueError("Ordenação não suportada.")
        return self


class Query(StrictModel):
    criteria: Criteria = Field(default_factory=Criteria)
    sort: Sort = Field(default_factory=Sort)
    asset_class: Literal["b3", "all", "acao_br", "fii", "acao_us", "reit", "titulo", "etf", "outros"] = "b3"


def columns():
    return [{"key": key, "label": label,
             "unit": "fraction" if key in PERCENT_FIELDS else "money" if key in ("cotacao", "liq2m", "patrliq") else "number" if key == "setor" else "multiple" if key in ("pl", "pvp", "evebitda", "evebit") else "ratio",
             "digits": 0 if key in ("liq2m", "patrliq") else 2, "help": GLOSSARY.get(key)} for key, label in COLUMN_MAP.items()]


def normalize_market(raw):
    renames = dict(RENAMES)
    for column in raw.columns:
        normalized = str(column).lower().replace(" ", "").replace(".", "")
        if "patrim" in normalized and any(term in normalized for term in ("dív", "div")) and any(term in normalized for term in ("líq", "liq")):
            renames[column] = "divbpatr"
    result = raw.rename(columns=renames).copy()
    for column in result.columns:
        result[column] = clean_numeric_column(result[column])
    for column in ZERO_MEANS_MISSING:
        if column in result:
            result[column] = result[column].mask(result[column] == 0)
    # Negative book equity makes P/VP and ROE meaningless (a loss can show as a positive ROE).
    if "pvp" in result:
        negative_equity = result["pvp"] < 0
        result["pvp"] = result["pvp"].mask(negative_equity)
        if "roe" in result:
            result["roe"] = result["roe"].mask(negative_equity)
    result.attrs.update(raw.attrs)
    return result


def pt(value, decimals=None):
    return (f"{value:g}" if decimals is None else f"{value:,.{decimals}f}").translate(str.maketrans(".,", ",."))


def active_summary(criteria, multi=False):
    result = []
    if "liq2m_min" in criteria:
        result.append(f"Liquidez 2m ≥ {'' if multi else 'R$ '}{pt(criteria['liq2m_min'], 0)}")
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


def screen(raw, body, *, source="Fundamentus", normalized=False):
    if raw is None or raw.empty:
        raise HTTPException(503, "A fonte não retornou dados; isso é diferente de uma busca sem resultados.")
    frame = raw if normalized else normalize_market(raw)
    multi = body.asset_class not in {"b3", "acao_br"}
    criteria = body.criteria.active()
    try:
        matched = filter_stocks(frame, criteria)
    except ValueError as exc:
        raise HTTPException(422, f"Os dados recebidos não atendem ao esquema do filtro selecionado: {exc}") from exc
    if body.sort.field not in matched:
        raise HTTPException(422, f"Não é possível ordenar: a coluna {body.sort.field} não está disponível.")
    by_amount = multi and body.sort.field in {"cotacao", "liq2m", "patrliq"}
    matched = matched.sort_values(["currency", body.sort.field] if by_amount else body.sort.field,
                                  ascending=[True, body.sort.direction == "asc"] if by_amount else body.sort.direction == "asc",
                                  na_position="last", kind="stable")
    available = [column for column in columns() if column["key"] in matched]
    labels = dict(COLUMN_MAP)
    if multi:
        labels.update(cotacao="Cotação", liq2m="Liquidez 2m", patrliq="Patrim. Líq.", dy="Proventos / preço (12m) (%)")
        for column in available:
            column["label"] = labels[column["key"]]
    display_frame = matched[[column["key"] for column in available]]
    if multi:
        display_frame = display_frame.assign(currency=matched["currency"], asset_class=matched["asset_class"])
        labels.update(currency="Moeda", asset_class="Classe")
    rows = [{"ticker": str(ticker), **row} for ticker, row in zip(display_frame.index, display_frame.to_dict("records"))]
    result = {"total_count": len(frame), "matched_count": len(matched), "criteria": criteria,
              "active_filters": active_summary(criteria, multi), "source": source, **source_time(raw.attrs.get("fetched_at")),
              "metrics": {"mean_dy": matched["dy"].mean() * 100 if "dy" in matched and not matched.empty else None,
                          "mean_roe": matched["roe"].mean() * 100 if "roe" in matched and not matched.empty else None},
              "available_columns": available, "default_columns": [key for key in DEFAULT_COLUMNS if key in matched], "rows": rows}
    return clean_json(result), prepare_export(display_frame, labels)


@router.get("/config")
def config():
    asset_classes = [
        {"key": "b3", "label": "Ações B3 (Fundamentus)"},
        {"key": "all", "label": "Todos os ativos (B3, EUA, FIIs, REITs, Títulos e ETFs)"},
        *[{"key": item["key"], "label": item["label"]} for item in ASSET_CLASSES if item["key"] != "acao_br"],
    ]
    for item in asset_classes:
        item["fields"] = sorted(SCREEN_FIELDS[item["key"]])
    return {"presets": [{"name": name, "description": DESCRIPTIONS[name], "criteria": criteria} for name, criteria in PRESET_FILTERS.items()],
            "sorts": SORTS, "columns": columns(), "default_columns": DEFAULT_COLUMNS,
            "asset_classes": asset_classes,
            "bounds": {"pl": [-50, 200, .5], "roe": [0, 80, 1], "growth": [0, 100, 1], "dy": [0, 30, .5], "liquidity": [0, 100_000_000, 100_000]}}


@router.post("/query")
def query(body: Query, request: Request):
    ctx = session(request)
    with ctx.lock:
        revision = ctx.revision
    generation = begin(ctx, "screener", revision)
    try:
        fields = SCREEN_FIELDS[body.asset_class]
        if any(CRITERION_FIELDS[key] not in fields for key in body.criteria.active()):
            raise HTTPException(422, "Um ou mais critérios não se aplicam à classe de ativo selecionada.")
        if body.sort.field not in fields:
            raise HTTPException(422, "A ordenação não se aplica à classe de ativo selecionada.")

        b3 = pd.DataFrame()
        if body.asset_class in {"b3", "all", "acao_br"}:
            try:
                raw = get_full_market_data()
                if raw is not None:
                    b3 = normalize_market(raw.loc[~raw.index.isin(EXTRA_ASSETS_BY_TICKER)])
            except Exception:
                if body.asset_class != "all":
                    raise
        if body.asset_class in {"b3", "acao_br"}:
            result, exported = screen(b3, body, normalized=True)
        else:
            reference = catalog_screener_rows(body.asset_class, SCREEN_FIELDS)
            frame = pd.concat([b3.assign(currency="BRL", asset_class="acao_br"), reference]) if not b3.empty else reference
            if body.asset_class == "all":
                frame = frame.reindex(columns=frame.columns.union(sorted(fields), sort=False))
            source = f"Catálogo de referência · {ASSET_CLASS_LABELS.get(body.asset_class, 'Multi-Ativos')} (sem cotação ao vivo)"
            if not b3.empty:
                source += " + Fundamentus"
            result, exported = screen(frame, body, source=source, normalized=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Não foi possível carregar os dados do screener: {exc}") from exc
    result["id"] = uuid.uuid4().hex
    publish(ctx, "screener", revision, generation, {"id": result["id"], "result": result, "export": exported})
    return result


@router.get("/{id}/results.csv")
def results_csv(id: str, request: Request):
    saved = artifact(session(request), "screener", id)
    return csv_response(saved["export"], f"screener_b3_{dt.date.today():%Y%m%d}.csv")
