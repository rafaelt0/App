"""Full-market screening retains source fractions; only presentation/export scales percentages."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import logging
import uuid
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, model_validator

from backend.serialization import StrictModel, clean_json, csv_response
from backend.state import artifact, begin, publish, session
from utils.assets import ASSET_CLASSES, ASSET_INDICATORS, EXTRA_ASSETS_BY_TICKER, catalog_screener_rows
from utils.market_data import (clean_numeric_column, get_fii_market_data, get_full_market_data, get_us_market_data,
                               get_yahoo_quotes)
from utils.screener import PRESET_FILTERS, filter_stocks, prepare_export

router = APIRouter(prefix="/api/screener", tags=["screener"])
logger = logging.getLogger(__name__)
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
# Per-row provenance travels with each result (and the CSV) so mixed sources keep their own labels.
PROVENANCE_LABELS = {"currency": "Moeda", "asset_class": "Classe", "source": "Fonte",
                     "quote_date": "Data da cotação", "fetched_at": "Coleta"}
EXPORT_LABELS = {**COLUMN_MAP, **PROVENANCE_LABELS}
REFERENCE_SOURCE = "Catálogo de referência · sem cotação ao vivo"
LIVE_REFERENCE_SOURCE = "Yahoo Finance · cotação com atraso; demais indicadores de referência"
FII_REFERENCE_SOURCE = "Catálogo de referência · FIIs (sem cotação ao vivo)"
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


def _provenance(frame, source, *, asset_class=None, currency=None, fetched_at=None, quote_date=None):
    """Attach per-row provenance without touching any financial value or filter input."""
    result = frame.copy()
    if asset_class is not None:
        result["asset_class"] = asset_class
    if currency is not None:
        result["currency"] = currency
    result["source"] = source
    result["quote_date"] = quote_date
    result["fetched_at"] = fetched_at
    result.attrs = dict(frame.attrs)
    return result


def _ensure_provenance(frame, body, source):
    """Guarantee provenance columns for callers that build frames outside the loaders."""
    defaults = {"source": source, "quote_date": None, "fetched_at": frame.attrs.get("fetched_at")}
    if body.asset_class in {"b3", "acao_br"}:
        defaults.update(asset_class="acao_br", currency="BRL")
    elif body.asset_class in {"fii", "etf", "titulo"}:
        defaults.update(asset_class=body.asset_class, currency="BRL")
    elif body.asset_class in {"acao_us", "reit", "outros"}:
        defaults.update(asset_class=body.asset_class, currency="USD")
    result = frame.copy()
    for key, value in defaults.items():
        if key not in result:
            result[key] = value
    result.attrs = dict(frame.attrs)
    return result


def _with_live_quotes(reference):
    """Price catalog instruments with delayed Yahoo quotes, keeping reference indicators labelled."""
    symbols = {ticker: EXTRA_ASSETS_BY_TICKER[ticker]["yahoo_symbol"]
               for ticker in reference.index
               if EXTRA_ASSETS_BY_TICKER.get(ticker, {}).get("yahoo_symbol")}
    try:
        quotes = get_yahoo_quotes(tuple(sorted(set(symbols.values())))) if symbols else {}
    except Exception:
        logger.warning("Yahoo Finance quotes unavailable for catalog screener rows", exc_info=True)
        quotes = {}
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    result = reference.copy()
    sources, dates, stamps = [], [], []
    live = 0
    for ticker in result.index:
        quote = quotes.get(symbols.get(ticker))
        if quote:
            live += 1
            result.at[ticker, "cotacao"] = quote["price"]
            result.at[ticker, "currency"] = quote["currency"]
            sources.append(LIVE_REFERENCE_SOURCE)
            dates.append(quote.get("last_quote_date"))
            stamps.append(fetched_at)
        else:
            sources.append(REFERENCE_SOURCE)
            dates.append(None)
            stamps.append(None)
    result["source"] = sources
    result["quote_date"] = dates
    result["fetched_at"] = stamps
    result.attrs = dict(reference.attrs)
    result.attrs["live_quotes"] = live
    return result


def _catalog_source(frame):
    live = int(frame.attrs.get("live_quotes") or 0) if "source" in frame else 0
    if live:
        return f"Yahoo Finance · cotação com atraso e catálogo de referência ({live} de {len(frame)} com cotação)"
    return "Catálogo de referência · sem cotação ao vivo"


def _merge_reference_funds(live):
    """Keep catalog FIIs absent from the live Fundamentus table, clearly marked as reference."""
    reference = catalog_screener_rows("fii", SCREEN_FIELDS)
    if reference.empty:
        return live, 0
    missing = reference.index.difference(live.index)
    if missing.empty:
        return live, 0
    extra = _provenance(reference.loc[missing], FII_REFERENCE_SOURCE, asset_class="fii", currency="BRL")
    combined = pd.concat([live, extra])
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.attrs["fetched_at"] = live.attrs.get("fetched_at")
    return combined, len(missing)


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
    matched = _ensure_provenance(matched, body, source)
    if body.sort.field not in matched:
        raise HTTPException(422, f"Não é possível ordenar: a coluna {body.sort.field} não está disponível.")
    by_amount = multi and body.sort.field in {"cotacao", "liq2m", "patrliq"}
    matched = matched.sort_values(["currency", body.sort.field] if by_amount else body.sort.field,
                                  ascending=[True, body.sort.direction == "asc"] if by_amount else body.sort.direction == "asc",
                                  na_position="last", kind="stable")
    available = [column for column in columns() if column["key"] in matched]
    labels = dict(EXPORT_LABELS)
    if multi:
        labels.update(cotacao="Cotação", liq2m="Liquidez 2m", patrliq="Patrim. Líq.", dy="Proventos / preço (12m) (%)")
        for column in available:
            column["label"] = labels[column["key"]]
    meta_keys = [key for key in PROVENANCE_LABELS if key in matched]
    display_frame = matched[[column["key"] for column in available] + meta_keys]
    rows = [{"ticker": str(ticker), **row} for ticker, row in zip(display_frame.index, display_frame.to_dict("records"))]
    result = {"total_count": len(frame), "matched_count": len(matched), "criteria": criteria,
              "active_filters": active_summary(criteria, multi), "source": source, **source_time(frame.attrs.get("fetched_at")),
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
        fii = pd.DataFrame()
        us = pd.DataFrame()
        reference_funds = 0

        def _load_b3():
            raw = get_full_market_data()
            if raw is None:
                return pd.DataFrame()
            frame = normalize_market(raw.loc[~raw.index.isin(EXTRA_ASSETS_BY_TICKER)])
            return _provenance(frame, "Fundamentus", asset_class="acao_br", currency="BRL",
                               fetched_at=raw.attrs.get("fetched_at"))

        def _load_fii():
            frame = get_fii_market_data()
            fetched_at = frame.attrs.get("fetched_at")
            frame = frame.rename(columns={"price": "cotacao"})
            frame["dy"] = frame["dy"] / 100
            return _provenance(frame, "Fundamentus · FIIs", asset_class="fii", currency="BRL", fetched_at=fetched_at)

        def _load_us():
            frame = get_us_market_data()
            return _ensure_provenance(frame, body, "Yahoo Finance · Ações EUA")

        if body.asset_class in {"b3", "acao_br"}:
            b3 = _load_b3()
            result, exported = screen(b3, body, source="Fundamentus", normalized=True)
        elif body.asset_class == "fii":
            try:
                fii, reference_funds = _merge_reference_funds(_load_fii())
            except Exception as exc:
                raise HTTPException(503, "O painel de FIIs da Fundamentus está indisponível. Tente novamente.") from exc
            source = "Fundamentus · FIIs"
            if reference_funds:
                source += f" + {FII_REFERENCE_SOURCE} ({reference_funds} fundo{'s' if reference_funds != 1 else ''})"
            result, exported = screen(fii, body, source=source, normalized=True)
        elif body.asset_class == "acao_us":
            try:
                us = _load_us()
            except Exception as exc:
                raise HTTPException(503, "Os dados de ações americanas no Yahoo Finance estão indisponíveis. Tente novamente.") from exc
            result, exported = screen(us, body, source="Yahoo Finance · Ações EUA", normalized=True)
            result["unavailable_indicators"] = [
                column["label"] for column in result["available_columns"]
                if column["key"] in CRITERION_FIELDS.values() and us[column["key"]].isna().all()
            ]
        else:
            if body.asset_class == "all":
                def _safe(fn):
                    try:
                        return fn()
                    except Exception:
                        return pd.DataFrame()

                with ThreadPoolExecutor(max_workers=3) as pool:
                    b3_fut = pool.submit(_safe, _load_b3)
                    fii_fut = pool.submit(_safe, lambda: _merge_reference_funds(_load_fii())[0])
                    us_fut = pool.submit(_safe, _load_us)
                    b3, fii, us = b3_fut.result(), fii_fut.result(), us_fut.result()
            reference = catalog_screener_rows(body.asset_class, SCREEN_FIELDS)
            if body.asset_class == "all" and not reference.empty:
                b3 = b3.loc[~b3.index.isin(fii.index)]
                reference = reference.loc[~reference["asset_class"].isin({"fii", "acao_us"})]
            if not reference.empty:
                reference = _with_live_quotes(reference)
            source = "Multi-Ativos · fonte por linha" if body.asset_class == "all" else _catalog_source(reference)
            frame = reference
            if body.asset_class == "all":
                parts = [part for part in (b3, fii, us, reference) if not part.empty]
                frame = pd.concat(parts) if parts else pd.DataFrame()
                frame = frame.reindex(columns=frame.columns.union(sorted(fields), sort=False))
                source += " + Fundamentus · ações B3" if not b3.empty else " (ações B3 indisponíveis)"
                source += " + Fundamentus · FIIs" if not fii.empty else " (FIIs indisponíveis)"
                source += " + Yahoo Finance · ações EUA" if not us.empty else " (ações EUA indisponíveis)"
            result, exported = screen(frame, body, source=source, normalized=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Não foi possível carregar os dados do screener: {exc}") from exc
    result["id"] = uuid.uuid4().hex
    publish(ctx, "screener", revision, generation,
            {"id": result["id"], "result": result, "export": exported, "asset_class": body.asset_class})
    return result


@router.get("/{id}/results.csv")
def results_csv(id: str, request: Request):
    saved = artifact(session(request), "screener", id)
    asset_class = saved.get("asset_class") or "b3"
    return csv_response(saved["export"], f"screener_{asset_class}_{dt.date.today():%Y%m%d}.csv")
