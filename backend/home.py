"""Home fundamentals, consensus and source catalog. Detail units deliberately differ from peers."""
import datetime as dt
import logging
import math
import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from backend.serialization import StrictModel, clean_json, csv_response
from backend.state import artifact, begin, publish, session
from utils.formatting import extract_debt_metric, get_ev_ebitda_context
from utils.home_data import (build_analyst_synthesis, clear_fundamentus_cache,
                             compute_sector_ranking, get_fundamentus_data, get_sector_peers)
from utils.market_data import (clean_numeric_column, compute_target_upside, get_listed_stocks,
                               get_market_target_data, get_sorted_tickers_by_liquidity)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/home", tags=["home"])
catalog_router = APIRouter(prefix="/api", tags=["catalog"])
EXPORT_FIELDS = ["Empresa", "Setor", "Subsetor", "Cotacao", "Min_52_sem", "Max_52_sem",
                 "Marg_Liquida", "Marg_EBIT", "ROE", "ROIC", "Div_Yield", "Cres_Rec_5a", "PL", "EV_EBITDA", "PVP"]
DEBT_ALERT = 3
DEBT_WARNING = 1.5
LIQUIDITY_ALERT = 1
LIQUIDITY_WARNING = 1.5
HEALTH_SUMMARY = (f"Endividamento e liquidez da empresa. Dívida Líquida/PL acima de {DEBT_ALERT}× e liquidez corrente abaixo de {LIQUIDITY_ALERT}× "
                  "são sinais de alerta; dívida líquida negativa indica caixa líquido.")
INDICATORS = [
    ("pl", "PL", "P/L", "multiple", "Valoração", "Preço/Lucro: preço dividido pelo lucro por ação dos últimos 12 meses. Múltiplo menor pode indicar desconto, mas depende do setor e do ciclo."),
    ("pvp", "PVP", "P/VP", "multiple", "Valoração", "Preço/Valor Patrimonial: compara o preço de mercado com o valor contábil. <1 pode indicar desconto."),
    ("ev_ebitda", "EV_EBITDA", "EV/EBITDA", "multiple", "Valoração", "Enterprise Value / EBITDA dos últimos 12 meses: múltiplo que considera a dívida líquida. Menor pode indicar desconto, conforme o setor."),
    ("roe", "ROE", "ROE", "percent", "Rentabilidade", "Return on Equity: lucro dos últimos 12 meses para cada R$ de patrimônio. Acima de 15% costuma ser considerado bom, mas o referencial varia por setor."),
    ("roic", "ROIC", "ROIC", "percent", "Rentabilidade", "Return on Invested Capital: eficiência no uso de todo o capital (próprio + dívida) nos últimos 12 meses. Referência varia por setor."),
    ("net_margin", "Marg_Liquida", "Margem Líquida", "percent", "Rentabilidade", "Percentual da receita que vira lucro líquido. Quanto maior, mais lucrativa a empresa."),
    ("ebit_margin", "Marg_EBIT", "Margem EBIT", "percent", "Rentabilidade", "Margem operacional antes de juros e impostos. Mede a eficiência operacional."),
    ("dy", "Div_Yield", "Dividend Yield", "percent", "Crescimento e dividendos","Proventos pagos nos últimos 12 meses (dividendos e JCP, antes do IR) divididos pelo preço atual. Retorno bruto e passado; não garante proventos futuros."),
    ("growth", "Cres_Rec_5a", "Crescimento Receita (5 anos)", "percent", "Crescimento e dividendos","Taxa de crescimento anual composta da receita nos últimos 5 anos. Referência varia por setor."),
]
MARKET = [("price", "Cotacao"), ("low_52w", "Min_52_sem"), ("high_52w", "Max_52_sem"),
          ("volume_2m", "Vol_med_2m"), ("market_cap", "Valor_de_mercado")]


class HomeRequest(StrictModel):
    tickers: list[str] = Field(min_length=1, max_length=20)
    revision: int = Field(ge=0)


class SuggestionsRequest(StrictModel):
    sectors: list[str] = Field(default_factory=list)


def positive(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def numeric(value):
    return clean_json(clean_numeric_column(pd.Series([value])).iloc[0])


def recommendation(key, score):
    labels = {"strongbuy": "Compra forte", "buy": "Compra", "hold": "Manter", "underperform": "Abaixo da média", "sell": "Venda", "strongsell": "Venda forte"}
    normalized = str(key or "").lower().replace("_", "").replace("-", "")
    if normalized in labels:
        return labels[normalized]
    if normalized and normalized != "none":
        return str(key).replace("_", " ").title()
    value = positive(score)
    if value is None or value > 5:
        return "—"
    return "Compra forte" if value <= 1.5 else "Compra" if value <= 2.5 else "Manter" if value <= 3.5 else "Abaixo da média" if value <= 4.5 else "Venda"


def consensus(tickers, refresh=False):
    rows, failed, has_targets = [], [], False
    for ticker in tickers:
        if refresh:
            get_market_target_data.clear(ticker)
        market = get_market_target_data(ticker) or {}
        if "_error" in market:
            failed.append(ticker)
            market = {}
        price = positive(market.get("price")) or positive(market.get("regular_price"))
        targets = {key: positive(market.get(key)) for key in ("target_low", "target_mean", "target_median", "target_high")}
        has_targets |= any(value is not None for value in targets.values())
        count = positive(market.get("analyst_count"))
        rows.append({"ticker": ticker, "company": market.get("company_name") or ticker,
                     "currency": str(market.get("currency") or "BRL").upper(), "price": price, **targets,
                     "upside": compute_target_upside(price, targets["target_mean"]),
                     "analyst_count": int(count) if count else None,
                     "recommendation": recommendation(market.get("recommendation"), market.get("recommendation_mean"))})
    return {"rows": rows, "failed_tickers": failed, "has_targets": has_targets, "source": "Yahoo Finance"}


def health(row):
    debt = clean_json(extract_debt_metric(row, ["Div_Liq_Patrim"]))
    debt = debt / 100 if debt is not None else None  # Fundamentus hundredths; negative = net cash
    liquidity = clean_json(extract_debt_metric(row, ["Liquidez_Corr"]))
    liquidity = liquidity / 100 if liquidity is not None else None
    ev = clean_json(extract_debt_metric(row, ["EV_EBIT", "EV/EBIT"]))
    ev = ev / 100 if ev is not None else None
    metrics, diagnostics = [], []
    if debt is not None:
        if debt < 0:
            status = "positive"
            text = f"Dívida Líquida/PL de {debt:.2f}×: caixa líquido — a empresa tem mais caixa do que dívida líquida."
        else:
            status = "negative" if debt > DEBT_ALERT else "warning" if debt > DEBT_WARNING else "positive"
            text = f"Dívida Líquida/PL de {debt:.2f}× " + ("é elevada — verifique capacidade de pagamento" if debt > DEBT_ALERT else "é moderada — monitorar" if debt > DEBT_WARNING else "é saudável")
        metrics.append({"label": "Dív. Líquida/PL", "value": debt, "unit": "multiple", "status": status})
        diagnostics.append({"status": status, "text": text})
    if liquidity is not None:
        status = "negative" if liquidity < LIQUIDITY_ALERT else "warning" if liquidity < LIQUIDITY_WARNING else "positive"
        metrics.append({"label": "Liquidez Corrente", "value": liquidity, "unit": "multiple", "status": status})
        text = f"Liquidez Corrente {liquidity:.2f}× " + ("< 1 — risco de dificuldade de caixa" if liquidity < LIQUIDITY_ALERT else "— margem estreita" if liquidity < LIQUIDITY_WARNING else "— empresa com boa folga de caixa")
        diagnostics.append({"status": status, "text": text})
    if ev is not None:
        metric = {"label": "EV / EBIT", "value": ev, "unit": "multiple", "digits": 1,
                  "status": "negative" if ev > 20 else "warning" if ev > 12 else "positive"}
        if ev <= 0:
            metric.update(status="neutral", help="EBIT negativo — múltiplo não interpretável.")
        metrics.append(metric)
    return {"metrics": metrics, "diagnostics": diagnostics}


def indicator_metric(key, label, unit, help_text, value, sector, negative_equity=False):
    metric = {"label": label, "value": value, "unit": unit, "digits": 2, "help": help_text, "status": "neutral", "classification": None}
    if value is None:  # missing source data stays null: no classification, no colour
        return metric
    if key == "ev_ebitda" and abs(value) < .01:
        context = get_ev_ebitda_context(sector)
        metric.update(value=None, classification=context[0] if context else "N/D", help=context[1] if context else "Dado não disponível para este ticker via Fundamentus.")
    elif key in ("pvp", "roe") and negative_equity:
        metric["classification"] = "Patrimônio líquido negativo — indicador não interpretável"
    elif key == "pl" and value < 0:
        metric["classification"] = "Prejuízo — múltiplo não interpretável"
    elif key == "ev_ebitda" and value < 0:
        metric["classification"] = "EBITDA negativo — múltiplo não interpretável"
    elif key in ("roe", "roic", "net_margin", "ebit_margin", "growth"):
        high, low = (12, 5) if key == "roic" else (10, 0) if key == "growth" else (15, 5)
        strong = "Alta" if "margin" in key else "Forte"
        weak = "Baixa" if "margin" in key else "Negativo" if key == "growth" else "Fraco"
        middle = "Média" if "margin" in key else "Moderado"
        metric.update(classification=strong if value > high else weak if value < low else middle,
                      status="positive" if value > high else "negative" if value < low else "neutral")
    elif key == "dy":
        metric.update(classification="Alto" if value > 6 else "Baixo" if value < 2 else "Moderado", status="positive" if value > 4 else "neutral")
    return metric


def normalize_detail(frame):
    result = frame.copy()
    result.index = [str(value).replace(".SA", "").strip().upper() for value in result.index]
    return result


def build_analysis(raw, peers, tickers, universe):
    raw = normalize_detail(raw)
    latest = raw[~raw.index.duplicated(keep="last")]
    indicator_columns = [source for _, source, *_ in INDICATORS if source in raw] + (["Empresa"] if "Empresa" in raw else [])
    indicator_frame = raw[indicator_columns].drop_duplicates(keep="last")
    indicator_frame = indicator_frame[~indicator_frame.index.duplicated(keep="last")]
    companies = []
    sector_lookup = universe.drop_duplicates("Ticker").set_index("Ticker")["Setor"].to_dict()
    for ticker in tickers:
        present = ticker in latest.index
        row = latest.loc[ticker] if present else pd.Series(dtype=object)
        ind_row = indicator_frame.loc[ticker] if ticker in indicator_frame.index else pd.Series(dtype=object)
        sector = str(row.get("Setor", "")) if pd.notna(row.get("Setor", "")) else ""
        indicators, metric_groups, indicator_available = {}, {}, {}
        for key, source, label, unit, group, help_text in INDICATORS:
            value = numeric(ind_row.get(source))
            if value is not None and unit == "multiple":
                value /= 100
            if key in ("net_margin", "ebit_margin") and value == 0:
                value = None  # Fundamentus prints 0,0% for banks and insurers, where margins do not apply
            indicators[key] = value
            negative_equity = indicators.get("pvp") is not None and indicators["pvp"] < 0
            metric = indicator_metric(key, label, unit, help_text, value, sector, negative_equity)
            metric_groups.setdefault(group, []).append(metric)
            indicator_available[key] = metric["value"] is not None
        local_sector = sector_lookup.get(ticker, "")
        local_sector = local_sector.strip() if isinstance(local_sector, str) else ""
        synthesis = build_analyst_synthesis(peers, ticker, local_sector, universe)
        ranking = compute_sector_ranking(peers, ticker, local_sector, universe) if not peers.empty and local_sector else pd.DataFrame()
        details = []
        if not ranking.empty:
            for detail in ranking.to_dict("records"):
                if detail["Peers (n)"] < 3 or pd.isna(detail["Valor"]) or pd.isna(detail["Percentil"]):
                    continue
                is_percent = detail["Múltiplo"].endswith(" (%)")
                details.append({"label": detail["Múltiplo"], "value": detail["Valor"] * (100 if is_percent else 1),
                                "unit": "percent" if is_percent else "multiple", "median": detail["Mediana Setor"] * (100 if is_percent else 1),
                                "mean": detail["Média Setor"] * (100 if is_percent else 1), "observations": detail["Peers (n)"],
                                "percentile": detail["Percentil"], "verdict": detail["Veredicto"], "category": detail["Categoria"]})
        companies.append({"ticker": ticker, "available": present,
                          "identity": {"company": row.get("Empresa"), "sector": sector or None, "subsector": row.get("Subsetor")},
                          "market": {**{key: numeric(row.get(source)) for key, source in MARKET}, "last_quote_date": row.get("Data_ult_cot"), "currency": "BRL"},
                          "indicators": indicators, "indicator_available": indicator_available,
                          "metric_groups": metric_groups, "indicators_available": ticker in indicator_frame.index,
                          "health": health(row), "synthesis": {**synthesis, "sector": local_sector or None, "details": details,
                          "favorable_categories": sum(c["veredicto"] == "Favorável" for c in synthesis["categorias"].values())}})
    return clean_json({"requested_tickers": tickers, "missing_tickers": [t for t in tickers if t not in latest.index],
                       "companies": companies, "health_summary": HEALTH_SUMMARY, "comparison_indicators": [{"key": k, "label": label, "unit": unit} for k, _, label, unit, *_ in INDICATORS],
                       "sources": {"fundamentals": "Fundamentus", "peer_units": "fractions", "detail_percentages": "percentage_points", "detail_multiples": "ratio (scraper /100 correction)"}})


def validate_selection(ctx, body):
    tickers = [str(t).strip().upper().removesuffix(".SA") for t in body.tickers]
    if len(set(tickers)) != len(tickers):
        raise HTTPException(422, "A seleção contém tickers repetidos.")
    with ctx.lock:
        if body.revision != ctx.revision or tickers != ctx.selection:
            raise HTTPException(409, "A seleção mudou; clique em Analisar novamente.")
    known = set(get_listed_stocks()["Ticker"])
    if any(t not in known for t in tickers):
        raise HTTPException(422, "A seleção contém ações fora da lista B3.")
    return tickers


@catalog_router.get("/universe")
def universe():
    try:
        frame = get_listed_stocks()
        return clean_json({"source": "local_csv", "stocks": [{"ticker": row["Ticker"], "company": row.get("Empresa", ""), "legal_name": row.get("RazaoSocial", ""), "sector": row.get("Setor", "")} for row in frame.to_dict("records")],
                           "sectors": sorted(frame["Setor"].dropna().unique()), "selection_limit": 20, "sector_autopick_limit": 8})
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"Não foi possível carregar a lista de ações da B3: {exc}") from exc


@catalog_router.post("/universe/suggestions")
def suggestions(body: SuggestionsRequest):
    frame = get_listed_stocks()
    matching = frame if not body.sectors or "Todos" in body.sectors else frame[frame["Setor"].isin(body.sectors)]
    tickers = list(dict.fromkeys(get_sorted_tickers_by_liquidity(matching["Ticker"].tolist())))
    return {"tickers": tickers, "count": len(tickers), "autopick": tickers[:8], "source": "Fundamentus · ordenação por liquidez"}


@catalog_router.post("/fundamentus/refresh")
def refresh_fundamentus():
    removed = clear_fundamentus_cache()
    get_sorted_tickers_by_liquidity.clear()
    return {"removed": removed, "scope": "global Fundamentus; consenso Yahoo não alterado"}


@router.post("/analyze")
def analyze(body: HomeRequest, request: Request):
    ctx = session(request)
    tickers = validate_selection(ctx, body)
    generation = begin(ctx, "home", body.revision)
    try:
        universe_frame = get_listed_stocks()
        raw = get_fundamentus_data(tickers)
        if raw is None or raw.empty:
            raise HTTPException(503, "A Fundamentus não retornou dados para os ativos selecionados. Tente novamente.")
        peers = get_sector_peers()
        result = build_analysis(raw, peers, tickers, universe_frame)
        if result["missing_tickers"]:
            clear_fundamentus_cache()
        result["consensus"] = consensus(tickers)
        result["id"] = uuid.uuid4().hex
        result["revision"] = body.revision
        raw_export = raw[[column for column in EXPORT_FIELDS if column in raw]].copy()
        publish(ctx, "home", body.revision, generation, {"id": result["id"], "result": result, "export": raw_export})
        return clean_json(result)
    except HTTPException:
        raise
    except OSError as exc:
        from utils.cache import clear_all
        clear_all()
        logger.exception("Home I/O failure")
        raise HTTPException(503, f"Erro de I/O ao buscar dados. O cache foi limpo automaticamente. Detalhe técnico: {exc}") from exc
    except Exception as exc:
        logger.exception("Home analysis failed")
        raise HTTPException(503, "Não foi possível concluir a análise agora. A fonte de dados pode estar temporariamente indisponível. Verifique a seleção e tente novamente.") from exc


@router.post("/consensus/refresh")
def refresh_consensus(body: HomeRequest, request: Request):
    ctx = session(request)
    tickers = validate_selection(ctx, body)
    with ctx.lock:
        current = ctx.artifacts.get("home")
        if not current or current["result"]["requested_tickers"] != tickers or current["result"]["revision"] != body.revision:
            raise HTTPException(409, "Clique em Analisar antes de atualizar os consensos.")
    generation = begin(ctx, "home-consensus", body.revision)
    result = consensus(tickers, refresh=True)
    publish(ctx, "home-consensus", body.revision, generation, {"id": uuid.uuid4().hex, "result": result})
    return clean_json(result)


@router.get("/current")
def current_analysis(request: Request):
    ctx = session(request)
    with ctx.lock:
        saved = ctx.artifacts.get("home")
        if not saved or saved["result"]["revision"] != ctx.revision or saved["result"]["requested_tickers"] != ctx.selection:
            return None
        return clean_json(saved["result"])


@router.get("/{id}/fundamentals.csv")
def fundamentals_csv(id: str, request: Request):
    saved = artifact(session(request), "home", id)
    tickers = saved["result"]["requested_tickers"]
    return csv_response(saved["export"], f"fundamentus_{'+'.join(tickers)}_{dt.date.today().isoformat()}.csv")
