"""Home fundamentals, consensus and source catalog. Detail units deliberately differ from peers."""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import logging
import math
import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from backend.serialization import StrictModel, clean_json, csv_response
from backend.state import artifact, begin, publish, session
from utils.assets import (
    ASSET_CLASSES, ASSET_INDICATORS, EQUITY_CLASSES, asset_metadata, get_extra_asset_fundamentals, get_full_universe,
    is_valid_asset_ticker, load_fii_market_data, market_asset_detail, normalize_ticker, parse_fixed_income_asset, provider_symbol,
    search_yahoo_assets, supports_share_allocation, uses_market_quote,
)
from utils.formatting import extract_debt_metric, get_ev_ebitda_context
from utils.home_data import (build_analyst_synthesis, clear_fundamentus_cache,
                             compute_sector_ranking, get_fundamentus_data, get_sector_peers)
from utils.market_data import (clean_numeric_column, compute_target_upside, get_fii_market_data, get_liquidity_by_ticker,
                               get_listed_stocks, get_universe_changes, get_market_target_data, get_sorted_tickers_by_liquidity,
                               get_yahoo_info, get_yahoo_quotes)
from utils.treasury_data import TREASURY_TITLES
from utils.equity_history import get_equity_history

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/home", tags=["home"])
catalog_router = APIRouter(prefix="/api", tags=["catalog"])
EXPORT_FIELDS = ["Empresa", "Setor", "Subsetor", "Classe", "Moeda", "Fonte",
                 "Cotacao", "Data_ult_cot", "Min_52_sem", "Max_52_sem", "Vol_med_2m", "Valor_de_mercado",
                 "Marg_Liquida", "Marg_EBIT", "ROE", "ROIC", "Div_Yield", "Cres_Rec_5a",
                 "PL", "EV_EBITDA", "EV_EBIT", "PVP", "Div_Liq_Patrim", "Liquidez_Corr"]
# Flattened from the detail frame's `Tesouro` mapping when at least one selected title has official data.
TREASURY_EXPORT_FIELDS = {
    "buy_price": "Tesouro_PU_Compra", "sell_price": "Tesouro_PU_Venda",
    "buy_rate": "Tesouro_Taxa_Compra", "sell_rate": "Tesouro_Taxa_Venda",
    "rate_unit": "Tesouro_Unidade_Taxa", "maturity": "Tesouro_Vencimento",
    "quote_date": "Tesouro_Data_Cotacao", "source": "Tesouro_Fonte",
}
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


def consensus(tickers, refresh=False, asset_classes=None, skip_yahoo=None):
    eligible = [
        ticker for ticker in tickers
        if (asset_classes[ticker] if asset_classes is not None else asset_metadata(ticker)["asset_class"]) in EQUITY_CLASSES | {"reit"}
    ]
    skip_yahoo = set(skip_yahoo or ())

    def _fetch_one(ticker):
        if refresh:
            get_market_target_data.clear(ticker)
            get_yahoo_info.clear(provider_symbol(ticker))
        if not refresh and ticker in skip_yahoo:
            return ticker, {"_error": "Yahoo Finance indisponível"}
        return ticker, get_market_target_data(ticker) or {}

    if len(eligible) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(eligible))) as pool:
            fetched = list(pool.map(_fetch_one, eligible))
    else:
        fetched = [_fetch_one(ticker) for ticker in eligible]

    rows, failed, has_targets = [], [], False
    for ticker, market in fetched:
        if "_error" in market:
            failed.append(ticker)
            market = {}
        price = positive(market.get("price")) or positive(market.get("regular_price"))
        targets = {key: positive(market.get(key)) for key in ("target_low", "target_mean", "target_median", "target_high")}
        has_targets |= any(value is not None for value in targets.values())
        count = positive(market.get("analyst_count"))
        rows.append({"ticker": ticker, "company": market.get("company_name") or ticker,
                     "currency": str(market.get("currency") or asset_metadata(ticker)["currency"]).upper(), "price": price, **targets,
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
        debt_text = f"{debt:.2f}".replace(".", ",")
        if debt < 0:
            status = "positive"
            text = f"Dívida Líquida/PL de {debt_text}×: caixa líquido — a empresa tem mais caixa do que dívida líquida."
        else:
            status = "negative" if debt > DEBT_ALERT else "warning" if debt > DEBT_WARNING else "positive"
            text = f"Dívida Líquida/PL de {debt_text}× " + ("é elevada — verifique capacidade de pagamento" if debt > DEBT_ALERT else "é moderada — monitorar" if debt > DEBT_WARNING else "é saudável")
        metrics.append({"label": "Dív. Líquida/PL", "value": debt, "unit": "multiple", "status": status})
        diagnostics.append({"status": status, "text": text})
    if liquidity is not None:
        liquidity_text = f"{liquidity:.2f}".replace(".", ",")
        status = "negative" if liquidity < LIQUIDITY_ALERT else "warning" if liquidity < LIQUIDITY_WARNING else "positive"
        metrics.append({"label": "Liquidez Corrente", "value": liquidity, "unit": "multiple", "status": status})
        text = f"Liquidez Corrente {liquidity_text}× " + ("< 1 — risco de dificuldade de caixa" if liquidity < LIQUIDITY_ALERT else "— margem estreita" if liquidity < LIQUIDITY_WARNING else "— empresa com boa folga de caixa")
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
        meta = asset_metadata(ticker, universe)
        asset_class = row["Classe"] if pd.notna(row.get("Classe")) else meta["asset_class"]
        source_name = row["Fonte"] if pd.notna(row.get("Fonte")) else "Fundamentus" if ticker in sector_lookup else None
        sector = str(row["Setor"]) if pd.notna(row.get("Setor")) else meta["sector"]
        applicable = ASSET_INDICATORS[asset_class]
        indicators, metric_groups, indicator_available = {}, {}, {}
        for key, source, label, unit, group, help_text in INDICATORS:
            if key not in applicable:
                continue
            value = numeric(ind_row.get(source))
            if value is not None and unit == "multiple":
                value /= 100
            if source_name == "Fundamentus" and key in ("net_margin", "ebit_margin") and value == 0:
                value = None  # Fundamentus prints 0,0% for banks and insurers, where margins do not apply
            indicators[key] = value
            negative_equity = indicators.get("pvp") is not None and indicators["pvp"] < 0
            if asset_class not in EQUITY_CLASSES:
                if key == "dy":
                    label, group, help_text = "Rendimento distribuído (12m)", "Rendimentos", "Proventos dos últimos 12 meses divididos pelo preço. Não representa retorno total ou futuro."
                metric = {"label": label, "unit": unit, "value": value, "help": help_text, "classification": None, "status": "neutral"}
            else:
                metric = indicator_metric(key, label, unit, help_text, value, sector, negative_equity)
            metric_groups.setdefault(group, []).append(metric)
            indicator_available[key] = metric["value"] is not None
        local_sector = sector_lookup.get(ticker, "") if asset_class == "acao_br" else ""
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
        price_unit = row.get("Unidade_cotacao")
        quote_pair = row.get("Par_cotacao")
        volume_window = row.get("Janela_volume")
        contract = row.get("Contrato")
        fund = row.get("Fundo")
        treasury = row.get("Tesouro")
        history = row.get("Historico")
        companies.append({"ticker": ticker, "available": present, "asset_class": asset_class, "source": source_name,
                          "fixed_income": row.get("fixed_income"),
                          "identity": {"company": row.get("Empresa", meta["company"]), "sector": sector or None, "subsector": row.get("Subsetor")},
                          "market": {**{key: numeric(row.get(source)) for key, source in MARKET}, "last_quote_date": row.get("Data_ult_cot"),
                                     "currency": str(row["Moeda"]) if pd.notna(row.get("Moeda")) else meta["currency"],
                                     "price_unit": price_unit if price_unit in {"money", "points"} else "money",
                                     "quote_pair": quote_pair if isinstance(quote_pair, str) else None,
                                     "volume_24h": numeric(row.get("Volume_24h")),
                                     "session_volume": numeric(row.get("Volume_sessao")),
                                     "volume_window": volume_window if isinstance(volume_window, str) else None,
                                     "contract": contract if isinstance(contract, dict) else None},
                          "fund": fund if isinstance(fund, dict) else None,
                          "treasury": treasury if isinstance(treasury, dict) else None,
                          "history": history if isinstance(history, dict) else None,
                          "indicators": indicators, "indicator_available": indicator_available,
                          "metric_groups": metric_groups, "indicators_available": any(indicator_available.values()),
                          "health": health(row) if asset_class in EQUITY_CLASSES else {"metrics": [], "diagnostics": []},
                          "synthesis": {**synthesis, "sector": local_sector or sector or None, "details": details,
                          "favorable_categories": sum(c["veredicto"] == "Favorável" for c in synthesis["categorias"].values())}})
    shown = set().union(*(ASSET_INDICATORS[company["asset_class"]] for company in companies))
    fund_yields = any(company["asset_class"] in {"fii", "reit", "etf"} for company in companies)
    sources = dict.fromkeys(company["source"] for company in companies if company["available"])
    return clean_json({"requested_tickers": tickers, "missing_tickers": [t for t in tickers if t not in latest.index],
                       "companies": companies, "health_summary": HEALTH_SUMMARY, "comparison_indicators": [{"key": key, "label": "Proventos / preço (12m)" if key == "dy" and fund_yields else label, "unit": unit} for key, _, label, unit, *_ in INDICATORS if key in shown],
                       "sources": {"fundamentals": " / ".join(sources), "peer_units": "fractions", "detail_percentages": "percentage_points", "detail_multiples": "ratio (scraper /100 correction)"}})


def _extra_fundamentals_frame(tickers: list[str], b3_stocks: dict[str, dict]) -> pd.DataFrame:
    """Convert non-Fundamentus assets (FIIs, US stocks, REITs, Títulos/RF, ETFs) into build_analysis detail schema."""
    quote_only = {ticker for ticker in tickers if uses_market_quote(ticker)}
    needs_fii = any(ticker not in b3_stocks and asset_metadata(ticker)["asset_class"] == "fii" for ticker in tickers)
    fii_table = load_fii_market_data() if needs_fii else pd.DataFrame()
    detail_tickers = [ticker for ticker in tickers if ticker not in quote_only]

    def _load_detail(ticker):
        return ticker, get_extra_asset_fundamentals(ticker, fii_table=fii_table, b3_stock=b3_stocks.get(ticker))

    def _load_quote_info(ticker):
        try:
            return ticker, get_yahoo_info(provider_symbol(ticker)) or {}
        except Exception as exc:
            logger.warning("Yahoo Finance info unavailable for %s: %s", ticker, exc)
            return ticker, {}

    if len(detail_tickers) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(detail_tickers))) as pool:
            details = dict(pool.map(_load_detail, detail_tickers))
    else:
        details = dict(_load_detail(ticker) for ticker in detail_tickers)
    # Quote-only assets carry unit, volume and contract metadata in Yahoo's info payload;
    # fetch it once per ticker in parallel instead of serially inside the row loop.
    if len(quote_only) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(quote_only))) as pool:
            quote_info = dict(pool.map(_load_quote_info, sorted(quote_only)))
    else:
        quote_info = dict(_load_quote_info(ticker) for ticker in quote_only)
    fii_quotes = {ticker for ticker, detail in details.items() if detail.get("source") == "Fundamentus · FIIs"}
    quote_tickers = quote_only | fii_quotes | {ticker for ticker, data in details.items()
                                               if not data.get("available") and ticker not in TREASURY_TITLES
                                               and parse_fixed_income_asset(ticker) is None}
    symbols = {ticker: provider_symbol(ticker) for ticker in quote_tickers}
    quotes = get_yahoo_quotes(tuple(sorted(set(symbols.values())))) if symbols else {}
    rows = []
    index = []
    for ticker in tickers:
        quote = quotes.get(symbols.get(ticker))
        if ticker in fii_quotes:
            data = details[ticker]
            if quote and quote["currency"] == "BRL":
                data = {**data, **quote, "source": "Fundamentus · FIIs / Yahoo Finance"}
        elif ticker in symbols:
            data = market_asset_detail(ticker, quote, allow_nasdaq=ticker in quote_only,
                                       b3_stock=b3_stocks.get(ticker), info=quote_info.get(ticker))
        else:
            data = details[ticker]
        if not data.get("available"):
            continue
        ind = data.get("indicators") or {}
        hlth = data.get("health") or {}
        def _scaled(val):
            return val * 100.0 if val is not None else None
        rows.append({
            "Empresa": data.get("company") or ticker,
            "Setor": data.get("sector") or "",
            "Subsetor": data.get("subsector") or "",
            "Classe": data["asset_class"],
            "Fonte": data["source"],
            "fixed_income": data.get("fixed_income"),
            "Remuneracao_modelo": (data.get("fixed_income") or {}).get("label"),
            "Taxa_modelo": (data.get("fixed_income") or {}).get("rate"),
            "Unidade_taxa_modelo": (data.get("fixed_income") or {}).get("unit"),
            "Moeda": data.get("currency") or "BRL",
            "Cotacao": data.get("price"),
            "Min_52_sem": data.get("low_52w"),
            "Max_52_sem": data.get("high_52w"),
            "Vol_med_2m": data.get("volume_2m"),
            "Valor_de_mercado": data.get("market_cap"),
            "Data_ult_cot": data.get("last_quote_date"),
            "Unidade_cotacao": data.get("price_unit") or "money",
            "Par_cotacao": data.get("quote_pair"),
            "Volume_24h": data.get("volume_24h"),
            "Volume_sessao": data.get("session_volume"),
            "Janela_volume": data.get("volume_window"),
            "Contrato": data.get("contract"),
            "Fundo": data.get("fund"),
            "Tesouro": data.get("treasury"),
            "Historico": data.get("history"),
            "PL": _scaled(ind.get("pl")),
            "PVP": _scaled(ind.get("pvp")),
            "EV_EBITDA": _scaled(ind.get("ev_ebitda")),
            "ROE": ind.get("roe"),
            "ROIC": ind.get("roic"),
            "Marg_Liquida": ind.get("net_margin"),
            "Marg_EBIT": ind.get("ebit_margin"),
            "Div_Yield": ind.get("dy"),
            "Cres_Rec_5a": ind.get("growth"),
            "Div_Liq_Patrim": _scaled(hlth.get("debt_equity")),
            "Liquidez_Corr": _scaled(hlth.get("current_ratio")),
            "EV_EBIT": _scaled(hlth.get("ev_ebit")),
        })
        index.append(ticker)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=index)


def validate_selection(ctx, body):
    tickers = [normalize_ticker(t) for t in body.tickers]
    if len(set(tickers)) != len(tickers):
        raise HTTPException(422, "A seleção contém tickers repetidos.")
    with ctx.lock:
        if body.revision != ctx.revision or tickers != ctx.selection:
            raise HTTPException(409, "A seleção mudou; clique em Analisar novamente.")
    known = set(get_listed_stocks()["Ticker"])
    if any(not is_valid_asset_ticker(t, known) for t in tickers):
        raise HTTPException(422, "A seleção contém ativos inválidos.")
    return tickers


@catalog_router.get("/universe")
def universe():
    try:
        frame = get_listed_stocks()
        stocks = get_full_universe(frame)
        return clean_json({"source": "multi_asset_catalog", "stocks": stocks,
                           "sectors": sorted(frame["Setor"].dropna().unique()),
                           "asset_classes": list(ASSET_CLASSES),
                           "selection_limit": 20, "sector_autopick_limit": 8})
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"Não foi possível carregar a lista de ativos: {exc}") from exc


@catalog_router.get("/universe/search")
def universe_search(q: str = "", asset_class: str = ""):
    query = str(q or "").strip()
    b3_frame = get_listed_stocks()
    catalog = get_full_universe(b3_frame)
    if asset_class:
        catalog = [item for item in catalog if item.get("asset_class") == asset_class]
    if not query:
        return clean_json({"items": catalog[:30], "query": query})

    needle = query.casefold()
    results: list[dict] = []
    seen: set[str] = set()

    # 1. Dynamic fixed-income expression (e.g. "CDB-112CDI", "IPCA+7.5", "PRE-13.5", "TESOURO-IPCA-2040")
    fi = parse_fixed_income_asset(query)
    if fi is not None and (not asset_class or asset_class == "titulo"):
        t = fi["ticker"]
        seen.add(t)
        results.append({
            "ticker": t,
            "company": fi["company"],
            "legal_name": fi["company"],
            "sector": fi["sector"],
            "asset_class": "titulo",
            "currency": fi["currency"],
            "supports_shares": False,
        })

    # 2. Local multi-asset catalog matches
    scored = []
    for item in catalog:
        t = item["ticker"]
        if t in seen:
            continue
        t_low = t.casefold()
        c_low = str(item.get("company") or "").casefold()
        s_low = str(item.get("sector") or "").casefold()
        rank = (0 if t_low == needle else
                1 if t_low.startswith(needle) else
                2 if c_low.startswith(needle) else
                3 if needle in c_low else
                4 if needle in s_low else -1)
        if rank >= 0:
            scored.append((rank, item))
    scored.sort(key=lambda pair: pair[0])
    for _, item in scored[:16]:
        seen.add(item["ticker"])
        results.append(item)

    # 3. Live Yahoo Finance search for any global ticker not yet covered
    if len(query) >= 2 and len(results) < 12:
        for remote in search_yahoo_assets(query):
            if asset_class and remote.get("asset_class") != asset_class:
                continue
            if remote["ticker"] not in seen:
                seen.add(remote["ticker"])
                results.append({
                    "ticker": remote["ticker"],
                    "company": remote["company"],
                    "legal_name": remote.get("legal_name", remote["company"]),
                    "sector": remote["sector"],
                    "asset_class": remote["asset_class"],
                    "currency": remote["currency"],
                    "supports_shares": supports_share_allocation(remote["ticker"]),
                })

    # 4. Direct custom ticker fallback if the user typed a valid symbol code
    candidate = normalize_ticker(query)
    if candidate not in seen and is_valid_asset_ticker(candidate) and len(candidate) >= 1:
        meta = asset_metadata(candidate, b3_frame)
        if not asset_class or meta.get("asset_class") == asset_class:
            results.append({
                "ticker": meta["ticker"],
                "company": meta["company"],
                "legal_name": meta["legal_name"],
                "sector": meta["sector"],
                "asset_class": meta["asset_class"],
                "currency": meta["currency"],
                "supports_shares": supports_share_allocation(meta["ticker"]),
            })

    return clean_json({"items": results[:20], "query": query})


@catalog_router.get("/universe/liquidity")
def universe_liquidity():
    tickers = get_listed_stocks()["Ticker"].tolist()
    try:
        series = get_liquidity_by_ticker()
        liquidity = series.reindex(tickers)
    except Exception as exc:
        raise HTTPException(503, f"Não foi possível carregar a liquidez da Fundamentus: {exc}") from exc
    return clean_json({"source": "Fundamentus · liquidez média diária em 2 meses (R$)", "fetched_at": series.attrs.get("fetched_at"), "liquidity": liquidity.to_dict()})


@catalog_router.get("/universe/changes")
def universe_changes():
    tickers = tuple(get_listed_stocks()["Ticker"])
    try:
        result = get_universe_changes(tickers)
    except Exception as exc:
        raise HTTPException(503, f"Não foi possível carregar as variações do Yahoo Finance: {exc}") from exc
    return clean_json({"source": "Yahoo Finance · fechamento ajustado, com atraso", "unit": "fraction", **result})


@catalog_router.post("/universe/suggestions")
def suggestions(body: SuggestionsRequest):
    frame = get_listed_stocks()
    matching = frame if not body.sectors or "Todos" in body.sectors else frame[frame["Setor"].isin(body.sectors)]
    tickers = list(dict.fromkeys(get_sorted_tickers_by_liquidity(matching["Ticker"].tolist())))
    return {"tickers": tickers, "count": len(tickers), "autopick": tickers[:8], "source": "Fundamentus · ordenação por liquidez"}


@catalog_router.post("/fundamentus/refresh")
def refresh_fundamentus():
    removed = clear_fundamentus_cache()
    get_fii_market_data.clear()
    get_sorted_tickers_by_liquidity.clear()
    return {"removed": removed, "scope": "global Fundamentus; consenso Yahoo não alterado"}


@router.post("/analyze")
def analyze(body: HomeRequest, request: Request):
    ctx = session(request)
    tickers = validate_selection(ctx, body)
    generation = begin(ctx, "home", body.revision)
    try:
        universe_frame = get_listed_stocks()
        b3_set = set(universe_frame["Ticker"])
        b3_tickers = [ticker for ticker in tickers if ticker in b3_set and asset_metadata(ticker, universe_frame)["asset_class"] == "acao_br"]
        non_b3_tickers = [ticker for ticker in tickers if ticker not in set(b3_tickers)]
        frames = []
        covered_b3 = set()
        non_b3_raw = pd.DataFrame()
        if b3_tickers and non_b3_tickers:
            with ThreadPoolExecutor(max_workers=2) as pool:
                non_b3_future = pool.submit(_extra_fundamentals_frame, non_b3_tickers, {})
                try:
                    raw_b3 = get_fundamentus_data(b3_tickers)
                    if raw_b3 is not None and not raw_b3.empty:
                        frames.append(raw_b3)
                        covered_b3.update(normalize_detail(raw_b3).index)
                except OSError:
                    raise
                except Exception:
                    logger.warning("Fundamentus detail unavailable for %s", ", ".join(b3_tickers), exc_info=True)
                non_b3_raw = non_b3_future.result()
        elif b3_tickers:
            try:
                raw_b3 = get_fundamentus_data(b3_tickers)
                if raw_b3 is not None and not raw_b3.empty:
                    frames.append(raw_b3)
                    covered_b3.update(normalize_detail(raw_b3).index)
            except OSError:
                raise
            except Exception:
                logger.warning("Fundamentus detail unavailable for %s", ", ".join(b3_tickers), exc_info=True)
        elif non_b3_tickers:
            non_b3_raw = _extra_fundamentals_frame(non_b3_tickers, {})
        if not non_b3_raw.empty:
            frames.append(non_b3_raw)
        missing_b3 = [ticker for ticker in b3_tickers if ticker not in covered_b3]
        if missing_b3:
            fallback_b3 = {ticker: asset_metadata(ticker, universe_frame) for ticker in missing_b3}
            fallback_raw = _extra_fundamentals_frame(missing_b3, fallback_b3)
            if not fallback_raw.empty:
                frames.append(fallback_raw)
        raw = pd.concat(frames) if frames else pd.DataFrame()
        if raw is None or raw.empty:
            source = "A Fundamentus" if not non_b3_tickers and not missing_b3 else "A fonte de dados"
            raise HTTPException(503, f"{source} não retornou dados para os ativos selecionados. Tente novamente.")
        peers = get_sector_peers() if b3_tickers else pd.DataFrame()
        result = build_analysis(raw, peers, tickers, universe_frame)
        missing_history = [company["ticker"] for company in result["companies"]
                           if company["asset_class"] in EQUITY_CLASSES | {"reit"} and not company.get("history")]
        if missing_history:
            with ThreadPoolExecutor(max_workers=min(8, len(missing_history))) as pool:
                histories = dict(zip(missing_history, pool.map(
                    lambda ticker: get_equity_history(ticker, provider_symbol(ticker)), missing_history)))
            for company in result["companies"]:
                if company["ticker"] in histories:
                    company["history"] = histories[company["ticker"]]
        if set(b3_tickers) - covered_b3:
            clear_fundamentus_cache()
        skip_yahoo = {
            company["ticker"]
            for company in result["companies"]
            if company["asset_class"] in EQUITY_CLASSES | {"reit"} and company["source"] == "Nasdaq"
        }
        result["consensus"] = consensus(
            tickers,
            asset_classes={item["ticker"]: item["asset_class"] for item in result["companies"]},
            skip_yahoo=skip_yahoo,
        )
        result["id"] = uuid.uuid4().hex
        result["revision"] = body.revision
        export_fields = list(EXPORT_FIELDS)
        for column, default in (("Classe", "acao_br"), ("Moeda", "BRL"), ("Fonte", "Fundamentus")):
            if column in raw:
                raw[column] = raw[column].fillna(default)
        indicator_fields = {source for _, source, *_ in INDICATORS}
        applicable_fields = {source for key, source, *_ in INDICATORS
                             if any(key in company["indicators"] for company in result["companies"])}
        export_fields = [field for field in export_fields
                         if field not in indicator_fields or field in applicable_fields]
        if any(company["fixed_income"] for company in result["companies"]):
            export_fields += ["Remuneracao_modelo", "Taxa_modelo", "Unidade_taxa_modelo"]
        if "Tesouro" in raw and raw["Tesouro"].apply(lambda value: isinstance(value, dict) and bool(value)).any():
            treasury = raw["Tesouro"].apply(lambda value: value if isinstance(value, dict) else {})
            for key, column in TREASURY_EXPORT_FIELDS.items():
                raw[column] = treasury.map(lambda value, key=key: value.get(key))
            export_fields += list(TREASURY_EXPORT_FIELDS.values())
        raw_export = raw[[column for column in export_fields if column in raw]].copy()
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
        asset_classes = {item["ticker"]: item["asset_class"] for item in current["result"]["companies"]}
    generation = begin(ctx, "home-consensus", body.revision)
    result = consensus(tickers, refresh=True, asset_classes=asset_classes)
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
    prefix = "ativos" if "Classe" in saved["export"] else "fundamentus"
    return csv_response(saved["export"], f"{prefix}_{'+'.join(tickers)}_{dt.date.today().isoformat()}.csv")
