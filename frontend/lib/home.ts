import { api } from "./api";
import type { Metric, Column } from "../components/ui";
import type { AssetClass, Stock } from "../components/AppProvider";

export type Consensus = {rows: Record<string, unknown>[]; failed_tickers: string[]; has_targets: boolean; source: string};
export type SynthesisDetail = {label:string;value:number;unit:"percent"|"multiple";median:number;mean:number;observations:number;percentile:number;verdict:string;category:string};
export type PriceUnit = "money" | "points";
export type ContractInfo = {symbol:string|null;expiry:string|null;unit:string|null;open_interest:number|null};
export type FundLeverage = {net_debt_to_ebitda:number|null;debt_to_market_cap:number|null};
/** Fields published by Yahoo Finance or by Fundamentus' FII detail page; the unused ones stay null. */
export type FundInfo = {
  fund_family:string|null; category:string|null; legal_type:string|null;
  expense_ratio:number|null; total_assets:number|null; distribution_yield:number|null;
  beta:number|null; inception:string|null; nav_price:number|null;
  vacancy:number|null; cap_rate:number|null; ffo_yield:number|null; properties:number|null;
  properties_area_m2:number|null; units:number|null; segment:string|null; mandate:string|null;
  management:string|null; manager:string|null; nav_per_share:number|null; ffo_per_share:number|null;
  dividend_per_share:number|null; market_value:number|null; report_date:string|null;
  leverage:FundLeverage|null; underlying:string|null; underlying_ratio:number|null; source:string|null;
};
/** Official Tesouro Direto morning buy/sell prices and rates, in % a.a. */
export type TreasuryInfo = {buy_price:number|null;sell_price:number|null;buy_rate:number|null;sell_rate:number|null;rate_unit:string|null;maturity:string|null;quote_date:string|null;source:string};
export type FixedIncomeInfo = {label:string;rate:number|null;unit:string|null;indexer:string|null;index_rate:number|null;index_rate_unit:string|null;index_rate_date:string|null;index_rate_source:string|null;index_rate_fallback:boolean};
export type DividendYear = {year:number;total:number};
export type DividendHistory = {total_12m:number|null;last_payment:string|null;payments_12m:number|null;annual:DividendYear[]};
/** One annual period; monetary values are in the company's quote currency. */
export type StatementPeriod = {period_end:string;revenue:number|null;net_income:number|null;operating_income:number|null;operating_cash_flow:number|null;free_cash_flow:number|null;total_assets:number|null;total_equity:number|null};
export type EquityHistory = {available:boolean;currency:string|null;dividends:DividendHistory|null;statements:StatementPeriod[];fiscal_year_end:string|null;revenue_growth:number|null;earnings_growth:number|null;source:string;note:string|null};
export type Company = {
  ticker:string; available:boolean; indicators_available:boolean;
  asset_class:AssetClass; source:string|null;
  fixed_income:FixedIncomeInfo|null;
  identity:{company:string|null;sector:string|null;subsector:string|null};
  market:{price:number|null;low_52w:number|null;high_52w:number|null;volume_2m:number|null;market_cap:number|null;last_quote_date:string|null;currency:string;
    price_unit:PriceUnit;quote_pair:string|null;volume_24h:number|null;session_volume:number|null;volume_window:"24h"|"sessão"|null;contract:ContractInfo|null};
  fund:FundInfo|null;
  treasury:TreasuryInfo|null;
  history:EquityHistory|null;
  indicators:Record<string,number|null>;
  indicator_available:Record<string,boolean>;
  metric_groups:Record<string,(Metric & {classification:string|null})[]>;
  health:{metrics:Metric[];diagnostics:{status:"negative"|"warning"|"positive";text:string}[]};
  synthesis:{sector:string|null;categorias:Record<string,{percentil:number;veredicto:string;indicadores_validos:number}>;categorias_validas:number;indicadores_validos:number;favorable_categories:number;details:SynthesisDetail[]};
};
export type HomeResult = {id:string;revision:number;requested_tickers:string[];missing_tickers:string[];companies:Company[];health_summary:string;consensus:Consensus;comparison_indicators:{key:string;label:string;unit:"multiple"|"percent"}[]};
export type Suggestions = {tickers:string[];count:number;autopick:string[];source:string};
export const analyzeHome = (tickers:string[], revision:number) => api<HomeResult>("/api/home/analyze",{tickers,revision});
export const currentAnalysis = () => api<HomeResult|null>("/api/home/current");
export const refreshConsensus =(tickers:string[], revision:number) => api<Consensus>("/api/home/consensus/refresh",{tickers,revision});
export const suggestions = (sectors:string[]) => api<Suggestions>("/api/universe/suggestions",{sectors});
export const refreshFundamentus = () => api<{removed:number;scope:string}>("/api/fundamentus/refresh",{});
export const searchUniverse = (q: string, assetClass = "") => api<{ items: Omit<Stock, "liquidity">[]; query: string }>(`/api/universe/search?q=${encodeURIComponent(q)}${assetClass ? `&asset_class=${encodeURIComponent(assetClass)}` : ""}`);
/** Quote columns resolve their unit per row: indices are points, everything else is money. */
export const marketColumns:Column[] = [{key:"ticker",label:"Ticker"},{key:"price",label:"Cotação",unit:"money",unitKey:"price_unit"},{key:"unit_label",label:"Unidade"},{key:"low_52w",label:"Mín. 52 semanas",unit:"money",unitKey:"price_unit"},{key:"high_52w",label:"Máx. 52 semanas",unit:"money",unitKey:"price_unit"},{key:"volume_2m",label:"Vol. médio 2m",unit:"money",digits:0,compact:true},{key:"market_cap",label:"Valor de mercado",unit:"money",digits:0,compact:true},{key:"last_quote_date",label:"Última cotação"}];
export const globalColumns:Column[] = [{key:"ticker",label:"Ativo"},{key:"price",label:"Cotação",unit:"money",unitKey:"price_unit"},{key:"unit_label",label:"Unidade"},{key:"quote_pair",label:"Par de cotação"},{key:"volume",label:"Volume",unit:"number",digits:0,compact:true},{key:"volume_window",label:"Janela do volume"},{key:"contract_label",label:"Contrato futuro"},{key:"expiry",label:"Vencimento"},{key:"last_quote_date",label:"Última cotação"}];
/** Fund, FII and REIT detail. Percent fields arrive as percentage points; an unavailable field renders as a dash. */
export const fundColumns:Column[] = [{key:"ticker",label:"Ativo"},{key:"fund_family",label:"Gestora"},{key:"category",label:"Categoria / índice"},{key:"expense_ratio",label:"Taxa de administração",unit:"percent",digits:4},{key:"total_assets",label:"Patrimônio líquido",unit:"money",digits:0,compact:true},{key:"market_value",label:"Valor de mercado",unit:"money",digits:0,compact:true},{key:"nav_per_share",label:"VP/cota",unit:"money"},{key:"ffo_per_share",label:"FFO/cota",unit:"money"},{key:"distribution_yield",label:"Rendimento 12m",unit:"percent"},{key:"vacancy",label:"Vacância média",unit:"percent"},{key:"cap_rate",label:"Cap rate",unit:"percent"},{key:"ffo_yield",label:"FFO yield",unit:"percent"},{key:"properties",label:"Imóveis",unit:"integer"},{key:"segment",label:"Segmento"},{key:"mandate",label:"Mandato"},{key:"management",label:"Gestão"},{key:"manager",label:"Gestor"},{key:"beta",label:"Beta",unit:"number",digits:2},{key:"inception",label:"Início"},{key:"net_debt_to_ebitda",label:"Dív. líq./EBITDA",unit:"multiple",digits:2},{key:"debt_to_market_cap",label:"Dívida/valor de mercado",unit:"multiple",digits:2},{key:"underlying",label:"Ativo subjacente"},{key:"source",label:"Fonte"}];
export const treasuryColumns:Column[] = [{key:"ticker",label:"Título"},{key:"buy_price",label:"PU compra",unit:"money"},{key:"sell_price",label:"PU venda",unit:"money"},{key:"buy_rate",label:"Taxa compra",unit:"percent"},{key:"sell_rate",label:"Taxa venda",unit:"percent"},{key:"maturity",label:"Vencimento"},{key:"quote_date",label:"Data-base"},{key:"source",label:"Fonte"}];
export const consensusColumns:Column[] = [{key:"ticker",label:"Ticker"},{key:"company",label:"Empresa"},{key:"currency",label:"Moeda"},{key:"price",label:"Cotação atual",unit:"money"},{key:"target_low",label:"Alvo mínimo",unit:"money"},{key:"target_median",label:"Alvo mediano",unit:"money"},{key:"target_mean",label:"Alvo médio",unit:"money"},{key:"target_high",label:"Alvo máximo",unit:"money"},{key:"upside",label:"Potencial",unit:"percent",digits:1},{key:"analyst_count",label:"Opiniões",unit:"integer"},{key:"recommendation",label:"Recomendação"}];
/** Curated shortcut tickers only; sector labels come from the official B3 list in `stocks`. */
export const quickTickers = ["WEGE3","PETR4","VALE3","ITUB4","RENT3","ABEV3","EGIE3","RADL3"];
export const quickTickersByClass: Record<AssetClass, string[]> = {
  acao_br: ["WEGE3", "PETR4", "VALE3", "ITUB4", "BBAS3", "RENT3", "ABEV3", "EGIE3", "RADL3", "BPAC11", "PRIO3", "SUZB3"],
  fii: ["HGLG11", "KNRI11", "MXRF11", "XPML11", "VISC11", "BTLG11", "KNCR11", "KNIP11", "HGRU11", "TRXF11", "BCFF11", "XPLG11"],
  acao_us: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "JPM", "V", "KO"],
  reit: ["O", "PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "DLR", "VICI", "AVB", "WPC"],
  titulo: ["CDI", "TESOURO-SELIC", "TESOURO-IPCA-2035", "TESOURO-PRE", "CDB-110CDI", "IPCA+6", "IMAB11", "B5P211", "TLT", "SHY", "BND", "SGOV"],
  etf: ["BOVA11", "IVVB11", "SMAL11", "SPY", "VOO", "QQQ", "VT", "SCHD", "GLD", "AAPL34", "NVDC34", "VNQ"],
  outros: ["BTC-USD", "ETH-USD", "SOL-USD", "GC=F", "SI=F", "BZ=F", "USDBRL=X"],
};
/** Series colors from the brand chart palette; mirrors the CHART_* constants in utils/charts.py. */
export const comparisonColors = ["#8b93ff","#4cc9f0","#f5b544","#e879c6","#3ecf8e","#f2757a","#9aa2b4"];
export const quickSectors = [["Bancos","Intermediários Financeiros"],["Petróleo e Gás","Petróleo, Gás e Biocombustíveis"],["Mineração","Mineração"],["Energia Elétrica","Energia Elétrica"],["Tecnologia","Programas e Serviços"],["Bebidas","Bebidas"],["Saúde","Serv.Méd.Hospit. Análises e Diagnósticos"],["Varejo","Comércio"]];
/** Yahoo adjusted-close changes for the whole list, as fractions; null where a stock missed one of the sessions. */
export type UniverseChanges = {source:string;unit:"fraction";session:string;fetched_at:string;changes:Record<string,{day:number|null;week:number|null}>};
export const universeChanges = () => api<UniverseChanges>("/api/universe/changes");
