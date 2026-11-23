import { api } from "./api";
import type { Metric, Column } from "../components/ui";
import type { AssetClass, Stock } from "../components/AppProvider";

export type Consensus = {rows: Record<string, unknown>[]; failed_tickers: string[]; has_targets: boolean; source: string};
export type SynthesisDetail = {label:string;value:number;unit:"percent"|"multiple";median:number;mean:number;observations:number;percentile:number;verdict:string;category:string};
export type Company = {
  ticker:string; available:boolean; indicators_available:boolean;
  asset_class:AssetClass; source:string|null;
  fixed_income:{label:string;rate:number|null;unit:string|null}|null;
  identity:{company:string|null;sector:string|null;subsector:string|null};
  market:{price:number|null;low_52w:number|null;high_52w:number|null;volume_2m:number|null;market_cap:number|null;last_quote_date:string|null;currency:string};
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
export const marketColumns:Column[] = [{key:"ticker",label:"Ticker"},{key:"price",label:"Cotação",unit:"money"},{key:"low_52w",label:"Mín. 52 semanas",unit:"money"},{key:"high_52w",label:"Máx. 52 semanas",unit:"money"},{key:"volume_2m",label:"Vol. médio 2m",unit:"money",digits:0,compact:true},{key:"market_cap",label:"Valor de mercado",unit:"money",digits:0,compact:true},{key:"last_quote_date",label:"Última cotação"}];
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
