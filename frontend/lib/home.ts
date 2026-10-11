import { api } from "./api";
import type { Metric, Column } from "../components/ui";

export type Consensus = {rows: Record<string, unknown>[]; failed_tickers: string[]; has_targets: boolean; source: string};
export type SynthesisDetail = {label:string;value:number;unit:"percent"|"multiple";median:number;mean:number;observations:number;percentile:number;verdict:string;category:string};
export type Company = {
  ticker:string; available:boolean; indicators_available:boolean;
  identity:{company:string|null;sector:string|null;subsector:string|null};
  market:{price:number|null;low_52w:number|null;high_52w:number|null;volume_2m:number|null;market_cap:number|null;last_quote_date:string|null;currency:string};
  indicators:Record<string,number|null>;
  indicator_available:Record<string,boolean>;
  metric_groups:Record<string,(Metric & {classification:string|null})[]>;
  health:{metrics:Metric[];diagnostics:{status:"negative"|"warning"|"positive";text:string}[]};
  synthesis:{sector:string|null;categorias:Record<string,{percentil:number;veredicto:string;indicadores_validos:number}>;categorias_validas:number;indicadores_validos:number;favorable_categories:number;details:SynthesisDetail[]};
};
export type HomeResult = {id:string;revision:number;requested_tickers:string[];missing_tickers:string[];companies:Company[];consensus:Consensus;comparison_indicators:{key:string;label:string;unit:"multiple"|"percent"}[]};
export type Suggestions = {tickers:string[];count:number;autopick:string[];source:string};
export const analyzeHome = (tickers:string[], revision:number) => api<HomeResult>("/api/home/analyze",{tickers,revision});
export const refreshConsensus = (tickers:string[], revision:number) => api<Consensus>("/api/home/consensus/refresh",{tickers,revision});
export const suggestions = (sectors:string[]) => api<Suggestions>("/api/universe/suggestions",{sectors});
export const refreshFundamentus = () => api<{removed:number;scope:string}>("/api/fundamentus/refresh",{});
export const identityColumns:Column[] = [{key:"ticker",label:"Ticker"},{key:"company",label:"Empresa"},{key:"sector",label:"Setor"},{key:"subsector",label:"Subsetor"}];
export const marketColumns:Column[] = [{key:"ticker",label:"Ticker"},{key:"price",label:"Cotação",unit:"money"},{key:"low_52w",label:"Mín. 52 semanas",unit:"money"},{key:"high_52w",label:"Máx. 52 semanas",unit:"money"},{key:"volume_2m",label:"Vol. médio 2m",unit:"integer"},{key:"market_cap",label:"Valor de mercado",unit:"money",digits:0},{key:"last_quote_date",label:"Última cotação"}];
export const consensusColumns:Column[] = [{key:"ticker",label:"Ticker"},{key:"company",label:"Empresa"},{key:"currency",label:"Moeda"},{key:"price",label:"Cotação atual",unit:"money"},{key:"target_low",label:"Alvo mínimo",unit:"money"},{key:"target_median",label:"Alvo mediano",unit:"money"},{key:"target_mean",label:"Alvo médio",unit:"money"},{key:"target_high",label:"Alvo máximo",unit:"money"},{key:"upside",label:"Potencial",unit:"percent",digits:1},{key:"analyst_count",label:"Opiniões",unit:"integer"},{key:"recommendation",label:"Recomendação"}];
export const quickTickers = [["WEGE3","Máquinas"],["PETR4","Petróleo"],["VALE3","Mineração"],["ITUB4","Banco"],["RENT3","Locação"],["ABEV3","Bebidas"],["EGIE3","Energia"],["RADL3","Farmácia"]];
export const quickSectors = [["Bancos","Intermediários Financeiros"],["Petróleo e Gás","Petróleo, Gás e Biocombustíveis"],["Mineração","Mineração"],["Energia Elétrica","Energia Elétrica"],["Tecnologia","Programas e Serviços"],["Bebidas","Bebidas"],["Saúde","Serv.Méd.Hospit. Análises e Diagnósticos"],["Varejo","Comércio"]];
