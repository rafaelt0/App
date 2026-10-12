import { api } from "./api";
import { formatCompactValue, formatValue } from "../components/ui";
import type { Column, Unit } from "../components/ui";
export type Criteria = {pl_min?:number;pl_max?:number;pl_min_exclusive?:boolean;roe_min?:number;dy_min?:number;c5y_min?:number;liq2m_min?:number};
export type Sort = {field:string;direction:"asc"|"desc";label?:string};
export type ScreenerColumn = Column & {help?:string|null};
export type Config = {presets:{name:string;description:string;criteria:Criteria}[];sorts:Sort[];columns:ScreenerColumn[];default_columns:string[]};
export type Controls = {preset:string;pl_enabled:boolean;roe_enabled:boolean;dy_enabled:boolean;c5y_enabled:boolean;liq2m_enabled:boolean;pl_min:number;pl_max:number;pl_min_exclusive:boolean;roe_min:number;dy_min:number;c5y_min:number;liq2m_min:number};
export type Result = {id:string;total_count:number;matched_count:number;active_filters:string[];fetched_at:string|null;age_minutes:number|null;metrics:{mean_dy:number|null;mean_roe:number|null};available_columns:ScreenerColumn[];default_columns:string[];rows:Record<string,unknown>[]};
export const defaults:Controls = {preset:"Explorar B3",pl_enabled:false,roe_enabled:false,dy_enabled:false,c5y_enabled:false,liq2m_enabled:true,pl_min:0,pl_max:15,pl_min_exclusive:false,roe_min:12,dy_min:5,c5y_min:10,liq2m_min:1000000};
export function applyPreset(current:Controls, name:string, criteria:Criteria):Controls {
  if(name === "Personalizado") return {...current,preset:name};
  const next = {...defaults,preset:name,pl_enabled:false,roe_enabled:false,dy_enabled:false,c5y_enabled:false,liq2m_enabled:false,pl_min_exclusive:criteria.pl_min_exclusive ?? false};
  for(const [key,value] of Object.entries(criteria)) {
    if(key === "pl_min_exclusive") continue;
    if(key === "pl_min" || key === "pl_max") { next.pl_enabled=true; next[key]=Number(value); }
    else if(key === "roe_min") {next.roe_enabled=true;next.roe_min=Number(value)*100;}
    else if(key === "dy_min") {next.dy_enabled=true;next.dy_min=Number(value)*100;}
    else if(key === "c5y_min") {next.c5y_enabled=true;next.c5y_min=Number(value)*100;}
    else if(key === "liq2m_min") {next.liq2m_enabled=true;next.liq2m_min=Number(value);}
  }
  return next;
}
export function editControl<K extends keyof Controls>(controls:Controls,key:K,value:Controls[K]):Controls {
  return {...controls,[key]:value,preset:"Personalizado",pl_min_exclusive:key === "pl_min" ? false : controls.pl_min_exclusive};
}
export function canonicalCriteria(c:Controls):Criteria {
  return {...(c.pl_enabled ? {pl_min:c.pl_min,pl_max:c.pl_max,pl_min_exclusive:c.pl_min_exclusive}:{}),...(c.roe_enabled ? {roe_min:c.roe_min/100}:{}),...(c.dy_enabled ? {dy_min:c.dy_min/100}:{}),...(c.c5y_enabled ? {c5y_min:c.c5y_min/100}:{}),...(c.liq2m_enabled ? {liq2m_min:c.liq2m_min}: {})};
}
export type CriterionKey = "pl"|"roe"|"c5y"|"dy"|"liq2m";
export type CriterionDef = {key:CriterionKey;group:string;toggle:string;enabled:"pl_enabled"|"roe_enabled"|"c5y_enabled"|"dy_enabled"|"liq2m_enabled";chip:(c:Controls)=>string};
const amount=(value:number,unit:Unit,digits=Number.isInteger(value) ? 0:1)=>!Number.isFinite(value) ? "inválido":unit === "money" ? formatCompactValue(value,unit,1):formatValue(value,unit,digits);
export const criterionDefs:CriterionDef[] = [
  {key:"pl",group:"Valoração",toggle:"Aplicar faixa de P/L",enabled:"pl_enabled",chip:c=>c.pl_min_exclusive ? `P/L > ${amount(c.pl_min,"number")} e ≤ ${amount(c.pl_max,"multiple")}`:`P/L ${amount(c.pl_min,"number")}–${amount(c.pl_max,"multiple")}`},
  {key:"roe",group:"Qualidade",toggle:"Aplicar ROE mínimo",enabled:"roe_enabled",chip:c=>`ROE ≥ ${amount(c.roe_min,"percent")}`},
  {key:"c5y",group:"Crescimento",toggle:"Aplicar crescimento da receita (5 anos)",enabled:"c5y_enabled",chip:c=>`Cresc. receita 5a ≥ ${amount(c.c5y_min,"percent")}`},
  {key:"dy",group:"Dividendos",toggle:"Aplicar dividend yield mínimo",enabled:"dy_enabled",chip:c=>`DY ≥ ${amount(c.dy_min,"percent")}`},
  {key:"liq2m",group:"Liquidez",toggle:"Aplicar liquidez mínima",enabled:"liq2m_enabled",chip:c=>`Liquidez 2m ≥ ${amount(c.liq2m_min,"money",0)}`},
];
export const activeCriteria = (c:Controls) => criterionDefs.filter(def=>c[def.enabled]).map(def=>({key:def.key,enabled:def.enabled,label:def.chip(c),title:def.key === "liq2m" && Number.isFinite(c.liq2m_min) ? formatValue(c.liq2m_min,"money",2):undefined}));
export const sortKey = (sort:Pick<Sort,"field"|"direction">) => `${sort.field}:${sort.direction}`;
export const screenerConfig = () => api<Config>("/api/screener/config");
export const queryScreener = (criteria:Criteria,sort:Sort) => api<Result>("/api/screener/query",{criteria,sort:{field:sort.field,direction:sort.direction}});
