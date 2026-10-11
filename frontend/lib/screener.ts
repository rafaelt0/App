import { api } from "./api";
import type { Column } from "../components/ui";
export type Criteria = {pl_min?:number;pl_max?:number;pl_min_exclusive?:boolean;roe_min?:number;dy_min?:number;c5y_min?:number;liq2m_min?:number};
export type Sort = {field:"liq2m"|"pl"|"roe"|"dy";direction:"asc"|"desc";label?:string};
export type Config = {presets:{name:string;description:string;criteria:Criteria}[];sorts:Sort[];columns:Column[];default_columns:string[]};
export type Controls = {preset:string;pl_enabled:boolean;roe_enabled:boolean;dy_enabled:boolean;c5y_enabled:boolean;liq2m_enabled:boolean;pl_min:number;pl_max:number;pl_min_exclusive:boolean;roe_min:number;dy_min:number;c5y_min:number;liq2m_min:number};
export type Result = {id:string;total_count:number;matched_count:number;active_filters:string[];fetched_at:string|null;age_minutes:number|null;metrics:{mean_dy:number|null;mean_roe:number|null};available_columns:Column[];default_columns:string[];rows:Record<string,unknown>[]};
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
export const screenerConfig = () => api<Config>("/api/screener/config");
export const queryScreener = (criteria:Criteria,sort:Sort) => api<Result>("/api/screener/query",{criteria,sort:{field:sort.field,direction:sort.direction}});
