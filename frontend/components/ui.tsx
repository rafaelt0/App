"use client";

import Link from "next/link";
import { useId, useMemo, useState, type ReactNode } from "react";
import { download } from "../lib/api";
import type { Stock } from "./AppProvider";

export type Unit = "money" | "fraction" | "percent" | "ratio" | "multiple" | "number" | "integer" | "text";
export type Signal = "positive" | "negative" | "warning" | "neutral" | "info";
export type Metric = { label: string; value: number | string | null; unit?: Unit; currency?: string; digits?: number; status?: Signal; help?: string };
export type Column = { key: string; label: string; unit?: Unit; currency?: string; digits?: number };

export function formatValue(value: unknown, unit: Unit = "number", digits = 2, currency = "BRL"): string {
  if (value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value))) return "-";
  if (typeof value !== "number" || unit === "text") return String(value);
  const options: Intl.NumberFormatOptions = { minimumFractionDigits: unit === "integer" ? 0 : digits, maximumFractionDigits: unit === "integer" ? 0 : digits };
  if (unit === "money") { options.style = "currency"; options.currency = currency; }
  if (unit === "fraction") options.style = "percent";
  return new Intl.NumberFormat("pt-BR", options).format(value) + (unit === "percent" ? "%" : unit === "multiple" ? "×" : "");
}

const iconAssets: Record<string, string> = {
  home: "/icons/home.svg",
  portfolio: "/icons/portfolio.svg",
  simulation: "/icons/simulation.svg",
  news: "/icons/news.svg",
  filter: "/icons/screener.svg",
  valuation: "/icons/valuation.svg",
};
const paths: Record<string, string> = {
  chart: "M3 20V4m0 16h18M6 16V9m6 7V5m6 11v-6",
  menu: "M3 6h18M3 12h18M3 18h18",
  "panel-close": "M4 5h16v14H4zM9 5v14m7-9-2 2 2 2",
  "panel-open": "M4 5h16v14H4zM9 5v14m5-9 2 2-2 2",
};
export function Icon({ name = "chart" }: { name?: string }) {
  if (Object.hasOwn(iconAssets, name)) return <span className="app-icon" aria-hidden="true" style={{ maskImage: `url("${iconAssets[name]}")` }} />;
  return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] ?? paths.chart} /></svg>;
}
export function PageHeader({ title, description, icon = "chart" }: { title: string; description: string; icon?: string }) {
  return <header className="page-hero"><span className="page-hero-icon"><Icon name={icon} /></span><div className="page-hero-content"><h1 className="page-hero-title">{title}</h1><p className="page-hero-subtitle">{description}</p></div></header>;
}
export function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="analysis-section"><h2 className="ui-section-heading">{title}</h2>{children}</section>;
}
export function Metrics({ items }: { items: Metric[] }) {
  return <div className="mcard-grid">{items.map((item, index) => <div className={`mcard${item.unit === "money" ? " mcard-money" : ""}`} key={`${item.label}-${index}`}><div className="mcard-label">{item.label}</div><div className={`mcard-value${item.status ? ` is-${item.status}` : ""}`}>{formatValue(item.value, item.unit, item.digits, item.currency)}</div>{item.help && <p className="metric-help">{item.help}</p>}</div>)}</div>;
}
export function Table<T extends object>({ columns, rows, onSelect, selectedKey, rowKey = "ticker" }: { columns: Column[]; rows: T[]; onSelect?: (row: T) => void; selectedKey?: string; rowKey?: string }) {
  return <div className="table-scroll" tabIndex={0} aria-label="Tabela de dados"><table><thead><tr>{onSelect && <th scope="col">Selecionar</th>}{columns.map(column => <th key={column.key} scope="col" className={column.unit && column.unit !== "text" ? "numeric-cell" : undefined}>{column.label}</th>)}</tr></thead><tbody>{rows.map((row, index) => {
    const values = row as Record<string, unknown>;
    const key = String(values[rowKey] ?? index);
    return <tr key={key} className={selectedKey === key ? "selected-row" : undefined}>{onSelect && <td><button aria-label={`Selecionar ${key}`} aria-pressed={selectedKey === key} onClick={() => onSelect(row)}>{selectedKey === key ? "Selecionado" : "Selecionar"}</button></td>}{columns.map(column => <td key={column.key} className={column.unit && column.unit !== "text" ? "numeric-cell" : undefined}>{formatValue(values[column.key], column.unit, column.digits, column.currency ?? (typeof values.currency === "string" ? values.currency : "BRL"))}</td>)}</tr>;
  })}</tbody></table></div>;
}
export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: Signal }) {
  return <div className={`notice is-${tone}`} role={tone === "negative" ? "alert" : undefined}>{children}</div>;
}
export function EmptyState({ title, children, href, action }: { title: string; children: ReactNode; href?: string; action?: string }) {
  return <section className="empty-state-card"><Icon name="chart" /><h2 className="empty-state-title">{title}</h2><div className="empty-state-message">{children}</div>{href && <Link className="button-link" href={href}>{action ?? "Configurar portfólio"}</Link>}</section>;
}
export function Loading({ children }: { children?: ReactNode }) {
  return <div className="loading-container" role="status" aria-live="polite"><span className="loading-spinner" aria-hidden="true" /><p>{children ?? "Carregando dados para a análise."}</p></div>;
}
export function TickerPicker({ value, onChange, stocks, max = 20, label = "Escolha ações para analisar" }: { value: string[]; onChange: (tickers: string[]) => void | Promise<void>; stocks: Stock[]; max?: number; label?: string }) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [active, setActive] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const options = useMemo(() => {
    const needle = query.toLocaleLowerCase("pt-BR");
    return stocks.filter(stock => !value.includes(stock.ticker) && `${stock.ticker} ${stock.company} ${stock.sector}`.toLocaleLowerCase("pt-BR").includes(needle)).slice(0, 40);
  }, [query, stocks, value]);
  async function change(next: string[]) {
    setBusy(true); setError(null);
    try { await onChange(next); setQuery(""); setActive(0); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }
  const expanded = focused && options.length > 0 && value.length < max;
  return <div className="ticker-picker"><label htmlFor={id}>{label}</label><div className="ticker-chips">{value.map(ticker => <span className="ticker-chip" key={ticker}>{ticker}<button type="button" disabled={busy} aria-label={`Remover ${ticker}`} onClick={() => change(value.filter(item => item !== ticker))}>×</button></span>)}</div>
    <input id={id} role="combobox" type="search" autoComplete="off" placeholder="Digite ticker ou nome da empresa..." value={query} disabled={busy || value.length >= max} aria-autocomplete="list" aria-expanded={expanded} aria-controls={`${id}-options`} aria-activedescendant={expanded ? `${id}-option-${active}` : undefined} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)} onChange={event => { setQuery(event.target.value); setActive(0); }} onKeyDown={event => {
      if (event.key === "Escape") setFocused(false);
      if (event.key === "ArrowDown") { event.preventDefault(); setActive(index => Math.min(index + 1, options.length - 1)); }
      if (event.key === "ArrowUp") { event.preventDefault(); setActive(index => Math.max(index - 1, 0)); }
      if (event.key === "Enter" && expanded && options[active]) { event.preventDefault(); void change([...value, options[active].ticker]); }
    }} />
    {expanded && <ul className="ticker-options" id={`${id}-options`} role="listbox">{options.map((stock, index) => <li key={stock.ticker} id={`${id}-option-${index}`} role="option" aria-selected={active === index}><button type="button" onMouseDown={event => event.preventDefault()} onClick={() => change([...value, stock.ticker])}><strong>{stock.ticker}</strong><span>{[stock.company, stock.sector].filter(Boolean).join(" · ") || "Setor não informado"}</span></button></li>)}</ul>}
    <p className="caption">{value.length}/{max} ativos selecionados.</p>{error && <Notice tone="negative">{error}</Notice>}
  </div>;
}
export function ExportButton({ path, children }: { path: string; children: ReactNode }) {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  async function exportFile() {
    setBusy(true); setError(null);
    try { await download(path); } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }
  return <><button type="button" disabled={busy} onClick={exportFile}>{children}</button>{error && <Notice tone="negative">{error}</Notice>}</>;
}
