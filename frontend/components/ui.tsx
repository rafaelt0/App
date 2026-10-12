"use client";

import Link from "next/link";
import { useId, useMemo, useState, type ReactNode } from "react";
import { download } from "../lib/api";
import type { Stock } from "./AppProvider";

export type Unit = "money" | "fraction" | "percent" | "ratio" | "multiple" | "number" | "integer" | "text";
export type Signal = "positive" | "negative" | "warning" | "neutral" | "info";
export type Metric = { label: string; value: number | string | null; unit?: Unit; currency?: string; digits?: number; status?: Signal; help?: string; trend?: number[]; meter?: number };
export type Column = { key: string; label: string; unit?: Unit; currency?: string; digits?: number; signed?: boolean };

/** The one representation for a missing or non-finite value, in tables, metrics and text. */
export const missingValue = "—";
const minus = "−";
function localize(value: number, options: Intl.NumberFormatOptions): string {
  return new Intl.NumberFormat("pt-BR", options).format(value).replace(/-/g, minus);
}

export function formatValue(value: unknown, unit: Unit = "number", digits = 2, currency = "BRL", signed = false): string {
  if (value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value))) return missingValue;
  if (typeof value !== "number" || unit === "text") return String(value);
  const options: Intl.NumberFormatOptions = { signDisplay: signed ? "exceptZero" : "negative", minimumFractionDigits: unit === "integer" ? 0 : digits, maximumFractionDigits: unit === "integer" ? 0 : digits };
  if (unit === "money") { options.style = "currency"; options.currency = currency; }
  if (unit === "fraction") options.style = "percent";
  return localize(value, options) + (unit === "percent" ? "%" : unit === "multiple" ? "×" : "");
}

/** Fraction shown as a percent with one explicit precision; use it wherever a share or return is printed. */
export function formatPercent(value: unknown, digits = 1): string {
  return formatValue(value, "fraction", digits);
}

const brasiliaZone = "America/Sao_Paulo";
const brasiliaDate = new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: brasiliaZone });
const brasiliaTime = new Intl.DateTimeFormat("pt-BR", { hour: "2-digit", minute: "2-digit", timeZone: brasiliaZone });
/** Brasília wall-clock time with the zone spelled out, so no timestamp depends on the viewer's clock. */
export function formatBrasilia(value: string, withDate = true): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return missingValue;
  return `${(withDate ? brasiliaDate : brasiliaTime).format(date)} (horário de Brasília)`;
}
/** Age of a timestamp in short pt-BR text; the caller decides whether the age matters. */
export function relativeAge(value: string, now = Date.now()): string {
  const minutes = Math.round((now - Date.parse(value)) / 60000);
  if (!Number.isFinite(minutes)) return missingValue;
  if (minutes < 1) return "agora";
  if (minutes < 60) return `há ${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `há ${hours} h`;
  const days = Math.round(hours / 24);
  return `há ${days} ${days === 1 ? "dia" : "dias"}`;
}

/** Same text as formatValue, with the currency or unit symbol set apart so large figures read first. */
export function ValueText({ value, unit = "number", digits = 2, currency = "BRL" }: { value: unknown; unit?: Unit; digits?: number; currency?: string }) {
  const text = formatValue(value, unit, digits, currency);
  if (typeof value !== "number" || !Number.isFinite(value) || unit === "text") return <>{text}</>;
  if (unit === "money") {
    const parts = new Intl.NumberFormat("pt-BR", { style: "currency", currency, signDisplay: "negative", minimumFractionDigits: digits, maximumFractionDigits: digits }).formatToParts(value);
    const sign = parts.filter(part => part.type === "minusSign").map(() => minus).join("");
    const symbol = parts.filter(part => part.type === "currency").map(part => part.value).join("");
    const body = parts.filter(part => !["minusSign", "currency", "literal"].includes(part.type)).map(part => part.value).join("");
    return <>{sign}<span className="value-unit">{symbol}</span>{body}</>;
  }
  const suffix = unit === "fraction" ? "%" : unit === "percent" ? "%" : unit === "multiple" ? "×" : "";
  if (!suffix || !text.endsWith(suffix)) return <>{text}</>;
  return <>{text.slice(0, -suffix.length).trim()}<span className="value-unit is-suffix">{suffix}</span></>;
}

const statusGlyphs: Record<Signal, string> = { positive: "✓", negative: "✕", warning: "!", neutral: "•", info: "i" };
/** Decorative trend line for a value already shown as text; the series comes from the API unchanged. */
export function Sparkline({ values, tone = "info" }: { values: number[]; tone?: Signal }) {
  const points = values.filter(Number.isFinite);
  if (points.length < 2) return null;
  const min = Math.min(...points), max = Math.max(...points), span = max - min || 1;
  const coords = points.map((point, index) => [index / (points.length - 1) * 100, 34 - (point - min) / span * 30]);
  const line = coords.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const [lastX, lastY] = coords[coords.length - 1];
  return <svg className={`mcard-spark is-${tone}`} viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true" focusable="false">
    <polygon className="spark-area" points={`0,36 ${line} 100,36`} />
    <polyline className="spark-line" points={line} vectorEffect="non-scaling-stroke" />
    <line className="spark-end" x1={lastX} x2={lastX} y1={lastY} y2={lastY} vectorEffect="non-scaling-stroke" />
  </svg>;
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
  "arrow-right": "M5 12h14m-5-5 5 5-5 5",
  search: "M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  info: "M12 11v6m0-10h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  alert: "M12 8v5m0 4h.01M10 3 2 18a2 2 0 0 0 2 3h16a2 2 0 0 0 2-3L14 3a2 2 0 0 0-4 0",
  check: "M8 12l3 3 5-6M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
};
const noticeIcons: Record<Signal, string> = { positive: "check", negative: "alert", warning: "alert", neutral: "info", info: "info" };
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
  return <div className="mcard-grid">{items.map((item, index) => {
    const pill = !!item.status && !!item.help && item.help.length <= 24 && !item.help.includes(".");
    return <div className={`mcard${item.unit === "money" ? " mcard-money" : ""}`} key={`${item.label}-${index}`}>
      <div className="mcard-label"><span>{item.label}</span>{pill && <span className={`mcard-status is-${item.status}`}><span aria-hidden="true">{statusGlyphs[item.status!]}</span>{item.help}</span>}</div>
      <div className={`mcard-value${item.status ? ` is-${item.status}` : ""}`}><ValueText value={item.value} unit={item.unit} digits={item.digits} currency={item.currency} /></div>
      {item.trend && <Sparkline values={item.trend} tone={item.status ?? "info"} />}
      {typeof item.meter === "number" && Number.isFinite(item.meter) && <span className="mcard-meter" aria-hidden="true"><span style={{ width: `${Math.min(1, Math.max(0, item.meter)) * 100}%` }} /></span>}
      {item.help && !pill && <p className="metric-help">{item.help}</p>}
    </div>;
  })}</div>;
}
const isNumericColumn = (column: Column) => !!column.signed || (!!column.unit && column.unit !== "text");
/** Missing values sort last in both directions; numbers compare numerically, text in pt-BR order. */
function compareRows(a: object, b: object, key: string, direction: 1 | -1): number {
  const x = (a as Record<string, unknown>)[key], y = (b as Record<string, unknown>)[key];
  const missingX = x === null || x === undefined || (typeof x === "number" && !Number.isFinite(x));
  const missingY = y === null || y === undefined || (typeof y === "number" && !Number.isFinite(y));
  if (missingX || missingY) return Number(missingX) - Number(missingY);
  if (typeof x === "number" && typeof y === "number") return (x - y) * direction;
  return String(x).localeCompare(String(y), "pt-BR") * direction;
}
export function Table<T extends object>({ columns, rows, onSelect, selectedKey, rowKey = "ticker", caption, sortable = false }: { columns: Column[]; rows: T[]; onSelect?: (row: T) => void; selectedKey?: string; rowKey?: string; caption?: string; sortable?: boolean }) {
  const [sort, setSort] = useState<{ key: string; direction: 1 | -1 } | null>(null);
  const ordered = useMemo(() => sort ? [...rows].sort((a, b) => compareRows(a, b, sort.key, sort.direction)) : rows, [rows, sort]);
  const toggle = (key: string) => setSort(current => current?.key === key ? { key, direction: current.direction === 1 ? -1 : 1 } : { key, direction: 1 });
  return <div className="table-scroll" tabIndex={0}><table aria-label={caption ?? "Tabela de dados"}><thead><tr>{onSelect && <th scope="col">Selecionar</th>}{columns.map(column => {
    const numeric = isNumericColumn(column);
    const direction = sort?.key === column.key ? sort.direction : undefined;
    return <th key={column.key} scope="col" className={numeric ? "numeric-cell" : undefined} aria-sort={direction === undefined ? undefined : direction === 1 ? "ascending" : "descending"}>
      {sortable && numeric ? <button type="button" className="table-sort" onClick={() => toggle(column.key)}>{column.label}<span aria-hidden="true">{direction === undefined ? "↕" : direction === 1 ? "↑" : "↓"}</span></button> : column.label}
    </th>;
  })}</tr></thead><tbody>{ordered.map((row, index) => {
    const values = row as Record<string, unknown>;
    const key = String(values[rowKey] ?? index);
    return <tr key={key} className={selectedKey === key ? "selected-row" : undefined}>{onSelect && <td><button aria-label={`Selecionar ${key}`} aria-pressed={selectedKey === key} onClick={() => onSelect(row)}>{selectedKey === key ? "Selecionado" : "Selecionar"}</button></td>}{columns.map(column => <td key={column.key} className={isNumericColumn(column) ? "numeric-cell" : undefined}>{formatValue(values[column.key], column.unit, column.digits, column.currency ?? (typeof values.currency === "string" ? values.currency : "BRL"), column.signed)}</td>)}</tr>;
  })}</tbody></table></div>;
}
export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: Signal }) {
  return <div className={`notice is-${tone}`} role={tone === "negative" ? "alert" : undefined}><span className="notice-icon"><Icon name={noticeIcons[tone]} /></span><div className="notice-content">{children}</div></div>;
}
export function EmptyState({ title, children, href, action, icon = "chart" }: { title: string; children: ReactNode; href?: string; action?: string; icon?: string }) {
  return <section className="empty-state-card"><span className="empty-state-icon"><Icon name={icon} /></span><h2 className="empty-state-title">{title}</h2><div className="empty-state-message">{children}</div>{href && <Link className="button-link primary" href={href}>{action ?? "Configurar portfólio"}<Icon name="arrow-right" /></Link>}</section>;
}
export function Loading({ children }: { children?: ReactNode }) {
  return <div className="loading-container" role="status" aria-live="polite"><span className="loading-spinner" aria-hidden="true" /><p>{children ?? "Carregando dados para a análise."}</p><span className="loading-track" aria-hidden="true" /></div>;
}
export function TickerPicker({ value, onChange, stocks, max = 20, label = "Escolha ações para analisar" }: { value: string[]; onChange: (tickers: string[]) => void | Promise<void>; stocks: Stock[]; max?: number; label?: string }) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const options = useMemo(() => {
    const needle = query.toLocaleLowerCase("pt-BR");
    return stocks.filter(stock => !value.includes(stock.ticker) && `${stock.ticker} ${stock.company} ${stock.sector}`.toLocaleLowerCase("pt-BR").includes(needle)).slice(0, 40);
  }, [query, stocks, value]);
  async function change(next: string[]) {
    if (busy) return;
    setBusy(true); setError(null);
    try { await onChange(next); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }
  function pick(ticker: string) {
    if (busy) return;
    setQuery(""); setActive(0); setOpen(false);
    void change([...value, ticker]);
  }
  const expanded = open && options.length > 0 && value.length < max;
  return <div className="ticker-picker"><label htmlFor={id}>{label}</label><div className="ticker-chips">{value.map(ticker => <span className="ticker-chip" key={ticker}>{ticker}<button type="button" disabled={busy} aria-label={`Remover ${ticker}`} onClick={() => change(value.filter(item => item !== ticker))}>×</button></span>)}</div>
    <div className="ticker-search"><Icon name="search" /><input id={id} role="combobox" type="search" autoComplete="off" placeholder="Digite ticker ou nome da empresa..." value={query} disabled={value.length >= max} aria-busy={busy} aria-autocomplete="list" aria-expanded={expanded} aria-controls={`${id}-options`} aria-activedescendant={expanded ? `${id}-option-${active}` : undefined} onFocus={() => setOpen(true)} onClick={() => setOpen(true)} onBlur={() => setOpen(false)} onChange={event => { setQuery(event.target.value); setActive(0); setOpen(true); }} onKeyDown={event => {
      if (event.key === "Escape") { event.preventDefault(); if (!expanded) setQuery(""); setOpen(false); }
      if (event.key === "ArrowDown") { event.preventDefault(); if (!expanded) { setOpen(true); setActive(0); } else setActive(index => Math.min(index + 1, options.length - 1)); }
      if (event.key === "ArrowUp") { event.preventDefault(); setActive(index => Math.max(index - 1, 0)); }
      if (event.key === "Enter" && expanded && options[active]) { event.preventDefault(); pick(options[active].ticker); }
    }} /></div>
    {expanded && <ul className="ticker-options" id={`${id}-options`} role="listbox">{options.map((stock, index) => <li key={stock.ticker} id={`${id}-option-${index}`} role="option" aria-selected={active === index}><button type="button" onMouseDown={event => event.preventDefault()} onClick={() => pick(stock.ticker)}><strong>{stock.ticker}</strong><span>{[stock.company, stock.sector].filter(Boolean).join(" · ") || "Setor não informado"}</span></button></li>)}</ul>}
    <p className="caption" role="status">{value.length}/{max} ativos selecionados.</p>{error && <Notice tone="negative">{error}</Notice>}
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
