"use client";

import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { download } from "../lib/api";
import { formatCompactValue, formatPercent, formatValue, missingValue } from "../lib/formatting";
import type { Unit } from "../lib/formatting";

export { formatCompactValue, formatPercent, formatValue, missingValue };
export type { Unit };

export type Signal = "positive" | "negative" | "warning" | "neutral" | "info";
export type Metric = { label: string; value: number | string | null; unit?: Unit; currency?: string; digits?: number; compact?: boolean; status?: Signal; help?: string; trend?: number[]; meter?: number };
export type Column = { key: string; label: string; unit?: Unit; currency?: string; digits?: number; signed?: boolean; compact?: boolean };

const minus = "−";

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
export function ValueText({ value, unit = "number", digits = 2, currency = "BRL", compact = false }: { value: unknown; unit?: Unit; digits?: number; currency?: string; compact?: boolean }) {
  const text = compact ? formatCompactValue(value, unit, 1, currency) : formatValue(value, unit, digits, currency);
  const exact = compact ? formatValue(value, unit, unit === "money" ? Math.max(2, digits) : digits, currency) : text;
  if (compact && exact !== text) return <span title={exact} aria-label={exact}>{text}</span>;
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

/** Decorative trend line for a value already shown as text; the series comes from the API unchanged. */
export function Sparkline({ values, tone = "info" }: { values: number[]; tone?: Signal }) {
  const points = values.filter(Number.isFinite);
  if (points.length < 2) return null;
  const min = Math.min(...points), max = Math.max(...points), span = max - min || 1;
  const coords = points.map((point, index) => [index / (points.length - 1) * 100, 34 - (point - min) / span * 30]);
  const line = coords.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const [lastX, lastY] = coords[coords.length - 1];
  return <svg className={`spark is-${tone}`} viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true" focusable="false">
    <polygon className="spark-area" points={`0,36 ${line} 100,36`} />
    <polyline className="spark-line" points={line} vectorEffect="non-scaling-stroke" />
    <line className="spark-end" x1={lastX} x2={lastX} y1={lastY} y2={lastY} vectorEffect="non-scaling-stroke" />
  </svg>;
}

/** Matches a media query after mount; server render and first paint assume false. */
export function useMedia(query: string): boolean {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const list = window.matchMedia(query);
    const sync = () => setMatches(list.matches);
    sync();
    list.addEventListener("change", sync);
    return () => list.removeEventListener("change", sync);
  }, [query]);
  return matches;
}

/** A titled tile of the desk grid. On narrow screens the title bar collapses and expands the body. */
export function Pane({ title, meta, actions, children, className, area, id, flush = false, narrowCollapsed = false }: { title: ReactNode; meta?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; area?: string; id?: string; flush?: boolean; narrowCollapsed?: boolean }) {
  const titleId = useId();
  const narrow = useMedia("(max-width: 900px)");
  const [collapsed, setCollapsed] = useState(narrowCollapsed);
  const hidden = narrow && collapsed;
  return <section id={id} className={`pane${className ? ` ${className}` : ""}`} style={area ? { gridArea: area } : undefined} aria-labelledby={titleId} data-collapsed={hidden || undefined}>
    <header className="pane-bar">
      <h2 id={titleId}>{narrow ? <button type="button" aria-expanded={!hidden} onClick={() => setCollapsed(value => !value)}><span aria-hidden="true" className="pane-caret">{hidden ? "▸" : "▾"}</span>{title}</button> : title}</h2>
      {meta && <span className="pane-meta">{meta}</span>}
      {actions && <div className="pane-actions">{actions}</div>}
    </header>
    {!hidden && <div className={`pane-body${flush ? " is-flush" : ""}`}>{children}</div>}
  </section>;
}

const statusGlyphs: Record<Signal, string> = { positive: "▲", negative: "▼", warning: "!", neutral: "■", info: "i" };
export function Metrics({ items, size = "md" }: { items: Metric[]; size?: "lg" | "md" }) {
  return <dl className={`kv-grid is-${size}`}>{items.map((item, index) => {
    const pill = !!item.status && !!item.help && item.help.length <= 24 && !item.help.includes(".");
    return <div className="kv" key={`${item.label}-${index}`}>
      <dt>{item.label}</dt>
      <dd className={`kv-value${item.status ? ` is-${item.status}` : ""}`}>
        {item.status && <span className="kv-glyph" aria-hidden="true">{statusGlyphs[item.status]}</span>}
        <ValueText value={item.value} unit={item.unit} digits={item.digits} currency={item.currency} compact={item.compact} />
      </dd>
      {pill && <dd className={`kv-status is-${item.status}`}>{item.help}</dd>}
      {item.trend && <dd className="kv-spark"><Sparkline values={item.trend} tone={item.status ?? "info"} /></dd>}
      {typeof item.meter === "number" && Number.isFinite(item.meter) && <dd className="kv-meter" aria-hidden="true"><span style={{ width: `${Math.min(1, Math.max(0, item.meter)) * 100}%` }} /></dd>}
      {item.help && !pill && <dd className="kv-help" title={item.help}>{item.help}</dd>}
    </div>;
  })}</dl>;
}
const isNumericColumn = (column: Column) => !!column.signed || (!!column.unit && column.unit !== "text");
const finiteNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
/** Missing values sort last in both directions; numbers compare numerically, text in pt-BR order. */
function compareRows(a: object, b: object, key: string, direction: 1 | -1): number {
  const x = (a as Record<string, unknown>)[key], y = (b as Record<string, unknown>)[key];
  const missingX = x === null || x === undefined || (typeof x === "number" && !Number.isFinite(x));
  const missingY = y === null || y === undefined || (typeof y === "number" && !Number.isFinite(y));
  if (missingX || missingY) return Number(missingX) - Number(missingY);
  if (typeof x === "number" && typeof y === "number") return (x - y) * direction;
  return String(x).localeCompare(String(y), "pt-BR") * direction;
}
/** Dense sortable table. `extremes` marks the largest and smallest value of each numeric column without judging direction. */
export function Table<T extends object>({ columns, rows, rowKey = "ticker", caption, sortable = false, extremes = false, swatch }: { columns: Column[]; rows: T[]; rowKey?: string; caption?: string; sortable?: boolean; extremes?: boolean; swatch?: (row: T) => string | undefined }) {
  const [sort, setSort] = useState<{ key: string; direction: 1 | -1 } | null>(null);
  const ordered = useMemo(() => sort ? [...rows].sort((a, b) => compareRows(a, b, sort.key, sort.direction)) : rows, [rows, sort]);
  const bounds = useMemo(() => {
    const result = new Map<string, { max: number; min: number }>();
    if (!extremes || rows.length < 3) return result;
    for (const column of columns.filter(isNumericColumn)) {
      const values = rows.map(row => (row as Record<string, unknown>)[column.key]).filter(finiteNumber);
      if (values.length >= 3) result.set(column.key, { max: Math.max(...values), min: Math.min(...values) });
    }
    return result;
  }, [columns, extremes, rows]);
  const toggle = (key: string) => setSort(current => current?.key === key ? { key, direction: current.direction === 1 ? -1 : 1 } : { key, direction: 1 });
  return <div className="grid-scroll" tabIndex={0} role="region" aria-label={caption ?? "Tabela de dados"}><table className="grid"><thead><tr>{columns.map(column => {
    const numeric = isNumericColumn(column);
    const direction = sort?.key === column.key ? sort.direction : undefined;
    return <th key={column.key} scope="col" className={numeric ? "num" : undefined} aria-sort={direction === undefined ? undefined : direction === 1 ? "ascending" : "descending"}>
      {sortable && numeric ? <button type="button" className="th-sort" onClick={() => toggle(column.key)}>{column.label}<span aria-hidden="true">{direction === undefined ? "" : direction === 1 ? " ▲" : " ▼"}</span></button> : column.label}
    </th>;
  })}</tr></thead><tbody>{ordered.map((row, index) => {
    const values = row as Record<string, unknown>;
    const key = String(values[rowKey] ?? index);
    const color = swatch?.(row);
    return <tr key={key}>{columns.map((column, columnIndex) => {
      const value = values[column.key];
      const currency = column.currency ?? (typeof values.currency === "string" ? values.currency : "BRL");
      const exactDigits = column.compact && column.unit === "money" ? Math.max(2, column.digits ?? 2) : column.digits;
      const exact = formatValue(value, column.unit, exactDigits, currency, column.signed);
      const shown = column.compact ? formatCompactValue(value, column.unit, 1, currency, column.signed) : exact;
      const bound = bounds.get(column.key);
      const extreme = bound && finiteNumber(value) && bound.max !== bound.min ? value === bound.max ? "max" : value === bound.min ? "min" : null : null;
      const signedTone = column.signed && finiteNumber(value) ? value > 0 ? " is-pos" : value < 0 ? " is-neg" : "" : "";
      return <td key={column.key} className={`${isNumericColumn(column) ? "num" : ""}${extreme ? ` is-${extreme}` : ""}${signedTone}`.trim() || undefined}>
        {columnIndex === 0 && color && <i className="swatch" style={{ background: color }} aria-hidden="true" />}
        {shown === exact ? shown : <span title={exact} aria-label={exact}>{shown}</span>}
        {extreme && <span className="extreme" title={extreme === "max" ? "Maior valor da coluna" : "Menor valor da coluna"}><span aria-hidden="true">{extreme === "max" ? "▴" : "▾"}</span><span className="visually-hidden">{extreme === "max" ? " (maior da coluna)" : " (menor da coluna)"}</span></span>}
      </td>;
    })}</tr>;
  })}</tbody></table></div>;
}
const noticeGlyphs: Record<Signal, string> = { positive: "✓", negative: "✕", warning: "!", neutral: "•", info: "i" };
export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: Signal }) {
  return <div className={`note is-${tone}`} role={tone === "negative" ? "alert" : undefined}><b className="note-glyph" aria-hidden="true">{noticeGlyphs[tone]}</b><div className="note-body">{children}</div></div>;
}
export function Loading({ children }: { children?: ReactNode }) {
  return <div className="load" role="status" aria-live="polite"><span className="load-bar" aria-hidden="true" /><p>{children ?? "Carregando dados para a análise."}</p></div>;
}
/** Empty slot of the desk: what will appear here and the one action that fills it. */
export function Idle({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return <div className="idle"><p>{children}</p>{action}</div>;
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
