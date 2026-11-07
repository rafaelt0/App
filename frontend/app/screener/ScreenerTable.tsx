"use client";
import { useRef } from "react";
import { formatCompactValue, formatValue } from "../../components/ui";
import type { ScreenerColumn, Sort } from "../../lib/screener";

const compactMoneyColumns = new Set(["liq2m", "patrliq"]);

type Props = {
  columns: ScreenerColumn[];
  rows: Record<string, unknown>[];
  sort: Sort;
  onSort: (field: string) => void;
  selectedKey?: string;
  onSelect: (ticker: string) => void;
  onActivate: (ticker: string) => void;
  inSelection: Set<string>;
};

/** Whole-row activation with roving focus: a click or Enter adds the row to the selection, ↑↓ only move the highlight. */
export default function ScreenerTable({ columns, rows, sort, onSort, selectedKey, onSelect, onActivate, inSelection }: Props) {
  const body = useRef<HTMLTableSectionElement>(null);
  const tickers = rows.map(row => String(row.ticker));
  const focusIndex = Math.max(0, selectedKey ? tickers.indexOf(selectedKey) : 0);
  function move(index: number) {
    const next = Math.max(0, Math.min(tickers.length - 1, index));
    onSelect(tickers[next]);
    const button = body.current?.querySelectorAll<HTMLButtonElement>(".scr-row-select")[next];
    button?.focus();
    button?.closest("tr")?.scrollIntoView({ block: "nearest" });
  }
  return <table className="grid scr-grid" aria-label="Ações que atendem aos critérios"><thead><tr>
    <th scope="col">Papel</th>
    {columns.map(column => {
      const active = sort.field === column.key;
      const ariaSort = active ? (sort.direction === "asc" ? "ascending" : "descending") : "none";
      return <th key={column.key} scope="col" className={column.key === "setor" ? undefined : "num"} aria-sort={ariaSort}>
        <button type="button" className="th-sort" onClick={() => onSort(column.key)}>{column.label}<span aria-hidden="true">{active ? (sort.direction === "asc" ? " ▲" : " ▼") : ""}</span></button>
      </th>;
    })}
  </tr></thead><tbody ref={body} onKeyDown={event => {
    const index = tickers.indexOf(String((event.target as HTMLElement).closest("tr")?.dataset.ticker));
    if (index < 0) return;
    if (event.key === "ArrowDown") { event.preventDefault(); move(index + 1); }
    else if (event.key === "ArrowUp") { event.preventDefault(); move(index - 1); }
    else if (event.key === "Home") { event.preventDefault(); move(0); }
    else if (event.key === "End") { event.preventDefault(); move(tickers.length - 1); }
    else if (event.key === "Enter") { event.preventDefault(); onActivate(tickers[index]); }
  }}>{rows.map((row, index) => {
    const ticker = tickers[index];
    const selected = selectedKey === ticker;
    return <tr key={ticker} data-ticker={ticker} className={`scr-row${selected ? " selected-row" : ""}`} onClick={() => onActivate(ticker)}>
      <th scope="row"><button type="button" className="scr-row-select row-btn" tabIndex={index === focusIndex ? 0 : -1} aria-label={`Adicionar ${ticker} à seleção${inSelection.has(ticker) ? " (já está)" : ""}`} aria-pressed={selected}>
        <span className="scr-mark" aria-hidden="true">{inSelection.has(ticker) ? "✓" : selected ? "▸" : ""}</span><span className="scr-ticker">{ticker}</span>
      </button></th>
      {columns.map(column => {
        const currency = column.currency ?? (typeof row.currency === "string" ? row.currency : "BRL");
        const isCompactMoney = column.unit === "money" && compactMoneyColumns.has(column.key);
        const fullValue = formatValue(row[column.key], column.unit, isCompactMoney ? 2 : column.digits, currency);
        const shown = isCompactMoney ? formatCompactValue(row[column.key], column.unit, 1, currency) : fullValue;
        return <td key={column.key} className={column.key === "setor" ? "ell" : "num"} title={shown !== fullValue ? fullValue : undefined} aria-label={shown !== fullValue ? `${column.label}: ${fullValue}` : undefined}>{shown}</td>;
      })}
    </tr>;
  })}</tbody></table>;
}
