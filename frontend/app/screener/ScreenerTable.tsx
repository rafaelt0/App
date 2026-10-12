"use client";
import { formatValue } from "../../components/ui";
import type { ScreenerColumn, Sort } from "../../lib/screener";

type Props = {
  columns: ScreenerColumn[];
  rows: Record<string, unknown>[];
  sort: Sort;
  onSort: (field: string) => void;
  selectedKey?: string;
  onSelect: (ticker: string) => void;
};

/** Same markup as the shared Table, with sortable column headers. Missing values stay "—" via formatValue. */
export default function ScreenerTable({ columns, rows, sort, onSort, selectedKey, onSelect }: Props) {
  return <div className="table-scroll" tabIndex={0} aria-label="Tabela de dados"><table><thead><tr>
    <th scope="col">Selecionar</th>
    <th scope="col">Papel</th>
    {columns.map(column => {
      const active = sort.field === column.key;
      const ariaSort = active ? (sort.direction === "asc" ? "ascending" : "descending") : "none";
      return <th key={column.key} scope="col" className="numeric-cell" aria-sort={ariaSort}>
        <button type="button" className="scr-sort" onClick={() => onSort(column.key)}>{column.label}<span aria-hidden="true">{active ? (sort.direction === "asc" ? " ▲" : " ▼") : ""}</span></button>
      </th>;
    })}
  </tr></thead><tbody>{rows.map(row => {
    const ticker = String(row.ticker);
    return <tr key={ticker} className={selectedKey === ticker ? "selected-row" : undefined}>
      <td><button aria-label={`Selecionar ${ticker}`} aria-pressed={selectedKey === ticker} onClick={() => onSelect(ticker)}>{selectedKey === ticker ? "Selecionado" : "Selecionar"}</button></td>
      <td>{ticker}</td>
      {columns.map(column => <td key={column.key} className={column.unit && column.unit !== "text" ? "numeric-cell" : undefined}>{formatValue(row[column.key], column.unit, column.digits, column.currency ?? (typeof row.currency === "string" ? row.currency : "BRL"))}</td>)}
    </tr>;
  })}</tbody></table></div>;
}
