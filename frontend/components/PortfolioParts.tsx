import { useEffect, useState, type CSSProperties } from "react";
import { formatValue, Loading, ScrollX } from "./ui";
import { tickerColor } from "./Shell";
import type { PortfolioResult } from "../lib/portfolio";
import { period } from "../lib/portfolio";

const months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"];

/** Weights sorted largest first, with a stacked composition bar; colors follow the selection order. */
export function WeightList({ weights, selection }: { weights: PortfolioResult["weights"]; selection: string[] }) {
  const rows = [...weights].sort((a, b) => b.weight - a.weight);
  const max = Math.max(...rows.map(row => row.weight), 0.0001);
  const color = (ticker: string) => tickerColor(Math.max(0, selection.indexOf(ticker)));
  return <><div className="stack" aria-hidden="true">{rows.filter(row => row.weight > 0).map(row => <span key={row.ticker} style={{ flexGrow: row.weight, background: color(row.ticker) }} title={`${row.ticker} ${formatValue(row.weight, "fraction", 2)}`} />)}</div>
  <ul className="weights" aria-label="Pesos do portfólio em porcentagem">{rows.map(row => <li key={row.ticker}>
    <i className="swatch" style={{ background: color(row.ticker) }} aria-hidden="true" /><b>{row.ticker}</b>
    <span className="weights-track" aria-hidden="true"><span style={{ width: `${Math.max(row.weight / max, 0) * 100}%` }} /></span>
    <span className="weights-value">{formatValue(row.weight, "fraction", 2)}</span>
  </li>)}</ul></>;
}

/** Strategy, allocation and sample of the run, as short facts for a pane title bar. */
export function runFacts(result: PortfolioResult): string[] {
  return [result.mode, result.allocation_label, `${result.sample.selected_count} ativos`, `${result.sample.count.toLocaleString("pt-BR")} retornos diários (${period(result.sample)})`];
}

export function MonthlyReturns({ rows }: { rows: PortfolioResult["monthly_returns"] }) {
  const keys = [...months.map((_, index) => `m${index + 1}`), "ytd"];
  return <ScrollX label="Retornos mensais do portfólio, em pontos percentuais por mês">
    <table className="grid heat">
      <caption>Retorno mensal em %. + ganho; − perda; “—” mês sem dados. A intensidade do fundo acompanha o tamanho do movimento.</caption>
      <thead><tr><th scope="col">Ano</th>{[...months, "No ano"].map(label => <th scope="col" key={label} className="num">{label}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.year}><th scope="row">{row.year}</th>{keys.map(key => {
        const value = row[key];
        const tone = value == null ? "" : value > 0 ? "is-positive" : value < 0 ? "is-negative" : "is-neutral";
        const text = value == null ? "—" : `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
        const heat = value == null ? 0 : Math.min(Math.abs(value) / (key === "ytd" ? 30 : 8), 1);
        return <td key={key} className={`num heat-cell ${tone}`.trim()} style={{ "--heat": heat.toFixed(3) } as CSSProperties}>{text}</td>;
      })}</tr>)}</tbody>
    </table>
  </ScrollX>;
}

/** Loading with the run time in a hidden span, so the live region is not re-announced every second. */
export function ElapsedLoading({ children }: { children: string }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return <Loading>{children} <span aria-hidden="true">Rodando há {seconds} s.</span></Loading>;
}
