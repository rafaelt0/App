import { useEffect, useState, type CSSProperties } from "react";
import { formatValue, Loading } from "./ui";
import StockLogo from "./StockLogo";
import type { Stock } from "./AppProvider";
import type { PortfolioResult } from "../lib/portfolio";
import { period } from "../lib/portfolio";

const months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"];

export function WeightList({ weights, stocks }: { weights: PortfolioResult["weights"]; stocks: Stock[] }) {
  const rows = [...weights].sort((a, b) => b.weight - a.weight);
  const max = Math.max(...rows.map(row => row.weight), 0.0001);
  const shown = rows.filter(row => row.weight > 0);
  return <><div className="pf-strip" aria-hidden="true">{shown.map((row, index) => <span key={row.ticker} data-tone={index % 6} style={{ flexGrow: row.weight }} title={`${row.ticker} ${formatValue(row.weight, "fraction", 2)}`} />)}</div>
  <ul className="pf-weights" aria-label="Pesos do portfólio em porcentagem">{rows.map(row => <li key={row.ticker}>
    <span className="pf-weight-ticker"><span className="pf-swatch" data-tone={shown.indexOf(row) % 6} aria-hidden="true" /><StockLogo ticker={row.ticker} company={stocks.find(stock => stock.ticker === row.ticker)?.company ?? row.ticker} size={24}/><span>{row.ticker}</span></span>
    <span className="pf-weight-track" aria-hidden="true"><span className="pf-weight-fill" style={{ width: `${Math.max(row.weight / max, 0) * 100}%` }} /></span>
    <span className="pf-weight-value">{formatValue(row.weight, "fraction", 2)}</span>
  </li>)}</ul></>;
}

export function ResultSummary({ result }: { result: PortfolioResult }) {
  const items: [string, string][] = [
    ["Estratégia", result.mode],
    ["Alocação", result.allocation_label],
    ["Ativos selecionados", String(result.sample.selected_count)],
    ["Retornos diários completos", `${result.sample.count.toLocaleString("pt-BR")} (${period(result.sample)})`],
    ["Salvamento", result.saved ? "Carteira salva" : "Não foi possível salvar"],
  ];
  return <div className="context-strip pf-summary"><dl>{items.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    {result.sample.zero_excluded && <p className="caption">Ativos com peso zero foram excluídos da amostra.</p>}
    <p className="caption">{result.saved ? "Carteira salva para esta sessão anônima; não é uma conta." : "A análise vale nesta página, mas a carteira não foi salva para esta sessão anônima."}</p></div>;
}

export function MonthlyReturns({ rows }: { rows: PortfolioResult["monthly_returns"] }) {
  const keys = [...months.map((_, index) => `m${index + 1}`), "ytd"];
  return <div className="table-wrap pf-monthly" tabIndex={0} role="region" aria-label="Retornos mensais do portfólio, em pontos percentuais por mês">
    <table>
      <caption className="pf-caption">Retorno mensal em %. Sinal + indica ganho; sinal − indica perda; “—” indica mês sem dados. A intensidade do fundo acompanha o tamanho do movimento.</caption>
      <thead><tr><th scope="col">Ano</th>{[...months, "Acumulado no ano"].map(label => <th scope="col" key={label} className="numeric-cell">{label}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.year}><th scope="row">{row.year}</th>{keys.map(key => {
        const value = row[key];
        const tone = value == null ? "" : value > 0 ? "is-positive" : value < 0 ? "is-negative" : "is-neutral";
        const text = value == null ? "—" : `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
        const heat = value == null ? 0 : Math.min(Math.abs(value) / (key === "ytd" ? 30 : 8), 1);
        return <td key={key} className={`numeric-cell pf-heat ${tone}`.trim()} style={{ "--heat": heat.toFixed(3) } as CSSProperties}>{text}</td>;
      })}</tr>)}</tbody>
    </table>
  </div>;
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
