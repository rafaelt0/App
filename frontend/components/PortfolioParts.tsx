import { formatValue } from "./ui";
import type { PortfolioResult } from "../lib/portfolio";
import { period } from "../lib/portfolio";

const months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"];

export function WeightList({ weights }: { weights: PortfolioResult["weights"] }) {
  const rows = [...weights].sort((a, b) => b.weight - a.weight);
  const max = Math.max(...rows.map(row => row.weight), 0.0001);
  return <ul className="pf-weights" aria-label="Pesos do portfólio em porcentagem">{rows.map(row => <li key={row.ticker}>
    <span className="pf-weight-ticker">{row.ticker}</span>
    <span className="pf-weight-track" aria-hidden="true"><span className="pf-weight-fill" style={{ width: `${Math.max(row.weight / max, 0) * 100}%` }} /></span>
    <span className="pf-weight-value">{formatValue(row.weight, "fraction", 2)}</span>
  </li>)}</ul>;
}

export function ResultSummary({ result }: { result: PortfolioResult }) {
  const items: [string, string][] = [
    ["Estratégia", result.mode],
    ["Alocação", result.allocation_label],
    ["Ativos selecionados", String(result.sample.selected_count)],
    ["Retornos diários completos", `${result.sample.count.toLocaleString("pt-BR")} (${period(result.sample)})`],
    ["Registro", result.saved ? "Alocação salva" : "Alocação não salva"],
  ];
  return <div className="pf-summary"><dl>{items.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    {result.sample.zero_excluded && <p className="caption">Ativos com peso zero foram excluídos da amostra.</p>}</div>;
}

export function MonthlyReturns({ rows }: { rows: PortfolioResult["monthly_returns"] }) {
  const keys = [...months.map((_, index) => `m${index + 1}`), "ytd"];
  return <div className="table-wrap pf-monthly" tabIndex={0} role="region" aria-label="Retornos mensais do portfólio, em pontos percentuais por mês">
    <table>
      <caption className="pf-caption">Retorno mensal em %. Sinal + indica ganho; sinal − indica perda; “—” indica mês sem dados.</caption>
      <thead><tr><th scope="col">Ano</th>{[...months, "Acumulado no ano"].map(label => <th scope="col" key={label} className="numeric-cell">{label}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.year}><th scope="row">{row.year}</th>{keys.map(key => {
        const value = row[key];
        const tone = value == null ? "" : value > 0 ? "is-positive" : value < 0 ? "is-negative" : "is-neutral";
        const text = value == null ? "—" : `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
        return <td key={key} className={`numeric-cell ${tone}`.trim()}>{text}</td>;
      })}</tr>)}</tbody>
    </table>
  </div>;
}
