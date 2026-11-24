import Link from "next/link";
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { MAX_SELECTION, SELECTION_FULL_MESSAGE, useApp, type SelectionUpdate } from "./AppProvider";
import StockPicker from "./StockPicker";
import { formatValue, Loading, Notice, ScrollX } from "./ui";
import { tickerColor } from "./Shell";
import { api } from "../lib/api";
import type { PortfolioResult } from "../lib/portfolio";
import { period } from "../lib/portfolio";

const months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"];

function PortfolioSetup() {
  const { session, stocks, setSelection, refreshSession } = useApp();
  const [pending, setPending] = useState<"selection" | "analysis" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);
  if (!session) return null;
  const { selection, revision } = session;
  const full = selection.length >= MAX_SELECTION;
  const stockByTicker = new Map(stocks.map(stock => [stock.ticker, stock]));

  async function run(kind: "selection" | "analysis", action: () => Promise<unknown>) {
    if (inFlight.current) return;
    inFlight.current = true;
    setPending(kind);
    setError(null);
    try {
      await action();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      // Resync if another page changed the selection or the calculation already published.
      await refreshSession().catch(() => {});
    } finally {
      inFlight.current = false;
      setPending(null);
    }
  }

  function changeSelection(update: SelectionUpdate) {
    void run("selection", () => setSelection(update));
  }

  function calculate() {
    if (selection.length < 2) return;
    void run("analysis", async () => {
      await api("/api/portfolio/analyze", { tickers: selection, revision });
      await refreshSession();
    });
  }

  return <div className="portfolio-setup">
    <StockPicker selection={selection} action="Adicionar" disabled={!!pending || full} closeOnChoose
      onChoose={ticker => changeSelection(current => {
        if (current.includes(ticker)) return [...current];
        if (current.length >= MAX_SELECTION) throw new Error(SELECTION_FULL_MESSAGE);
        return [...current, ticker];
      })} />
    {selection.length > 0 && <ul className="asset-list" aria-label="Ativos da carteira">
      {selection.map((ticker, index) => {
        const stock = stockByTicker.get(ticker);
        return <li key={ticker}>
          <i className="swatch" style={{ background: tickerColor(index) }} aria-hidden="true" />
          <b>{ticker}</b><span>{stock?.company ?? "Nome indisponível"}</span>
          <button type="button" className="icon-btn" disabled={!!pending} aria-label={`Remover ${ticker}`}
            onClick={() => changeSelection(current => current.filter(item => item !== ticker))}>×</button>
        </li>;
      })}
    </ul>}
    {full && <p className="param-hint">Remova um ativo para adicionar outro.</p>}
    <div className="flow-links">
      <button type="button" className="primary" disabled={!!pending || selection.length < 2} onClick={calculate}>
        {pending === "analysis" ? "Calculando…" : "Calcular carteira"}
      </button>
      <Link className="button-link" href="/portfolio">Personalizar na Carteira <span aria-hidden="true">→</span></Link>
    </div>
    <p className="param-hint">Cálculo com a estratégia de mínima volatilidade e dois anos de histórico.</p>
    {pending === "selection" && <Loading>Atualizando a seleção…</Loading>}
    {pending === "analysis" && <ElapsedLoading>Baixando cotações e calculando a carteira…</ElapsedLoading>}
    {error && <Notice tone="negative">{error}</Notice>}
  </div>;
}

export function PortfolioRequired({ count, feature, title = "Calcule uma carteira primeiro", allowSetup = false, children }: { count: number; feature: string; title?: string; allowSetup?: boolean; children?: ReactNode }) {
  return <section className="workspace-start">
    <h2>{title}</h2>
    <p>{count === 0 ? `Adicione pelo menos dois ativos e calcule a carteira para ${feature}.`
      : count === 1 ? `Adicione mais um ativo e calcule a carteira para ${feature}.`
      : `Calcule a carteira com os ativos já selecionados para ${feature}.`}</p>
    {allowSetup ? <PortfolioSetup /> : <Link className="button-link primary" href="/portfolio">{count >= 2 ? "Configurar carteira" : "Escolher ativos da carteira"} <span aria-hidden="true">→</span></Link>}
    {children}
  </section>;
}

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
