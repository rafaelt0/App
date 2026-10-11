"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useApp } from "../../components/AppProvider";
import Chart from "../../components/Chart";
import { EmptyState, ExportButton, Loading, Metrics, Notice, PageHeader, Section, Table } from "../../components/ui";
import { api } from "../../lib/api";
import {
  displayedTrajectories, initialInputs, periodLabel, sameInputs, simulationPresets,
  type CalibrationWindow, type SimulationInputs, type SimulationPortfolio, type SimulationResult,
} from "../../lib/simulation";

const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 1 });
const number = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1 });

export default function SimulationPage() {
  const { session, loading, error: sessionError, refreshSession } = useApp();
  const [portfolio, setPortfolio] = useState<SimulationPortfolio | null>(null);
  const [inputs, setInputs] = useState<SimulationInputs>(initialInputs);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [displayCount, setDisplayCount] = useState(10);
  const [displayDraft, setDisplayDraft] = useState("10");
  const [preparing, setPreparing] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restoreFailed, setRestoreFailed] = useState(false);
  const [restoredNotice, setRestoredNotice] = useState(false);
  const generation = useRef(0);
  const restoreRequest = useRef<{ revision: number; promise: Promise<unknown> } | null>(null);
  const activeRevision = useRef(session?.revision);
  activeRevision.current = session?.revision;
  const contextRevision = session?.revision;
  const returnReady = session?.simulation_ready ?? false;
  const restoredContext = session?.restored ?? false;
  const portfolioReady = session?.portfolio_ready ?? false;
  const refreshSessionRef = useRef(refreshSession);
  refreshSessionRef.current = refreshSession;

  useEffect(() => {
    if (loading || contextRevision === undefined) return;
    const token = ++generation.current;
    const revision = contextRevision;
    let cancelled = false;
    setPortfolio(null);
    setResult(null);
    setError(null);
    setRestoreFailed(false);
    setRunning(false);
    setPreparing(true);
    async function prepare() {
      try {
        if (!returnReady && (restoredContext || portfolioReady)) {
          if (restoreRequest.current?.revision !== revision) {
            restoreRequest.current = { revision, promise: api("/api/portfolio/restore", { target: "simulation", revision }) };
          }
          await restoreRequest.current.promise;
          if (cancelled || activeRevision.current !== revision || token !== generation.current) return;
          setRestoredNotice(true);
          await refreshSessionRef.current();
        }
        const snapshot = await api<SimulationPortfolio>("/api/portfolio");
        if (!cancelled && activeRevision.current === revision && token === generation.current) setPortfolio(snapshot);
      } catch (cause) {
        if (!cancelled && activeRevision.current === revision && token === generation.current) {
          setError(cause instanceof Error ? cause.message : String(cause));
          setRestoreFailed(!returnReady && (restoredContext || portfolioReady));
        }
      } finally {
        if (!cancelled && token === generation.current) setPreparing(false);
      }
    }
    void prepare();
    return () => { cancelled = true; };
    // Session readiness, not result or form state, controls portfolio restoration.
  }, [loading, contextRevision, returnReady, portfolioReady, restoredContext]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session || !portfolio?.id || !portfolio.simulation_ready) return;
    const token = ++generation.current;
    const revision = session.revision;
    const portfolioId = portfolio.id;
    const submitted = { ...inputs, revision, portfolio_id: portfolioId };
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const calculated = await api<SimulationResult>("/api/simulation/run", submitted);
      if (token !== generation.current || activeRevision.current !== revision || calculated.portfolio_id !== portfolioId) return;
      setResult(calculated);
      setDisplayCount(Math.min(10, calculated.inputs.simulations));
      setDisplayDraft(String(Math.min(10, calculated.inputs.simulations)));
      await refreshSession();
    } catch (cause) {
      if (token === generation.current && activeRevision.current === revision) setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (token === generation.current) setRunning(false);
    }
  }

  const visible = result && session && result.revision === session.revision && result.portfolio_id === portfolio?.id && sameInputs(inputs, result.inputs) ? result : null;
  const individual = useMemo(() => visible ? displayedTrajectories(visible.charts.individual, displayCount) : null, [visible, displayCount]);
  const ready = !!portfolio?.id && portfolio.simulation_ready && portfolio.revision === session?.revision;
  const rawWeights = portfolio ? portfolio.tickers.map(ticker => ({ ticker, weight: portfolio.weights[ticker] })) : [];
  const total = rawWeights.reduce((sum, row) => sum + row.weight, 0);
  const allocation = visible?.allocation ?? rawWeights.map(row => ({ ticker: row.ticker, weight: row.weight / total }));

  return <>
    <PageHeader title="Simulação de portfólio" description="Projete faixas de retorno e risco a partir dos pesos definidos na carteira." icon="simulation" />
    {sessionError && <Notice tone="negative">{sessionError}</Notice>}
    {error && <Notice tone="negative">{error}</Notice>}
    {loading || preparing ? <Loading>Restaurando contexto da carteira…</Loading> : !ready ?
      <EmptyState title={restoreFailed ? "Não foi possível restaurar a carteira" : session?.selection.length ? "Atualize o portfólio" : "Portfólio não configurado"} href="/portfolio" action="Ir para Portfolio">
        {restoreFailed ? "A carteira salva foi encontrada, mas seu histórico não pôde ser carregado. Abra Portfolio e carregue a análise novamente." : session?.selection.length ? "A seleção atual não possui histórico analisado. Volte para Portfolio e clique em Carregar portfólio antes de rodar a simulação." : "Configure seu portfólio na página Portfolio e carregue a análise para liberar a Simulação Monte Carlo."}
      </EmptyState> : <>
      {restoredNotice && <Notice>Carteira salva restaurada. Confira os ativos e pesos antes de rodar a simulação.</Notice>}
      <Section title="Alocação usada na simulação">
        <p>{portfolio?.mode}</p>
        <Metrics items={allocation.map(row => ({ label: row.ticker, value: row.weight, unit: "fraction", digits: 1 }))} />
        <p className="discreet-status">Pesos normalizados a partir da carteira carregada; ativos com peso zero permanecem zerados.</p>
      </Section>
      <form onSubmit={submit}>
        <p>Escolha um ponto de partida; você pode ajustar os valores abaixo.</p>
        <div className="actions">{simulationPresets.map(preset => <button key={preset.label} type="button" disabled={running} title={preset.description} onClick={() => setInputs(current => ({ ...current, simulations: preset.simulations, initial_capital: preset.initial_capital, years: preset.years }))}>{preset.label}</button>)}</div>
        <label>Janela de calibração
          <select value={inputs.calibration} onChange={event => setInputs(current => ({ ...current, calibration: (event.target.value === "portfolio" ? "portfolio" : Number(event.target.value)) as CalibrationWindow }))}>
            <option value="portfolio">Histórico do Portfolio</option><option value="2">2 Anos</option><option value="3">3 Anos</option><option value="5">5 Anos</option>
          </select>
        </label>
        <p className="discreet-status">Define o histórico usado para estimar retornos e risco; não altera os pesos. Janelas maiores podem reduzir o erro-padrão estimado, mas incluem regimes mais antigos.</p>
        <div className="controls-grid">
          <label>Número de Simulações<input type="number" required min={10} max={3000} step={1} value={inputs.simulations} onChange={event => setInputs(current => ({ ...current, simulations: event.target.valueAsNumber }))} /></label>
          <label>Capital Inicial (R$)<input type="number" required min={100} step={1} value={inputs.initial_capital} onChange={event => setInputs(current => ({ ...current, initial_capital: event.target.valueAsNumber }))} /></label>
          <label>Anos<input type="number" required min={1} max={10} step={1} value={inputs.years} onChange={event => setInputs(current => ({ ...current, years: event.target.valueAsNumber }))} /></label>
        </div>
        <p className="discreet-status">O horizonte é independente da janela de calibração. São 252 dias úteis por ano, com limite de 10 anos. As datas simuladas incluem dias de semana, não o calendário de feriados da B3.</p>
        <button type="submit" className="primary" disabled={running}>Rodar Simulação</button>
      </form>
      {running && <Loading>Rodando simulações Monte Carlo multivariadas…</Loading>}
      {!visible && !running && <Notice>Configure os parâmetros acima e clique em Rodar Simulação para ver os resultados.</Notice>}
      {visible && <>
        <Section title="Calibração histórica">
          <p>{visible.calibration.count} retornos diários completos ({periodLabel(visible.calibration)}). Média log anual estimada: {percent.format(visible.calibration.annual_log_mean)} ±{number.format(visible.calibration.iid_standard_error * 100)} p.p. (1 erro-padrão IID; não mede a incerteza dos retornos futuros).</p>
          {visible.calibration.sensitivity.length > 0 && <>
            <h3>Sensibilidade à janela histórica (pesos fixos)</h3>
            <Table columns={[{ key: "window", label: "Janela" }, { key: "count", label: "Retornos completos", unit: "integer" }, { key: "period", label: "Período" }, { key: "annual_log_mean", label: "Média log anual", unit: "fraction", digits: 1 }, { key: "error", label: "Erro-padrão IID (p.p.)", unit: "number", digits: 1 }]} rows={visible.calibration.sensitivity.map(row => ({ ...row, window: `${row.years} anos`, period: periodLabel(row), error: row.iid_standard_error * 100 }))} />
            <p>Janelas aninhadas usam a mesma alocação e o mesmo endpoint. O erro-padrão pressupõe dias independentes; mudanças entre janelas podem indicar efeito de regimes de mercado.</p>
          </>}
          {visible.warnings.map(warning => <Notice key={warning} tone="warning">{warning}</Notice>)}
        </Section>
        <Section title="Estatísticas da Simulação Monte Carlo">
          <Metrics items={[
            { label: "CAGR do valor final médio", value: visible.stats.mean_implied_cagr, unit: "fraction", digits: 1, help: "Crescimento anual implícito no valor final médio; não é a média dos CAGRs das trajetórias." },
            { label: "Probabilidade de Ganho", value: visible.stats.gain_probability, unit: "fraction", digits: 1, help: `Percentual de simulações acima do capital inicial sob o modelo normal. Erro Monte Carlo de até ±${number.format(visible.monte_carlo_error_pp)} p.p. (95% aproximado, pior caso); não inclui incerteza dos parâmetros.` },
            { label: "Retorno final P5", value: visible.stats.p5_return, unit: "fraction", digits: 1, help: "Retorno no percentil 5 dos valores finais simulados; não é uma estimativa de VaR nem representa perda máxima." },
          ]} />
          <Metrics items={[
            { label: "Valor Esperado Final", value: visible.stats.mean_terminal, unit: "money" },
            { label: "CAGR do valor final P75", value: visible.stats.p75_implied_cagr, unit: "fraction", digits: 1 },
            { label: "Valor final P5", value: visible.stats.p5_terminal, unit: "money" },
            { label: "Média dos 5% menores valores", value: visible.stats.lower_tail_mean, unit: "money" },
            { label: "Menor valor simulado", value: visible.stats.minimum, unit: "money" },
            { label: "Maior valor simulado", value: visible.stats.maximum, unit: "money" },
          ]} />
          <Notice>Valor final P5: percentil 5 dos valores finais simulados; não é uma medida VaR de perda nem representa perda máxima.</Notice>
          <Notice>Média dos 5% menores valores: média dos resultados finais menores ou iguais ao P5 (inclui empates) na cauda inferior simulada.</Notice>
        </Section>
        <Section title="Comparação de modelos">
          <Table columns={[{ key: "model", label: "Modelo" }, { key: "p5", label: "Valor final P5", unit: "money" }, { key: "p95", label: "Valor final P95", unit: "money" }, { key: "gain_probability", label: "Ganho sobre o capital inicial", unit: "fraction", digits: 1 }]} rows={visible.comparison} />
          <p>O modelo normal usa retornos logarítmicos diários correlacionados. A reamostragem sorteia dias históricos inteiros (preserva choques simultâneos entre ativos), mas não cria crises ausentes da amostra nem preserva sequências de volatilidade. A diferença entre modelos indica sensibilidade às hipóteses, não incerteza coberta pelas faixas.</p>
        </Section>
        <Section title="Trajetórias Individuais das Simulações">
          <label>Número de trajetórias exibidas<input type="number" min={5} max={Math.min(50, visible.inputs.simulations)} step={5} value={displayDraft} onChange={event => {
            setDisplayDraft(event.target.value);
            const count = event.target.valueAsNumber;
            if (Number.isInteger(count) && count >= 5 && count <= Math.min(50, visible.inputs.simulations)) setDisplayCount(count);
          }} onBlur={() => {
            const count = Number(displayDraft);
            const bounded = Number.isFinite(count) ? Math.max(5, Math.min(Math.min(50, visible.inputs.simulations), Math.trunc(count))) : displayCount;
            setDisplayCount(bounded);
            setDisplayDraft(String(bounded));
          }} /></label>
          <p className="discreet-status">Este controle só muda a exibição das primeiras trajetórias; não roda novamente a simulação nem altera estatísticas ou percentis.</p>
          {individual && <Chart figure={individual} />}
        </Section>
        <Section title="Faixas de percentis"><Chart figure={visible.charts.fan} />
          <div className="actions"><ExportButton path={`/api/simulation/${visible.id}/summary.csv`}>Resumo (CSV)</ExportButton><ExportButton path={`/api/simulation/${visible.id}/percentiles.csv`}>Percentis por Data (CSV)</ExportButton></div>
        </Section>
        <Section title="Distribuição do Valor Final do Portfólio"><Chart figure={visible.charts.histogram} />
          <Metrics items={[
            { label: "Mínimo", value: visible.distribution.minimum, unit: "money" }, { label: "Q1 (25%)", value: visible.distribution.q1, unit: "money" },
            { label: "Mediana (50%)", value: visible.distribution.median, unit: "money" }, { label: "Q3 (75%)", value: visible.distribution.q3, unit: "money" },
            { label: "Máximo", value: visible.distribution.maximum, unit: "money" }, { label: "Média", value: visible.distribution.mean, unit: "money" },
            { label: "Desvio Padrão", value: visible.distribution.standard_deviation, unit: "money", help: "Desvio padrão amostral dos valores finais (ddof=1)." },
          ]} />
        </Section>
        <Section title="Síntese do Analista">
          <h3>Projeção {visible.inputs.years} ano(s) — {visible.inputs.simulations} simulações</h3>
          <ul>
            <li>{percent.format(visible.stats.gain_probability)} dos cenários do modelo normal terminaram acima do capital inicial.</li>
            <li>CAGR implícito no valor final médio: {percent.format(visible.stats.mean_implied_cagr)} a.a. ao longo de {visible.inputs.years} ano(s).</li>
            <li>Retorno final P5: {percent.format(visible.stats.p5_return)}.</li>
            <li>Amplitude entre P5 e P95: {percent.format(visible.spread_fraction)} do capital inicial.</li>
          </ul>
          <p>Modelo normal de retornos logarítmicos multivariados ({visible.calibration.count} observações, {periodLabel(visible.calibration)}); pesos rebalanceados diariamente em ambos os modelos. Não inclui taxas nem impostos. Resultados são cenários, não previsões nem recomendação de investimento.</p>
          <p>Complemente a leitura da projeção com as notícias das empresas.</p><Link href="/noticias">Abrir Notícias</Link>
        </Section>
      </>}
    </>}
  </>;
}
