"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useApp } from "../../components/AppProvider";
import Chart from "../../components/Chart";
import { ElapsedLoading } from "../../components/PortfolioParts";
import { ExportButton, formatValue, Idle, Loading, Metrics, Notice, Pane, Table } from "../../components/ui";
import { tickerColor } from "../../components/Shell";
import { api } from "../../lib/api";
import type { PortfolioResult, Sample } from "../../lib/portfolio";
import {
  displayedTrajectories, initialInputs, periodLabel, polishFigure, sameInputs, simulationPresets,
  type CalibrationWindow, type SimulationInputs, type SimulationPortfolio, type SimulationResult,
} from "../../lib/simulation";

const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 1 });
const number = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1 });
const yearsLabel = (years: number) => `${number.format(years)} ${years === 1 ? "ano" : "anos"}`;

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
  const [sample, setSample] = useState<Sample | null>(null);
  const [portfolioChanged, setPortfolioChanged] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
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
        // Reads only: the owned analysis and last run for this selection are shown, never recalculated.
        const [snapshot, analysis, run] = await Promise.all([
          api<SimulationPortfolio>("/api/portfolio"),
          api<PortfolioResult | null>("/api/portfolio/current"),
          api<SimulationResult | null>("/api/simulation/current"),
        ]);
        if (cancelled || activeRevision.current !== revision || token !== generation.current) return;
        setPortfolio(snapshot);
        setSample(analysis?.sample ?? null);
        if (run) {
          const { calibration, simulations, initial_capital, years } = run.inputs;
          setInputs({ calibration, simulations, initial_capital, years });
          setResult(run);
          setDisplayCount(Math.min(10, run.inputs.simulations));
          setDisplayDraft(String(Math.min(10, run.inputs.simulations)));
        }
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
  }, [loading, contextRevision, returnReady, portfolioReady, restoredContext, reloadKey]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session || !portfolio?.id || !portfolio.simulation_ready || !inputsValid) return;
    const token = ++generation.current;
    const revision = session.revision;
    const portfolioId = portfolio.id;
    const submitted = { ...inputs, revision, portfolio_id: portfolioId };
    setRunning(true);
    setError(null);
    setPortfolioChanged(false);
    setResult(null);
    try {
      const calculated = await api<SimulationResult>("/api/simulation/run", submitted);
      if (token !== generation.current || activeRevision.current !== revision || calculated.portfolio_id !== portfolioId) return;
      setResult(calculated);
      setDisplayCount(Math.min(10, calculated.inputs.simulations));
      setDisplayDraft(String(Math.min(10, calculated.inputs.simulations)));
      await refreshSession();
    } catch (cause) {
      if (token === generation.current && activeRevision.current === revision) {
        const message = cause instanceof Error ? cause.message : String(cause);
        setError(message);
        setPortfolioChanged(/mudou/.test(message));
      }
    } finally {
      if (token === generation.current) setRunning(false);
    }
  }

  // The last run stays visible for the same portfolio; changed parameters mark it as outdated instead of hiding it.
  const sameContext = !!result && !!session && result.revision === session.revision && result.portfolio_id === portfolio?.id;
  const visible = sameContext ? result : null;
  const stale = !!visible && !sameInputs(inputs, visible.inputs);
  const simulationsValid = Number.isInteger(inputs.simulations) && inputs.simulations >= 10 && inputs.simulations <= 3000;
  const capitalValid = Number.isFinite(inputs.initial_capital) && inputs.initial_capital >= 100;
  const yearsValid = Number.isInteger(inputs.years) && inputs.years >= 1 && inputs.years <= 10;
  const inputsValid = simulationsValid && capitalValid && yearsValid;
  const calibrationText = inputs.calibration === "portfolio" ? "histórico do portfólio" : `últimos ${inputs.calibration} anos até hoje`;
  const individual = useMemo(() => visible ? displayedTrajectories(visible.charts.individual, displayCount) : null, [visible, displayCount]);
  const ready = !!portfolio?.id && portfolio.simulation_ready && portfolio.revision === session?.revision;
  const rawWeights = portfolio ? portfolio.tickers.map(ticker => ({ ticker, weight: portfolio.weights[ticker] })) : [];
  const total = rawWeights.reduce((sum, row) => sum + row.weight, 0);
  const allocation = visible?.allocation ?? rawWeights.map(row => ({ ticker: row.ticker, weight: row.weight / total }));

  const fan = useMemo(() => visible ? polishFigure(visible.charts.fan, "fan") : null, [visible]);
  const histogram = useMemo(() => visible ? polishFigure(visible.charts.histogram, "histogram") : null, [visible]);
  const trajectoryMax = visible ? Math.min(50, visible.inputs.simulations) : 50;

  const prerequisite = restoreFailed
    ? { title: "Não foi possível restaurar a carteira", message: "A carteira salva foi encontrada, mas seu histórico não pôde ser carregado. Abra Portfólio e carregue a análise novamente." }
    : session?.selection.length === 1
      ? { title: "Adicione mais um ativo", message: "A simulação precisa de pelo menos duas ações na carteira. Adicione mais um ativo na Portfólio para simular." }
    : session?.selection.length
      ? { title: "Atualize o portfólio", message: "A seleção atual não possui histórico analisado. Volte para Portfólio e clique em Carregar portfólio antes de rodar a simulação." }
      : { title: "Portfólio não configurado", message: "Configure seu portfólio na página Portfólio e carregue a análise para liberar a Simulação Monte Carlo." };

  const field = (id: string, label: string, valid: boolean, help: string, error: string, value: number, change: (value: number) => void, limits: { min: number; max?: number }) => <label>{label}
    <input type="number" required min={limits.min} max={limits.max} step={1} inputMode="numeric" disabled={running} aria-invalid={!valid} aria-describedby={valid ? `sim-help-${id}` : `sim-help-${id} sim-error-${id}`} value={Number.isNaN(value) ? "" : value} onChange={event => change(event.target.valueAsNumber)} />
    <span className="param-hint" id={`sim-help-${id}`}>{help}</span>
    {!valid && <span className="field-error" id={`sim-error-${id}`}>{error}</span>}
  </label>;
  const allocationList = <ul className="weights" aria-label="Pesos da carteira na simulação">{allocation.map(row => <li key={row.ticker}>
    <i className="swatch" style={{ background: tickerColor(Math.max(0, session?.selection.indexOf(row.ticker) ?? 0)) }} aria-hidden="true" /><b>{row.ticker}</b>
    <span className="weights-track" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(1, row.weight)) * 100}%` }} /></span>
    <span className="weights-value">{percent.format(row.weight)}</span>
  </li>)}</ul>;
  const notes = <>
    {sessionError && <Notice tone="negative">{sessionError}</Notice>}
    {error && <Notice tone="negative">{error}</Notice>}
    {portfolioChanged && <Notice tone="warning">A carteira mudou em outra aba ou página. <button type="button" onClick={() => { setPortfolioChanged(false); setError(null); setReloadKey(key => key + 1); void refreshSession(); }}>Atualizar carteira</button></Notice>}
  </>;

  if (loading || preparing) return <div className="desk">{notes}<Loading>Restaurando contexto da carteira…</Loading></div>;
  if (!ready) return <div className="desk desk-split">
    <Pane title="Parâmetros" className="pane-params" meta="bloqueado">
      {notes}
      <p className="locked-title">{prerequisite.title}</p>
      <p className="pane-lead">{prerequisite.message}</p>
      <div className="param-run"><Link className="button-link primary" href="/portfolio"><kbd>F2</kbd> Ir para Carteira</Link></div>
      <h3 className="pane-sub">O que a simulação calcula</h3>
      <ul className="plain-list">
        <li>Faixas de percentis do valor da carteira ao longo do horizonte.</li>
        <li>Distribuição dos valores finais, com quartis, média e desvio padrão.</li>
        <li>Probabilidade de ganho e estatísticas da cauda inferior.</li>
        <li>Calibração histórica e comparação entre modelo normal e reamostragem.</li>
      </ul>
    </Pane>
    <div className="board">
      <Pane title="Faixas de percentis" className="span-8" meta="aguardando carteira">
        <div className="example">
          <svg viewBox="0 0 600 220" role="img" aria-label="Ilustração de exemplo de faixas de percentis, sem dados">
            {[60, 110, 160].map(y => <line key={y} x1="0" x2="600" y1={y} y2={y} stroke="#1d2229" />)}
            <path d="M0 120 C150 100 300 60 600 20 L600 210 C300 175 150 150 0 120 Z" fill="rgb(255 178 36 / .08)" />
            <path d="M0 120 C150 110 300 90 600 70 L600 160 C300 150 150 135 0 120 Z" fill="rgb(255 178 36 / .18)" />
            <path d="M0 120 C150 116 300 112 600 108" stroke="#ffb224" strokeWidth="1.5" fill="none" />
            <line x1="0" x2="600" y1="120" y2="120" stroke="#8a929e" strokeDasharray="4 4" />
          </svg>
          <p>Faixa escura P25 a P75, clara P5 a P95, linha central a mediana, tracejado o capital inicial. Os números reais aparecem após rodar a simulação da sua carteira.</p>
        </div>
      </Pane>
      <Pane title="Estatísticas" className="span-4" meta="aguardando carteira">
        <Metrics items={["Valor esperado final", "CAGR do valor final médio", "Probabilidade de ganho", "Retorno final P5", "Valor final P5", "Média dos 5% menores"].map(label => ({ label, value: null }))} />
      </Pane>

    </div>
  </div>;

  return <div className="desk desk-split">
    <Pane title="Parâmetros" className="pane-params" meta={portfolio?.mode ?? undefined}>
      {notes}
      {restoredNotice && <Notice>Carteira salva restaurada. Confira ativos e pesos antes de rodar.</Notice>}
      <form className="param-group param-fields" onSubmit={submit} aria-busy={running} noValidate>
        <h3 className="pane-sub">Pontos de partida</h3>
        <div className="presets" role="group" aria-label="Pontos de partida">
          {simulationPresets.map(preset => {
            const active = inputs.simulations === preset.simulations && inputs.initial_capital === preset.initial_capital && inputs.years === preset.years;
            return <button key={preset.label} type="button" disabled={running} aria-pressed={active} title={preset.description} onClick={() => setInputs(current => ({ ...current, simulations: preset.simulations, initial_capital: preset.initial_capital, years: preset.years }))}>{preset.label}</button>;
          })}
        </div>
        <h3 className="pane-sub">Cenário</h3>
        {field("n", "Número de simulações (trajetórias)", simulationsValid, "Entre 10 e 3.000.", "Use um número inteiro entre 10 e 3.000.", inputs.simulations, value => setInputs(current => ({ ...current, simulations: value })), { min: 10, max: 3000 })}
        {field("c", "Capital inicial (R$)", capitalValid, "Mínimo de R$ 100.", "Informe um valor de pelo menos R$ 100.", inputs.initial_capital, value => setInputs(current => ({ ...current, initial_capital: value })), { min: 100 })}
        {field("y", "Horizonte (anos)", yearsValid, "Entre 1 e 10 anos.", "Use um número inteiro entre 1 e 10 anos.", inputs.years, value => setInputs(current => ({ ...current, years: value })), { min: 1, max: 10 })}
        <label>Janela de calibração
          <select disabled={running} aria-describedby="sim-help-w" value={String(inputs.calibration)} onChange={event => setInputs(current => ({ ...current, calibration: (event.target.value === "portfolio" ? "portfolio" : Number(event.target.value)) as CalibrationWindow }))}>
            <option value="portfolio">Histórico do Portfólio</option><option value="2">2 Anos</option><option value="3">3 Anos</option><option value="5">5 Anos</option>
          </select>
          <span className="param-hint" id="sim-help-w">Histórico usado para estimar retornos e risco.</span>
        </label>
        <div className="param-run"><button type="submit" className="primary" disabled={running || !inputsValid}>{running ? "Rodando simulação…" : "Rodar Simulação"}</button></div>
        <p className="param-hint">Parâmetros só valem ao rodar; se mudarem depois, o resultado fica marcado como desatualizado.</p>
      </form>
      <h3 className="pane-sub">Alocação usada</h3>
      {sample && <p className="param-hint">Calibração: {calibrationText}. Histórico: {periodLabel(sample)} ({number.format(sample.count)} retornos diários).</p>}
      {allocationList}
      <p className="param-hint">Pesos normalizados da carteira carregada; peso zero permanece zerado.</p>
      <details className="disclosure"><summary>Premissas e calibração</summary><div><p>A janela de calibração não altera os pesos. Janelas maiores podem reduzir o erro-padrão estimado, mas incluem regimes mais antigos.</p><p>O horizonte é independente da janela: 252 dias úteis por ano, até 10 anos. As datas simuladas incluem dias de semana, não o calendário de feriados da B3.</p></div></details>
    </Pane>
    <div className="board">
      {(stale || (visible && visible.warnings.length > 0)) && <div className="span-12 board-notes">
        {stale && <div role="status"><Notice tone="warning">Parâmetros alterados desde a última simulação. Os resultados abaixo são da anterior; clique em Rodar Simulação para atualizá-los.</Notice></div>}
        {visible?.warnings.map(warning => <Notice key={warning} tone="warning"><strong>Atenção: </strong>{warning}</Notice>)}
      </div>}
      {!visible && <>
        <Pane title="Estatísticas da simulação" className="span-12" meta={running ? undefined : "aguardando execução"}>
          {running ? <ElapsedLoading>Rodando simulações Monte Carlo multivariadas…</ElapsedLoading> : <Idle>Ajuste os parâmetros e pressione Rodar Simulação.</Idle>}
          <Metrics size="lg" items={["Valor esperado final", "CAGR do valor final médio (a.a.)", "Probabilidade de ganho"].map(label => ({ label, value: null }))} />
        </Pane>
        <Pane title="Faixas de percentis" className="span-8"><Idle>Evolução do valor da carteira por percentil.</Idle></Pane>
        <Pane title="Distribuição do valor final" className="span-4"><Idle>Histograma dos valores finais com quartis.</Idle></Pane>

      </>}
      {visible && <>
        <Pane id="sim-stats" title="Estatísticas da simulação" className="span-12" meta={`${yearsLabel(visible.inputs.years)} · ${number.format(visible.inputs.simulations)} trajetórias · capital ${formatValue(visible.inputs.initial_capital, "money")} · calibração ${visible.inputs.calibration === "portfolio" ? "histórico do portfólio" : `${visible.inputs.calibration} anos`}`}>
          <Metrics size="lg" items={[
            { label: "Valor esperado final", value: visible.stats.mean_terminal, unit: "money" },
            { label: "CAGR do valor final médio (a.a.)", value: visible.stats.mean_implied_cagr, unit: "fraction", digits: 1, help: "Crescimento anual implícito no valor final médio; não é a média dos CAGRs das trajetórias." },
            { label: "Probabilidade de ganho", value: visible.stats.gain_probability, unit: "fraction", digits: 1, meter: visible.stats.gain_probability, help: `Simulações acima do capital inicial no modelo normal. Erro Monte Carlo até ±${number.format(visible.monte_carlo_error_pp)} p.p. (95% aprox., pior caso); não inclui incerteza dos parâmetros.` },
            { label: "Retorno final P5", value: visible.stats.p5_return, unit: "fraction", digits: 1, help: "Percentil 5 dos valores finais; não é VaR nem perda máxima." },
          ]} />
          <Metrics items={[
            { label: "Valor final P5", value: visible.stats.p5_terminal, unit: "money" },
            { label: "Média dos 5% menores", value: visible.stats.lower_tail_mean, unit: "money", help: "Média dos finais ≤ P5, com empates." },
            { label: "Menor valor simulado", value: visible.stats.minimum, unit: "money" },
            { label: "CAGR do valor final P75 (a.a.)", value: visible.stats.p75_implied_cagr, unit: "fraction", digits: 1 },
            { label: "Maior valor simulado", value: visible.stats.maximum, unit: "money" },
          ]} />
        </Pane>
        <Pane id="sim-faixas" title="Faixas de percentis" className="span-8" meta="R$ · escura P25–P75 · clara P5–P95 · tracejado capital inicial" actions={<><ExportButton path={`/api/simulation/${visible.id}/summary.csv`}>Resumo CSV</ExportButton><ExportButton path={`/api/simulation/${visible.id}/percentiles.csv`}>Percentis CSV</ExportButton></>}>
          {fan && <Chart figure={fan} height={320} />}
        </Pane>
        <Pane id="sim-dist" title="Distribuição do valor final" className="span-4" meta="R$ · Q1, mediana e Q3 marcados">
          {histogram && <Chart figure={histogram} height={200} />}
          <Metrics items={[
            { label: "Mínimo", value: visible.distribution.minimum, unit: "money" }, { label: "Q1 (25%)", value: visible.distribution.q1, unit: "money" },
            { label: "Mediana", value: visible.distribution.median, unit: "money" }, { label: "Q3 (75%)", value: visible.distribution.q3, unit: "money" },
            { label: "Máximo", value: visible.distribution.maximum, unit: "money" }, { label: "Média", value: visible.distribution.mean, unit: "money" },
            { label: "Desvio padrão", value: visible.distribution.standard_deviation, unit: "money", help: "Amostral (ddof=1)." },
          ]} />
        </Pane>
        <Pane id="sim-trajetorias" title="Trajetórias individuais" className="span-8" actions={<label className="inline-field">Exibidas
          <input type="number" min={5} max={trajectoryMax} step={5} inputMode="numeric" aria-describedby="sim-help-display" value={displayDraft} onChange={event => {
            setDisplayDraft(event.target.value);
            const count = event.target.valueAsNumber;
            if (Number.isInteger(count) && count >= 5 && count <= trajectoryMax) setDisplayCount(count);
          }} onBlur={() => {
            const count = Number(displayDraft);
            const bounded = Number.isFinite(count) ? Math.max(5, Math.min(trajectoryMax, Math.trunc(count))) : displayCount;
            setDisplayCount(bounded);
            setDisplayDraft(String(bounded));
          }} /></label>}>
          {individual && <Chart figure={polishFigure(individual, "individual")} height={280} />}
          <p className="pane-foot" id="sim-help-display">De 5 a {trajectoryMax}. Só muda a exibição das primeiras trajetórias; não roda de novo nem altera estatísticas.</p>
        </Pane>
        <Pane id="sim-sintese" title="Síntese do analista" className="span-4" meta={`${yearsLabel(visible.inputs.years)} · ${number.format(visible.inputs.simulations)} simulações`}>
          <ul className="plain-list">
            <li>{percent.format(visible.stats.gain_probability)} dos cenários do modelo normal terminaram acima do capital inicial.</li>
            <li>CAGR implícito no valor final médio: {percent.format(visible.stats.mean_implied_cagr)} a.a. ao longo de {yearsLabel(visible.inputs.years)}.</li>
            <li>Retorno final P5: {percent.format(visible.stats.p5_return)}.</li>
            <li>Amplitude entre P5 e P95: {percent.format(visible.spread_fraction)} do capital inicial.</li>
          </ul>
          <p className="pane-foot">Modelo normal de retornos log multivariados ({visible.calibration.count} observações, {periodLabel(visible.calibration)}); pesos rebalanceados diariamente. Sem taxas e impostos. Cenários, não previsões nem recomendação.</p>
        </Pane>
        <Pane id="sim-calibracao" title="Calibração histórica" className="span-6" flush>
          <p className="pane-lead">{visible.calibration.count} retornos diários completos ({periodLabel(visible.calibration)}). Média log anual: {percent.format(visible.calibration.annual_log_mean)} ±{number.format(visible.calibration.iid_standard_error * 100)} p.p. (1 erro-padrão IID; não mede a incerteza dos retornos futuros).</p>
          {visible.calibration.sensitivity.length > 0 && <>
            <Table caption="Sensibilidade à janela histórica (pesos fixos)" rowKey="window" columns={[{ key: "window", label: "Janela" }, { key: "count", label: "Retornos", unit: "integer" }, { key: "period", label: "Período" }, { key: "annual_log_mean", label: "Média log a.a.", unit: "fraction", digits: 1 }, { key: "error", label: "Erro-padrão p.p.", unit: "number", digits: 1 }]} rows={visible.calibration.sensitivity.map(row => ({ ...row, window: `${row.years} anos`, period: periodLabel(row), error: row.iid_standard_error * 100 }))} />
            <p className="pane-foot">Janelas aninhadas com a mesma alocação e o mesmo fim. O erro-padrão pressupõe dias independentes; diferenças entre janelas podem indicar regimes de mercado.</p>
          </>}
        </Pane>
        <Pane id="sim-modelos" title="Comparação de modelos" className="span-6" flush>
          <Table caption="Comparação de modelos" rowKey="model" columns={[{ key: "model", label: "Modelo" }, { key: "p5", label: "Final P5", unit: "money" }, { key: "p95", label: "Final P95", unit: "money" }, { key: "gain_probability", label: "Ganho s/ capital", unit: "fraction", digits: 1 }]} rows={visible.comparison} />
          <p className="pane-foot">O modelo normal usa retornos log diários correlacionados. A reamostragem sorteia dias históricos inteiros (preserva choques simultâneos), mas não cria crises ausentes nem preserva sequências de volatilidade. A diferença indica sensibilidade às hipóteses, não incerteza coberta pelas faixas.</p>
        </Pane>
        <div className="span-12 board-foot"><p>Próximo <Link href="/noticias"><kbd>F4</kbd> Notícias</Link> lê o tom das notícias dos ativos da carteira.</p></div>
      </>}
    </div>
  </div>;
}
