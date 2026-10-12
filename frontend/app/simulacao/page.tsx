"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import "./simulacao.css";
import { useApp } from "../../components/AppProvider";
import Chart from "../../components/Chart";
import { ElapsedLoading } from "../../components/PortfolioParts";
import { ExportButton, formatValue, Icon, Loading, Metrics, Notice, PageHeader, Section, Table } from "../../components/ui";
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

  return <>
    <PageHeader title="Simulação de portfólio" description="Projete faixas de retorno e risco a partir dos pesos definidos na carteira." icon="simulation" />
    {sessionError && <Notice tone="negative">{sessionError}</Notice>}
    {error && <Notice tone="negative">{error}</Notice>}
    {portfolioChanged && <Notice tone="warning">A carteira mudou em outra aba ou página. <button type="button" onClick={() => { setPortfolioChanged(false); setError(null); setReloadKey(key => key + 1); void refreshSession(); }}>Atualizar carteira</button></Notice>}
    {loading || preparing ? <Loading>Restaurando contexto da carteira…</Loading> : !ready ?
      <section className="sim-prereq" aria-labelledby="sim-prereq-title">
        <div className="sim-prereq-main">
          <span className="sim-prereq-icon"><Icon name="simulation" /></span>
          <h2 id="sim-prereq-title" className="sim-prereq-title">{prerequisite.title}</h2>
          <p className="sim-prereq-message">{prerequisite.message}</p>
          <Link className="button-link sim-prereq-link" href="/portfolio">Ir para Portfólio</Link>
        </div>
        <div className="sim-prereq-outputs">
          <h3>O que a simulação calcula</h3>
          <ul>
            <li>Faixas de percentis do valor do portfólio ao longo do horizonte escolhido.</li>
            <li>Distribuição dos valores finais, com quartis, média e desvio padrão.</li>
            <li>Probabilidade de ganho e estatísticas da cauda inferior.</li>
            <li>Calibração histórica e comparação entre o modelo normal e a reamostragem.</li>
          </ul>
          <p>Os cenários dependem do histórico da carteira e das premissas do modelo.</p>
        </div>
      </section> : <>
      {restoredNotice && <Notice>Carteira salva restaurada. Confira os ativos e pesos antes de rodar a simulação.</Notice>}
      <div className="sim-workbench">
        <form className="sim-form" onSubmit={submit} aria-busy={running} noValidate>
          <div className="task-panel sim-task">
            <h2 className="ui-section-heading">Parâmetros da simulação</h2>
            <p className="sim-step">Ponto de partida</p>
            <div className="sim-presets" role="group" aria-label="Pontos de partida">
              {simulationPresets.map(preset => {
                const active = inputs.simulations === preset.simulations && inputs.initial_capital === preset.initial_capital && inputs.years === preset.years;
                return <button key={preset.label} type="button" disabled={running} aria-pressed={active} onClick={() => setInputs(current => ({ ...current, simulations: preset.simulations, initial_capital: preset.initial_capital, years: preset.years }))}>
                  {active ? `✓ ${preset.label} (selecionado)` : preset.label}<small>{preset.description}</small>
                </button>;
              })}
            </div>
            <div className="controls-grid">
              <label>Número de simulações (trajetórias)
                <input type="number" required min={10} max={3000} step={1} inputMode="numeric" disabled={running} aria-invalid={!simulationsValid} aria-describedby={simulationsValid ? "sim-help-n" : "sim-help-n sim-error-n"} value={Number.isNaN(inputs.simulations) ? "" : inputs.simulations} onChange={event => setInputs(current => ({ ...current, simulations: event.target.valueAsNumber }))} />
                <span className="sim-hint" id="sim-help-n">Entre 10 e 3.000.</span>
                {!simulationsValid && <span className="sim-error" id="sim-error-n">Use um número inteiro entre 10 e 3.000.</span>}
              </label>
              <label>Capital inicial (R$)
                <input type="number" required min={100} step={1} inputMode="numeric" disabled={running} aria-invalid={!capitalValid} aria-describedby={capitalValid ? "sim-help-c" : "sim-help-c sim-error-c"} value={Number.isNaN(inputs.initial_capital) ? "" : inputs.initial_capital} onChange={event => setInputs(current => ({ ...current, initial_capital: event.target.valueAsNumber }))} />
                <span className="sim-hint" id="sim-help-c">Mínimo de R$ 100.</span>
                {!capitalValid && <span className="sim-error" id="sim-error-c">Informe um valor de pelo menos R$ 100.</span>}
              </label>
              <label>Horizonte (anos)
                <input type="number" required min={1} max={10} step={1} inputMode="numeric" disabled={running} aria-invalid={!yearsValid} aria-describedby={yearsValid ? "sim-help-y" : "sim-help-y sim-error-y"} value={Number.isNaN(inputs.years) ? "" : inputs.years} onChange={event => setInputs(current => ({ ...current, years: event.target.valueAsNumber }))} />
                <span className="sim-hint" id="sim-help-y">Entre 1 e 10 anos.</span>
                {!yearsValid && <span className="sim-error" id="sim-error-y">Use um número inteiro entre 1 e 10 anos.</span>}
              </label>
              <label>Janela de calibração
                <select disabled={running} aria-describedby="sim-help-w" value={String(inputs.calibration)} onChange={event => setInputs(current => ({ ...current, calibration: (event.target.value === "portfolio" ? "portfolio" : Number(event.target.value)) as CalibrationWindow }))}>
                  <option value="portfolio">Histórico do Portfólio</option><option value="2">2 Anos</option><option value="3">3 Anos</option><option value="5">5 Anos</option>
                </select>
                <span className="sim-hint" id="sim-help-w">Histórico usado para estimar retornos e risco.</span>
              </label>
            </div>
            <p className="discreet-status">Os parâmetros só são aplicados ao clicar em Rodar Simulação; se mudarem depois, o resultado anterior fica marcado como desatualizado. A janela de calibração não altera os pesos. Janelas maiores podem reduzir o erro-padrão estimado, mas incluem regimes mais antigos. O horizonte é independente da janela: são 252 dias úteis por ano, com limite de 10 anos. As datas simuladas incluem dias de semana, não o calendário de feriados da B3.</p>
            <div className="actions">
              <button type="submit" className="primary" disabled={running || !inputsValid}>{running ? "Rodando simulação…" : "Rodar Simulação"}</button>
            </div>
          </div>
        </form>
        <aside className="sim-context" aria-label="Alocação usada na simulação">
          <Section title="Alocação usada na simulação">
            {portfolio?.mode && <p className="sim-mode">{portfolio.mode}</p>}
            {sample && <p className="sim-window">Calibração: {calibrationText}. Histórico do portfólio: {periodLabel(sample)} ({number.format(sample.count)} retornos diários).</p>}
            <ul className="sim-allocation" aria-label="Pesos da carteira na simulação">
              {allocation.map(row => <li key={row.ticker}>
                <span className="sim-ticker">{row.ticker}</span>
                <span className="sim-bar" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(1, row.weight)) * 100}%` }} /></span>
                <span className="sim-weight">{percent.format(row.weight)}</span>
              </li>)}
            </ul>
            <p className="discreet-status">Pesos normalizados a partir da carteira carregada; ativos com peso zero permanecem zerados.</p>
          </Section>
        </aside>
      </div>
      {running && <ElapsedLoading>Rodando simulações Monte Carlo multivariadas…</ElapsedLoading>}
      {!visible && !running && <div className="sim-idle"><Icon name="chart" /><p>Configure os parâmetros acima e clique em Rodar Simulação para ver os resultados.</p></div>}
      {visible && <div className="sim-results">
        {stale && <div role="status"><Notice tone="warning">Parâmetros alterados desde a última simulação. Os resultados abaixo são da simulação anterior; clique em Rodar Simulação para atualizá-los.</Notice></div>}
        {visible.warnings.map(warning => <Notice key={warning} tone="warning"><strong>Atenção: </strong>{warning}</Notice>)}
        <Section title="Estatísticas da Simulação Monte Carlo">
          <dl className="sim-run">
            <div><dt>Horizonte</dt><dd>{yearsLabel(visible.inputs.years)}</dd></div>
            <div><dt>Trajetórias</dt><dd>{number.format(visible.inputs.simulations)}</dd></div>
            <div><dt>Capital inicial</dt><dd>{formatValue(visible.inputs.initial_capital, "money")}</dd></div>
            <div><dt>Janela de calibração</dt><dd>{visible.inputs.calibration === "portfolio" ? "Histórico do Portfólio" : `${visible.inputs.calibration} anos`}</dd></div>
          </dl>
          <div className="sim-kpis"><Metrics items={[
            { label: "Valor Esperado Final", value: visible.stats.mean_terminal, unit: "money" },
            { label: "CAGR do valor final médio (a.a.)", value: visible.stats.mean_implied_cagr, unit: "fraction", digits: 1, help: "Crescimento anual implícito no valor final médio; não é a média dos CAGRs das trajetórias." },
            { label: "Probabilidade de Ganho", value: visible.stats.gain_probability, unit: "fraction", digits: 1, meter: visible.stats.gain_probability, help: `Percentual de simulações acima do capital inicial sob o modelo normal. Erro Monte Carlo de até ±${number.format(visible.monte_carlo_error_pp)} p.p. (95% aproximado, pior caso); não inclui incerteza dos parâmetros.` },
          ]} /></div>
          <h3 className="sim-subhead">Cauda inferior e extremos</h3>
          <Metrics items={[
            { label: "Retorno final P5", value: visible.stats.p5_return, unit: "fraction", digits: 1, help: "Retorno no percentil 5 dos valores finais simulados; não é uma estimativa de VaR nem representa perda máxima." },
            { label: "Valor final P5", value: visible.stats.p5_terminal, unit: "money" },
            { label: "Média dos 5% menores valores", value: visible.stats.lower_tail_mean, unit: "money" },
            { label: "Menor valor simulado", value: visible.stats.minimum, unit: "money" },
            { label: "CAGR do valor final P75 (a.a.)", value: visible.stats.p75_implied_cagr, unit: "fraction", digits: 1 },
            { label: "Maior valor simulado", value: visible.stats.maximum, unit: "money" },
          ]} />
          <dl className="sim-notes">
            <div><dt>Valor final P5</dt><dd>Percentil 5 dos valores finais simulados; não é uma medida VaR de perda nem representa perda máxima.</dd></div>
            <div><dt>Média dos 5% menores valores</dt><dd>Média dos resultados finais menores ou iguais ao P5 (inclui empates) na cauda inferior simulada.</dd></div>
          </dl>
        </Section>
        <Section title="Faixas de percentis">
          <p className="sim-chart-lead">Evolução do valor do portfólio em reais (R$): a faixa mais escura cobre de P25 a P75 e a mais clara, de P5 a P95. A linha tracejada marca o capital inicial.</p>
          {fan && <Chart figure={fan} />}
          <div className="sim-section-actions"><ExportButton path={`/api/simulation/${visible.id}/summary.csv`}>Resumo (CSV)</ExportButton><ExportButton path={`/api/simulation/${visible.id}/percentiles.csv`}>Percentis por Data (CSV)</ExportButton></div>
        </Section>
        <Section title="Distribuição do Valor Final do Portfólio">
          <p className="sim-chart-lead">Frequência dos valores finais simulados, em reais (R$), com Q1, mediana e Q3 marcados.</p>
          {histogram && <Chart figure={histogram} />}
          <Metrics items={[
            { label: "Mínimo", value: visible.distribution.minimum, unit: "money" }, { label: "Q1 (25%)", value: visible.distribution.q1, unit: "money" },
            { label: "Mediana (50%)", value: visible.distribution.median, unit: "money" }, { label: "Q3 (75%)", value: visible.distribution.q3, unit: "money" },
            { label: "Máximo", value: visible.distribution.maximum, unit: "money" }, { label: "Média", value: visible.distribution.mean, unit: "money" },
            { label: "Desvio Padrão", value: visible.distribution.standard_deviation, unit: "money", help: "Desvio padrão amostral dos valores finais (ddof=1)." },
          ]} />
        </Section>
        <Section title="Trajetórias Individuais das Simulações">
          <div className="sim-inline-control">
            <label>Número de trajetórias exibidas
              <input type="number" min={5} max={trajectoryMax} step={5} inputMode="numeric" aria-describedby="sim-help-display" value={displayDraft} onChange={event => {
                setDisplayDraft(event.target.value);
                const count = event.target.valueAsNumber;
                if (Number.isInteger(count) && count >= 5 && count <= trajectoryMax) setDisplayCount(count);
              }} onBlur={() => {
                const count = Number(displayDraft);
                const bounded = Number.isFinite(count) ? Math.max(5, Math.min(trajectoryMax, Math.trunc(count))) : displayCount;
                setDisplayCount(bounded);
                setDisplayDraft(String(bounded));
              }} />
            </label>
            <p className="caption" id="sim-help-display">De 5 a {trajectoryMax}. Só muda a exibição das primeiras trajetórias; não roda novamente a simulação nem altera estatísticas ou percentis.</p>
          </div>
          {individual && <Chart figure={polishFigure(individual, "individual")} />}
        </Section>
        <Section title="Calibração histórica">
          <p className="sim-prose sim-lead">{visible.calibration.count} retornos diários completos ({periodLabel(visible.calibration)}). Média log anual estimada: {percent.format(visible.calibration.annual_log_mean)} ±{number.format(visible.calibration.iid_standard_error * 100)} p.p. (1 erro-padrão IID; não mede a incerteza dos retornos futuros).</p>
          {visible.calibration.sensitivity.length > 0 && <>
            <h3 className="sim-subhead">Sensibilidade à janela histórica (pesos fixos)</h3>
            <Table columns={[{ key: "window", label: "Janela" }, { key: "count", label: "Retornos completos", unit: "integer" }, { key: "period", label: "Período" }, { key: "annual_log_mean", label: "Média log anual", unit: "fraction", digits: 1 }, { key: "error", label: "Erro-padrão IID (p.p.)", unit: "number", digits: 1 }]} rows={visible.calibration.sensitivity.map(row => ({ ...row, window: `${row.years} anos`, period: periodLabel(row), error: row.iid_standard_error * 100 }))} />
            <p className="discreet-status sim-prose">Janelas aninhadas usam a mesma alocação e o mesmo endpoint. O erro-padrão pressupõe dias independentes; mudanças entre janelas podem indicar efeito de regimes de mercado.</p>
          </>}
        </Section>
        <Section title="Comparação de modelos">
          <Table columns={[{ key: "model", label: "Modelo" }, { key: "p5", label: "Valor final P5", unit: "money" }, { key: "p95", label: "Valor final P95", unit: "money" }, { key: "gain_probability", label: "Ganho sobre o capital inicial", unit: "fraction", digits: 1 }]} rows={visible.comparison} />
          <p className="discreet-status sim-prose">O modelo normal usa retornos logarítmicos diários correlacionados. A reamostragem sorteia dias históricos inteiros (preserva choques simultâneos entre ativos), mas não cria crises ausentes da amostra nem preserva sequências de volatilidade. A diferença entre modelos indica sensibilidade às hipóteses, não incerteza coberta pelas faixas.</p>
        </Section>
        <Section title="Síntese do Analista">
          <div className="analyst-synthesis-card sim-synthesis">
            <h3>Projeção {yearsLabel(visible.inputs.years)} — {number.format(visible.inputs.simulations)} simulações</h3>
            <ul>
              <li>{percent.format(visible.stats.gain_probability)} dos cenários do modelo normal terminaram acima do capital inicial.</li>
              <li>CAGR implícito no valor final médio: {percent.format(visible.stats.mean_implied_cagr)} a.a. ao longo de {yearsLabel(visible.inputs.years)}.</li>
              <li>Retorno final P5: {percent.format(visible.stats.p5_return)}.</li>
              <li>Amplitude entre P5 e P95: {percent.format(visible.spread_fraction)} do capital inicial.</li>
            </ul>
            <p className="sim-prose">Modelo normal de retornos logarítmicos multivariados ({visible.calibration.count} observações, {periodLabel(visible.calibration)}); pesos rebalanceados diariamente em ambos os modelos. Não inclui taxas nem impostos. Resultados são cenários, não previsões nem recomendação de investimento.</p>
          </div>
          <div className="next-step-card">
            <h3>Próximo passo</h3>
            <p>Complemente a leitura da projeção com as notícias das empresas.</p>
            <Link className="button-link" href="/noticias">Abrir Notícias</Link>
          </div>
        </Section>
      </div>}
    </>}
  </>;
}
