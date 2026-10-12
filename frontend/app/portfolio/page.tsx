"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useApp } from "../../components/AppProvider";
import Chart from "../../components/Chart";
import { EmptyState, ExportButton, Loading, Metrics, Notice, PageHeader, Section, Table, TickerPicker } from "../../components/ui";
import { ElapsedLoading, MonthlyReturns, ResultSummary, WeightList } from "../../components/PortfolioParts";
import { api } from "../../lib/api";
import { handoffTickers, period, strategies } from "../../lib/portfolio";
import "./portfolio.css";
import type { PortfolioInputs, PortfolioResult, StressResult, Strategy } from "../../lib/portfolio";

/** Thins the portfolio value series already plotted below into a sparkline for the headline tile. */
function valueTrend(result: PortfolioResult): number[] | undefined {
  const values = (result.charts.value.data[0] as { y?: unknown[] } | undefined)?.y;
  if (!Array.isArray(values)) return undefined;
  const numbers = values.filter((value): value is number => typeof value === "number");
  const step = Math.max(1, Math.ceil(numbers.length / 120));
  return numbers.filter((_, index) => index % step === 0 || index === numbers.length - 1);
}

/** Mirrors the API limits in backend/portfolio.py so invalid edits never reach the server. */
function withinLimits(inputs: PortfolioInputs): boolean {
  return inputs.capital >= 100 && inputs.capital <= 1000000 && inputs.lookback_days >= 60 && inputs.lookback_days <= 5000 && inputs.gamma >= 0 && inputs.gamma <= 3 && Object.values(inputs.manual_shares).every(value => Number.isInteger(value) && value >= 0);
}

function hasManualShares(inputs: PortfolioInputs, selection: readonly string[]): boolean {
  return inputs.strategy !== "manual" || selection.some(ticker => inputs.manual_shares[ticker] > 0);
}

/** Pause after the last edit before recalculating, so typing a value does not send one request per keystroke. */
const AUTO_RECALC_DELAY_MS = 500;
const presetWindows = ["730", "365", "1095", "1825", "180"];

export default function PortfolioPage() {
  const { session, stocks, loading, error: appError, setSelection, refreshSession } = useApp();
  const [inputs, setInputs] = useState<PortfolioInputs>({ strategy: "min_volatility", capital: 10000, lookback_days: 730, manual_shares: {}, gamma: 1 });
  const [windowChoice, setWindowChoice] = useState("730");
  const [customDays, setCustomDays] = useState(500);
  const [loadedSelection, setLoadedSelection] = useState<string | null>(null);
  const [result, setResult] = useState<PortfolioResult | null>(null);
  const [resultKey, setResultKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectionBusy, setSelectionBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [longHistory, setLongHistory] = useState(false);
  const [stressResult, setStressResult] = useState<StressResult | null>(null);
  const [stressBusy, setStressBusy] = useState(false);
  const requestGeneration = useRef(0);
  const analysisPending = useRef(false);
  const stressGeneration = useRef(0);
  const lastSubmitted = useRef("");
  const handoffConsumed = useRef(false);
  const selectionKey = session ? JSON.stringify([session.revision, session.selection]) : "";
  const currentSelection = useRef(selectionKey);
  currentSelection.current = selectionKey;
  const fingerprint = JSON.stringify([selectionKey, inputs]);
  const currentFingerprint = useRef(fingerprint);
  currentFingerprint.current = fingerprint;
  const validInputs = withinLimits(inputs);
  const manualValid = hasManualShares(inputs, session?.selection ?? []);

  useEffect(() => () => {
    requestGeneration.current += 1;
    stressGeneration.current += 1;
  }, []);

  useEffect(() => {
    if (!session || !stocks.length || handoffConsumed.current) return;
    handoffConsumed.current = true;
    const url = new URL(window.location.href);
    const raw = url.searchParams.get("portfolio_tickers");
    if (raw !== null) {
      url.searchParams.delete("portfolio_tickers");
      window.history.replaceState(null, "", url);
      const tickers = handoffTickers(raw, new Set(stocks.map(stock => stock.ticker)));
      if (tickers.length) {
        setSelectionBusy(true);
        setSelection(tickers).then(() => setNotice("Seleção carregada da análise de ativos. Confira os papéis e clique em Carregar portfólio.")).catch(exc => setError(String(exc.message ?? exc))).finally(() => setSelectionBusy(false));
      }
    } else if (session.restored) {
      setNotice("Carteira salva restaurada. Clique em Carregar portfólio para atualizar a análise.");
    }
  }, [session, stocks, setSelection]);

  useEffect(() => {
    if (loadedSelection !== null && loadedSelection !== selectionKey) {
      requestGeneration.current += 1;
      stressGeneration.current += 1;
      setLoadedSelection(null);
      setResult(null);
      setStressResult(null);
      if (!analysisPending.current) setBusy(false);
      setStressBusy(false);
      setNotice("A seleção mudou. Clique em Carregar portfólio para atualizar a análise.");
    }
  }, [selectionKey, loadedSelection]);

  async function runAnalysis() {
    if (!session || selectionBusy || analysisPending.current) return;
    const generation = ++requestGeneration.current;
    const submittedSelection = selectionKey;
    const submittedKey = fingerprint;
    lastSubmitted.current = submittedKey;
    analysisPending.current = true;
    setLoadedSelection(submittedSelection);
    setBusy(true);
    setError(null);
    setNotice(null);
    setResult(null);
    setStressResult(null);
    setLongHistory(false);
    stressGeneration.current += 1;
    setStressBusy(false);
    try {
      const manualShares = Object.fromEntries(session.selection.map(ticker => [ticker, inputs.manual_shares[ticker] ?? 0]));
      const value = await api<PortfolioResult>("/api/portfolio/analyze", { ...inputs, manual_shares: manualShares, tickers: session.selection, revision: session.revision });
      if (generation !== requestGeneration.current || submittedSelection !== currentSelection.current || submittedKey !== currentFingerprint.current) return;
      setResult(value);
      setResultKey(submittedKey);
      setStressResult(value.stress);
      await refreshSession();
    } catch (exc) {
      if (generation === requestGeneration.current && submittedKey === currentFingerprint.current) {
        setError(exc instanceof Error ? exc.message : String(exc));
        await refreshSession();
      }
    } finally {
      analysisPending.current = false;
      setBusy(false);
    }
  }

  const runAnalysisRef = useRef(runAnalysis);
  runAnalysisRef.current = runAnalysis;
  useEffect(() => {
    if (!session?.portfolio_ready || session.restored || result || busy || selectionBusy) return;
    const requested = fingerprint;
    let cancelled = false;
    api<PortfolioResult | null>("/api/portfolio/current").then(value => {
      // Only the owner's analysis for this selection is shown; edits made while loading win over the stored parameters.
      if (cancelled || !value || value.restored || requested !== currentFingerprint.current) return;
      const restored: PortfolioInputs = { strategy: value.inputs.strategy, capital: value.inputs.capital, lookback_days: value.inputs.lookback_days, manual_shares: value.inputs.manual_shares, gamma: value.inputs.gamma };
      const key = JSON.stringify([selectionKey, restored]);
      const preset = presetWindows.includes(String(restored.lookback_days));
      lastSubmitted.current = key;
      setInputs(restored);
      setWindowChoice(preset ? String(restored.lookback_days) : "custom");
      setCustomDays(restored.lookback_days);
      setLoadedSelection(selectionKey);
      setResult(value);
      setResultKey(key);
      setStressResult(value.stress);
      setNotice("Resultado desta sessão restaurado. Os parâmetros foram preenchidos com os da última análise.");
    }).catch(() => {});
    return () => { cancelled = true; };
    // Runs once per session revision: the restored analysis is the one the server already holds.
  }, [session?.revision, session?.portfolio_ready, session?.restored]);
  useEffect(() => {
    if (busy || !session || loadedSelection !== selectionKey || lastSubmitted.current === fingerprint || !validInputs || !manualValid) return;
    const timer = window.setTimeout(() => void runAnalysisRef.current(), AUTO_RECALC_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [fingerprint, loadedSelection, selectionKey, session, busy, validInputs, manualValid]);

  async function changeSelection(tickers: string[]) {
    if (selectionBusy) return;
    requestGeneration.current += 1;
    stressGeneration.current += 1;
    setLoadedSelection(null);
    setResult(null);
    if (!analysisPending.current) setBusy(false);
    setStressBusy(false);
    setSelectionBusy(true);
    try { await setSelection(tickers); setNotice("Confira a seleção e clique em Carregar portfólio."); }
    catch (exc) { setError(exc instanceof Error ? exc.message : String(exc)); }
    finally { setSelectionBusy(false); }
  }

  async function clearPortfolio() {
    requestGeneration.current += 1;
    stressGeneration.current += 1;
    setLoadedSelection(null);
    setResult(null);
    setStressResult(null);
    if (!analysisPending.current) setBusy(false);
    setStressBusy(false);
    setSelectionBusy(true);
    try { await api("/api/portfolio", undefined, "DELETE"); await refreshSession(); setNotice("Carteira e seleção removidas."); }
    catch (exc) { setError(exc instanceof Error ? exc.message : String(exc)); }
    finally { setSelectionBusy(false); }
  }

  async function refreshQuotes() {
    requestGeneration.current += 1;
    stressGeneration.current += 1;
    setLoadedSelection(null);
    setResult(null);
    setStressResult(null);
    if (!analysisPending.current) setBusy(false);
    setStressBusy(false);
    setSelectionBusy(true);
    try { const response = await api<{ message: string }>("/api/portfolio/refresh", {}); await refreshSession(); setNotice(response.message); }
    catch (exc) { setError(exc instanceof Error ? exc.message : String(exc)); }
    finally { setSelectionBusy(false); }
  }

  async function loadStress(long: boolean) {
    if (!result) return;
    const generation = ++stressGeneration.current;
    const key = fingerprint;
    setLongHistory(long);
    setStressBusy(true);
    setError(null);
    try {
      const value = await api<StressResult>(`/api/portfolio/${result.id}/stress`, { long_history: long });
      if (generation === stressGeneration.current && key === currentFingerprint.current) setStressResult(value);
    } catch (exc) { if (generation === stressGeneration.current) setError(exc instanceof Error ? exc.message : String(exc)); }
    finally { if (generation === stressGeneration.current) setStressBusy(false); }
  }

  if (loading || !session) return appError ? <Notice tone="negative">{appError}</Notice> : <Loading>Carregando lista de ações e seleção…</Loading>;
  const activeResult = resultKey === fingerprint ? result : null;
  const staleResult = !!result && resultKey !== fingerprint && !busy;
  const count = session.selection.length;
  const core = [["total_return", "Retorno Total", "percent"], ["volatility", "Volatilidade Anual", "percent"], ["sharpe", "Índice Sharpe", "ratio"], ["sortino", "Índice Sortino", "ratio"], ["max_drawdown", "Queda máxima (Max Drawdown)", "percent"]] as const;
  const extra = [["annual_return", "Retorno Anualizado", "percent"], ["beta", "Beta vs IBOV", "ratio"], ["annual_alpha", "Alpha Anual", "percent"], ["daily_var", "VaR Diário 95% (paramétrico normal)", "percent"], ["historical_var_95", "VaR Histórico 95% (diário)", "percent"], ["cvar_95", "CVaR 95% (diário)", "percent"], ["information_ratio", "Information Ratio (anualizado)", "ratio"]] as const;

  return <>
    <PageHeader title="Otimização de portfólio" description="Monte uma carteira, compare risco e retorno e revise os pesos antes de investir." icon="portfolio" />
    {appError && <Notice tone="negative">{appError}</Notice>}
    {notice && <Notice>{notice}</Notice>}
    {error && <Notice tone="negative">{error}</Notice>}
    <Section title="Montagem da carteira">
      <div className="task-panel pf-setup">
        <div className="pf-setup-grid">
          <div className="pf-group">
            <h3 className="pf-step">1. Escolha os papéis</h3>
            <p className="pf-hint">Selecione de 2 a 20 ações para compor a carteira.</p>
            <TickerPicker value={session.selection} stocks={stocks} max={20} label="Ações da carteira" onChange={changeSelection} />
          </div>
          <div className="pf-group">
            <h3 className="pf-step">2. Defina os parâmetros</h3>
            <div className="controls-grid">
              <label>Período de análise<select disabled={busy} value={windowChoice} onChange={event => { const value = event.target.value; setWindowChoice(value); setInputs(previous => ({ ...previous, lookback_days: value === "custom" ? customDays : Number(value) })); }}>
                <option value="730">2 Anos (Padrão)</option><option value="365">1 Ano</option><option value="1095">3 Anos</option><option value="1825">5 Anos</option><option value="180">6 Meses</option><option value="custom">Personalizado (Dias)</option>
              </select></label>
              {windowChoice === "custom" && <label>Dias de lookback<input type="number" disabled={busy} min={60} max={5000} step={10} value={customDays} onChange={event => { const days = Number(event.target.value); setCustomDays(days); setInputs(previous => ({ ...previous, lookback_days: days })); }} /></label>}
              <label>Capital disponível (R$)<input type="number" disabled={busy} min={100} max={1000000} step={100} value={inputs.capital} onChange={event => setInputs(previous => ({ ...previous, capital: Number(event.target.value) }))} /></label>
              <label className="pf-wide">Estratégia de alocação<select disabled={busy} value={inputs.strategy} onChange={event => setInputs(previous => ({ ...previous, strategy: event.target.value as Strategy }))}>{strategies.map(strategy => <option key={strategy.value} value={strategy.value}>{strategy.label}</option>)}</select></label>
            </div>
            <p className="caption">Dados a partir de {new Date(Date.now() - inputs.lookback_days * 86400000).toLocaleDateString("pt-BR")} · dias corridos.</p>
            {inputs.strategy === "manual" && <div className="pf-settings">
              <h3 className="pf-step">Cotas por ativo</h3><p className="pf-hint">Os pesos serão calculados pelo valor de mercado das cotas informadas; o capital disponível é usado só no histórico de valor do Portfólio. A Simulação tem capital inicial próprio.</p>
              <div className="controls-grid">{session.selection.map(ticker => <label key={ticker}>Cotas de {ticker}<input type="number" disabled={busy} min={0} step={1} value={inputs.manual_shares[ticker] ?? 0} onChange={event => setInputs(previous => ({ ...previous, manual_shares: { ...previous.manual_shares, [ticker]: Number(event.target.value) } }))} /></label>)}</div>
            </div>}
            {inputs.strategy === "markowitz" && <div className="pf-settings">
              <h3 className="pf-step">Diversificação</h3><p className="pf-hint">Controle quanto a otimização evita concentrar o capital.</p>
              <div className="button-row pf-presets">{[{ label: "Concentrada", gamma: 0 }, { label: "Equilibrada", gamma: 1 }, { label: "Diversificada", gamma: 2 }].map(preset => <button type="button" key={preset.label} disabled={busy} aria-pressed={inputs.gamma === preset.gamma} className={inputs.gamma === preset.gamma ? "is-active" : undefined} onClick={() => setInputs(previous => ({ ...previous, gamma: preset.gamma }))}>{preset.label}</button>)}</div>
              <label className="pf-gamma">Diversificação (regularização L2, de 0 a 3)<input type="number" disabled={busy} min={0} max={3} step={.1} value={inputs.gamma} onChange={event => setInputs(previous => ({ ...previous, gamma: Number(event.target.value) }))} /></label>
            </div>}
          </div>
        </div>
        {!manualValid && <Notice tone="warning">A alocação manual precisa ter ao menos uma cota positiva.</Notice>}
        {!validInputs && <Notice tone="warning">Confira os limites de capital, histórico, cotas inteiras e diversificação.</Notice>}
        <div className="actions">
          <button type="button" className="primary pf-actions-primary" disabled={busy || count < 2 || !manualValid || !validInputs || selectionBusy} onClick={() => void runAnalysis()}>Carregar portfólio</button>
          <button type="button" disabled={selectionBusy} onClick={() => void refreshQuotes()}>Atualizar cotações</button>
          <button type="button" onClick={() => void clearPortfolio()} disabled={selectionBusy || (count === 0 && !session.portfolio_ready)}>Limpar carteira</button>
        </div>
        {loadedSelection === selectionKey && <p className="pf-actions-note">Após carregar, mudanças nos parâmetros recalculam a análise; trocar os ativos exige carregar novamente.</p>}
      </div>
    </Section>
    {count < 2 && <EmptyState icon="portfolio" title={count === 0 ? "Monte seu portfólio" : "Adicione mais um ativo"}>Selecione pelo menos duas ações para comparar risco, retorno e diversificação.</EmptyState>}
    {staleResult && <Notice>{validInputs && manualValid ? "Parâmetros alterados. A análise será atualizada em instantes." : "Parâmetros fora dos limites. O resultado anterior fica oculto até que sejam corrigidos."}</Notice>}
    {busy && <ElapsedLoading>Baixando cotações e calculando alocação, risco e retorno…</ElapsedLoading>}
    {activeResult && <div className="pf-results">
      {activeResult.warnings.map((warning, index) => <Notice tone="warning" key={index}>{warning}</Notice>)}
      <ResultSummary result={activeResult} />
      <div className="pf-overview">
        <Section title="Resumo de Desempenho">
          <div className="pf-kpis">
            <Metrics items={core.map(([key, label, unit]) => ({ label, value: activeResult.metrics[key], unit, status: activeResult.metric_signal[key], help: `${activeResult.metric_status[key]}. ${activeResult.metric_help[key]}`, trend: key === "total_return" ? valueTrend(activeResult) : undefined }))} />
          </div>
          {activeResult.benchmark_available && <Metrics items={extra.map(([key, label, unit]) => ({ label, value: activeResult.metrics[key], unit, help: activeResult.metric_help[key] }))} />}
          {!activeResult.benchmark_available && <Notice>Métricas relativas ao IBOVESPA e análises complementares ficam indisponíveis. Atualize as cotações quando o benchmark responder.</Notice>}
        </Section>
        <Section title="Pesos do Portfólio (%)">
          <WeightList weights={activeResult.weights} />
          <ExportButton path={`/api/portfolio/${activeResult.id}/allocation.csv`}>Exportar alocação (CSV)</ExportButton>
        </Section>
      </div>
      <div className="pf-evidence">
      <Section title="Sugestão de Compra de Cotas">
        <p className="pf-section-lead">Estimativas usam o último fechamento bruto disponível por ativo (pode haver atraso); confirme o preço de execução. Retornos seguem usando preços ajustados. Compra por arredondamento para baixo, sem completar posições com o caixa.</p>
        <Table columns={[{ key: "ticker", label: "Ativo" }, { key: "price", label: "Preço Unitário", unit: "money" }, { key: "quote_date", label: "Data da Cotação" }, { key: "shares", label: "Cotas a Comprar", unit: "integer" }, { key: "effective_value", label: "Valor Efetivo", unit: "money" }, { key: "suggested_value", label: "Valor Sugerido", unit: "money" }, { key: "suggested_weight", label: "Peso Sugerido (%)", unit: "fraction" }, { key: "effective_weight", label: "Peso Efetivo (%)", unit: "fraction" }]} rows={activeResult.trade_suggestions} rowKey="ticker" />
        {activeResult.trade_totals && <Metrics items={[{ label: "Total Alocado Efetivo", value: activeResult.trade_totals.allocated, unit: "money" }, { label: "Saldo Restante (Caixa)", value: activeResult.trade_totals.cash, unit: "money" }, { label: "Eficiência da Alocação", value: activeResult.trade_totals.efficiency, unit: "fraction" }]} />}
      </Section>
      {activeResult.charts.frontier && <Section title="Gráfico da Fronteira Eficiente"><Chart figure={activeResult.charts.frontier} /></Section>}
      <Section title="Evolução do Valor do Portfólio">
        {activeResult.comparison_sample && <p className="pf-note">Comparações com o IBOVESPA usam {activeResult.comparison_sample.count} pregões comuns ({period(activeResult.comparison_sample)}).</p>}
        <Chart figure={activeResult.charts.value} />
        <p className="pf-note">Retornos históricos hipotéticos, pesos fixos rebalanceados diariamente; taxas e impostos excluídos. {activeResult.strategy !== "manual" ? "Markowitz/HRP são avaliados na mesma amostra usada para otimizar os pesos (in-sample), não é validação fora da amostra." : "A alocação manual descreve a amostra histórica, não uma previsão."}</p>
      </Section>
      <Section title="Tabela de Retornos Mensais do Portfólio">
        <MonthlyReturns rows={activeResult.monthly_returns} />
      </Section>
      {activeResult.health && <Section title="Diagnóstico heurístico da carteira">
        <Metrics items={[{ label: "Indicador heurístico", value: activeResult.health.score == null ? null : `${activeResult.health.score}/100` }, { label: "Diagnóstico", value: activeResult.health.label }, { label: "Cobertura", value: activeResult.health.coverage, unit: "percent" }]} />
        {activeResult.health.diagnostics.map((row, index) => <Notice key={index} tone={row.status}>{row.text}</Notice>)}
      </Section>}
      {activeResult.stress && <Section title="Stress Test — Crises Históricas">
        <p className="pf-section-lead">Desempenho hipotético com pesos atuais rebalanceados diariamente, sem taxas ou impostos. Por padrão usa o lookback selecionado; histórico longo é opcional.</p>
        <label className="pf-stress-check"><input type="checkbox" checked={longHistory} disabled={stressBusy} onChange={event => void loadStress(event.target.checked)} />Carregar histórico longo para crises (desde 2008)</label>
        {stressBusy ? <Loading>Carregando histórico de crises…</Loading> : stressResult && <>
          {stressResult.warnings.map((warning, index) => <Notice tone="warning" key={index}>{warning}</Notice>)}
          {stressResult.rows.length === 0 && <Notice>Não há pelo menos 5 pregões completos nas crises para todos os ativos selecionados. Amplie o histórico acima para incluir crises mais antigas.</Notice>}
          {stressResult.rows.length > 0 && <Table columns={[{ key: "crisis", label: "Crise", unit: "text" }, { key: "period", label: "Período", unit: "text" }, { key: "portfolio", label: "Portfólio (%)", unit: "fraction" }, { key: "benchmark", label: "IBOVESPA (%)", unit: "fraction" }, { key: "difference_pp", label: "vs IBOV (p.p.)", unit: "number" }]} rows={stressResult.rows} rowKey="crisis" />}
          {stressResult.chart && <Chart figure={stressResult.chart} />}
        </>}
      </Section>}
      {activeResult.regime && <Section title="Regime de Mercado (Últimos 21 Dias)">
        <Metrics items={[{ label: "Regime", value: activeResult.regime.label }, { label: "Portfólio (21d)", value: activeResult.regime.portfolio, unit: "percent" }, { label: "vs IBOVESPA (21d, pp)", value: activeResult.regime.difference_pp, unit: "number" }, { label: "Volatilidade (21d, a.a.)", value: activeResult.regime.volatility, unit: "percent" }]} />
        <Notice tone="warning">Sugestão heurística: {activeResult.regime.advice}</Notice>
        <p className="pf-note">Regra simples sobre os últimos 21 pregões, sem previsão e sem recomendação de investimento.</p>
      </Section>}
      {activeResult.concentration && <Section title="Concentração">
        <Metrics items={[{ label: "HHI Concentração", value: activeResult.concentration.hhi, unit: "integer", help: activeResult.concentration.level }, { label: "Ativos Efetivos", value: `${activeResult.concentration.effective_assets}/${activeResult.concentration.selected_assets}` }, { label: "Maior Peso", value: activeResult.concentration.largest_weight, unit: "percent", status: activeResult.concentration.largest_signal, help: activeResult.concentration.largest_status }]} />
        <Notice tone={activeResult.concentration.diagnostic_status}>{activeResult.concentration.diagnosis}</Notice>
      </Section>}
      {activeResult.charts.drawdown && <Section title="Análise de Drawdown"><Chart figure={activeResult.charts.drawdown} /><h3 className="pf-sub">Máximo Drawdown por Ativo Individual</h3><Table columns={[{ key: "ticker", label: "Ativo" }, { key: "max_drawdown", label: "Máximo Drawdown (%)", unit: "percent" }, { key: "date", label: "Data do Máximo Drawdown" }]} rows={activeResult.asset_drawdowns} rowKey="ticker" /></Section>}
      {activeResult.charts.rolling_beta && <Section title="Beta Móvel (60 dias) vs IBOVESPA"><Chart figure={activeResult.charts.rolling_beta} /></Section>}
      {activeResult.charts.rolling_sharpe && <Section title="Índice de Sharpe Móvel Anualizado (60 dias)"><Chart figure={activeResult.charts.rolling_sharpe} /></Section>}
      </div>
      <p className="discreet-status">Fontes: {activeResult.source.adjusted_prices}; {activeResult.source.trade_quotes}; Selic: {activeResult.source.selic_source} ({(activeResult.source.selic_annual * 100).toLocaleString("pt-BR", { maximumFractionDigits: 2 })}% a.a.); taxa livre de risco histórica: {activeResult.source.historical_rf}.</p>
      <div className="next-step-card"><h3>Portfólio configurado</h3><p>Projete trajetórias com Monte Carlo.</p><Link className="button-link" href="/simulacao">Abrir Simulação</Link></div>
    </div>}
  </>;
}
