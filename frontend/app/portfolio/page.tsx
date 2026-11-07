"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { useApp } from "../../components/AppProvider";
import Chart from "../../components/Chart";
import { ExportButton, Idle, Loading, Metrics, Notice, Pane, Table } from "../../components/ui";
import { tickerColor } from "../../components/Shell";
import { ElapsedLoading, MonthlyReturns, runFacts, WeightList } from "../../components/PortfolioParts";
import { api } from "../../lib/api";
import { handoffTickers, period, strategies } from "../../lib/portfolio";
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

  if (loading || !session) return <div className="desk">{appError ? <Notice tone="negative">{appError}</Notice> : <Loading>Carregando lista de ações e seleção…</Loading>}</div>;
  const activeResult = resultKey === fingerprint ? result : null;
  const staleResult = !!result && resultKey !== fingerprint && !busy;
  const count = session.selection.length;
  const companies = new Map(stocks.map(stock => [stock.ticker, stock.company]));
  const core = [["total_return", "Retorno Total", "percent"], ["volatility", "Volatilidade Anual", "percent"], ["sharpe", "Índice Sharpe", "ratio"], ["sortino", "Índice Sortino", "ratio"], ["max_drawdown", "Queda máxima (Max Drawdown)", "percent"]] as const;
  const extra = [["annual_return", "Retorno Anualizado", "percent"], ["beta", "Beta vs IBOV", "ratio"], ["annual_alpha", "Alpha Anual", "percent"], ["daily_var", "VaR Diário 95% (paramétrico normal)", "percent"], ["historical_var_95", "VaR Histórico 95% (diário)", "percent"], ["cvar_95", "CVaR 95% (diário)", "percent"], ["information_ratio", "Information Ratio (anualizado)", "ratio"]] as const;
  const sections: [string, string][] = activeResult ? [["pf-resumo", "Resumo"], ["pf-pesos", "Pesos"], ["pf-valor", "Valor"], ["pf-compra", "Compra"], ...(activeResult.charts.frontier ? [["pf-fronteira", "Fronteira"]] as [string, string][] : []), ...(activeResult.charts.drawdown ? [["pf-drawdown", "Drawdown"]] as [string, string][] : []), ["pf-mensal", "Mensal"], ...(activeResult.stress ? [["pf-stress", "Stress"]] as [string, string][] : []), ["pf-risco", "Risco"], ...(activeResult.charts.rolling_beta || activeResult.charts.rolling_sharpe ? [["pf-moveis", "Móveis"]] as [string, string][] : [])] : [];
  const idle = (text: ReactNode) => <Idle>{text}</Idle>;
  const waiting = count < 2 ? `Selecione ao menos 2 ativos (${count}/2) pela busca, Ctrl+K.` : busy ? "Calculando…" : "Pressione Carregar portfólio para calcular pesos, risco e retorno.";

  return <div className="desk desk-split">
    <Pane title="Parâmetros" className="pane-params" meta={`${count}/20 ativos`}>
      <div className="param-group">
        <h3 className="pane-sub">Ativos da carteira</h3>
        {count ? <ol className="asset-list">{session.selection.map((ticker, index) => <li key={ticker}>
          <i className="swatch" style={{ background: tickerColor(index) }} aria-hidden="true" /><b>{ticker}</b><span>{companies.get(ticker) ?? "Fora da lista B3"}</span>
          <button type="button" className="icon-btn" disabled={selectionBusy} aria-label={`Remover ${ticker}`} onClick={() => void changeSelection(session.selection.filter(item => item !== ticker))}>×</button>
        </li>)}</ol> : <p className="pane-lead">Nenhum ativo selecionado.</p>}
        <button type="button" className="ghost-add" onClick={() => window.dispatchEvent(new Event("b3lab:focus-search"))}>+ Adicionar ativo <kbd>Ctrl K</kbd></button>
      </div>
      <div className="param-group param-fields">
        <h3 className="pane-sub">Janela e capital</h3>
        <label>Período de análise<select disabled={busy} value={windowChoice} onChange={event => { const value = event.target.value; setWindowChoice(value); setInputs(previous => ({ ...previous, lookback_days: value === "custom" ? customDays : Number(value) })); }}>
          <option value="730">2 Anos (Padrão)</option><option value="365">1 Ano</option><option value="1095">3 Anos</option><option value="1825">5 Anos</option><option value="180">6 Meses</option><option value="custom">Personalizado (Dias)</option>
        </select></label>
        {windowChoice === "custom" && <label>Dias de lookback<input type="number" disabled={busy} min={60} max={5000} step={10} value={customDays} onChange={event => { const days = Number(event.target.value); setCustomDays(days); setInputs(previous => ({ ...previous, lookback_days: days })); }} /></label>}
        <label>Capital disponível (R$)<input type="number" disabled={busy} min={100} max={1000000} step={100} value={inputs.capital} onChange={event => setInputs(previous => ({ ...previous, capital: Number(event.target.value) }))} /></label>
        <p className="param-hint">Dados a partir de {new Date(Date.now() - inputs.lookback_days * 86400000).toLocaleDateString("pt-BR")} · dias corridos.</p>
        <h3 className="pane-sub">Estratégia</h3>
        <label>Estratégia de alocação<select disabled={busy} value={inputs.strategy} onChange={event => setInputs(previous => ({ ...previous, strategy: event.target.value as Strategy }))}>{strategies.map(strategy => <option key={strategy.value} value={strategy.value}>{strategy.label}</option>)}</select></label>
        {inputs.strategy === "manual" && <>
          <p className="param-hint">Pesos pelo valor de mercado das cotas informadas; o capital só entra no histórico de valor.</p>
          <div className="param-pairs">{session.selection.map(ticker => <label key={ticker}>Cotas de {ticker}<input type="number" disabled={busy} min={0} step={1} value={inputs.manual_shares[ticker] ?? 0} onChange={event => setInputs(previous => ({ ...previous, manual_shares: { ...previous.manual_shares, [ticker]: Number(event.target.value) } }))} /></label>)}</div>
        </>}
        {inputs.strategy === "markowitz" && <>
          <div className="segmented" role="group" aria-label="Diversificação">{[{ label: "Concentrada", gamma: 0 }, { label: "Equilibrada", gamma: 1 }, { label: "Diversificada", gamma: 2 }].map(preset => <button type="button" key={preset.label} disabled={busy} aria-pressed={inputs.gamma === preset.gamma} onClick={() => setInputs(previous => ({ ...previous, gamma: preset.gamma }))}>{preset.label}</button>)}</div>
          <label>Diversificação (regularização L2, 0 a 3)<input type="number" disabled={busy} min={0} max={3} step={.1} value={inputs.gamma} onChange={event => setInputs(previous => ({ ...previous, gamma: Number(event.target.value) }))} /></label>
        </>}
      </div>
      {!manualValid && <Notice tone="warning">A alocação manual precisa ter ao menos uma cota positiva.</Notice>}
      {!validInputs && <Notice tone="warning">Confira os limites de capital, histórico, cotas inteiras e diversificação.</Notice>}
      <div className="param-run">
        <button type="button" className="primary" disabled={busy || count < 2 || !manualValid || !validInputs || selectionBusy} onClick={() => void runAnalysis()}>{busy ? "Calculando…" : "Carregar portfólio"}</button>
        <button type="button" disabled={selectionBusy} title="Limpa o cache de preços desta sessão e baixa as cotações de novo" onClick={() => void refreshQuotes()}>Rebaixar preços</button>
      </div>
      <p className="param-hint">{loadedSelection === selectionKey ? "Parâmetros recalculam sozinhos após meio segundo sem edições. Trocar ativos exige Carregar portfólio." : "Trocar ativos ou parâmetros exige Carregar portfólio uma vez; depois, parâmetros recalculam sozinhos."}</p>
      {sections.length > 0 && <nav className="jump" aria-label="Seções do resultado"><h3 className="pane-sub">Ir para</h3><ul>{sections.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}</ul></nav>}
    </Pane>
    <div className="board">
      {(appError || notice || error || staleResult) && <div className="span-12 board-notes">
        {appError && <Notice tone="negative">{appError}</Notice>}
        {notice && <Notice>{notice}</Notice>}
        {error && <Notice tone="negative">{error}</Notice>}
        {staleResult && <Notice>{validInputs && manualValid ? "Parâmetros alterados. A análise será atualizada em instantes." : "Parâmetros fora dos limites. O resultado anterior fica oculto até que sejam corrigidos."}</Notice>}
      </div>}
      {!activeResult && <>
        <Pane title="Resumo de desempenho" className="span-12" meta={busy ? undefined : "aguardando cálculo"}>
          {busy ? <ElapsedLoading>Baixando cotações e calculando alocação, risco e retorno…</ElapsedLoading> : idle(waiting)}
          <Metrics size="lg" items={core.map(([, label, unit]) => ({ label, value: null, unit }))} />
        </Pane>
        <Pane title="Pesos do portfólio" className="span-4">{idle("Pesos por ativo e barra de composição.")}</Pane>
        <Pane title="Evolução do valor" className="span-8">{idle("Valor histórico da carteira contra o IBOVESPA, rebalanceado diariamente.")}</Pane>
        <Pane title="Sugestão de compra" className="span-8">{idle("Cotas a comprar com o capital informado, pelo último fechamento bruto.")}</Pane>
        <Pane title="Fronteira eficiente" className="span-4">{idle("Disponível nas estratégias otimizadas.")}</Pane>

      </>}
      {activeResult && <>
        {activeResult.warnings.length > 0 && <div className="span-12 board-notes">{activeResult.warnings.map((warning, index) => <Notice tone="warning" key={index}>{warning}</Notice>)}</div>}
        <Pane id="pf-resumo" title="Resumo de desempenho" className="span-12" meta={runFacts(activeResult).join(" · ")}>
          <Metrics size="lg" items={core.map(([key, label, unit]) => ({ label, value: activeResult.metrics[key], unit, status: activeResult.metric_signal[key], help: `${activeResult.metric_status[key]}. ${activeResult.metric_help[key]}`, trend: key === "total_return" ? valueTrend(activeResult) : undefined }))} />
          {activeResult.benchmark_available ? <Metrics items={extra.map(([key, label, unit]) => ({ label, value: activeResult.metrics[key], unit, help: activeResult.metric_help[key] }))} />
            : <Notice>Comparações que dependem do IBOVESPA ficam indisponíveis; as métricas próprias do portfólio continuam válidas. Rebaixe os preços quando o benchmark responder.</Notice>}
          <p className="pane-foot">{activeResult.saved ? "Carteira salva para esta sessão anônima; não é uma conta." : "A carteira não foi salva para esta sessão anônima."}{activeResult.sample.zero_excluded ? " Ativos com peso zero foram excluídos da amostra." : ""}</p>
        </Pane>
        <Pane id="pf-pesos" title="Pesos do portfólio" className="span-4" meta={activeResult.mode} actions={<ExportButton path={`/api/portfolio/${activeResult.id}/allocation.csv`}>CSV</ExportButton>}>
          <WeightList weights={activeResult.weights} selection={session.selection} />
        </Pane>
        <Pane id="pf-valor" title="Evolução do valor" className="span-8" meta={activeResult.comparison_sample ? `vs IBOV em ${activeResult.comparison_sample.count} pregões comuns (${period(activeResult.comparison_sample)})` : undefined}>
          <Chart figure={activeResult.charts.value} height={300} />
          <p className="pane-foot">Retornos hipotéticos, pesos fixos rebalanceados diariamente; sem taxas e impostos. {activeResult.strategy !== "manual" ? "Pesos otimizados e avaliados na mesma amostra (in-sample)." : "A alocação manual descreve a amostra histórica, não uma previsão."}</p>
        </Pane>
        <Pane id="pf-compra" title="Sugestão de compra de cotas" className={activeResult.charts.frontier ? "span-8" : "span-12"} actions={<ExportButton path={`/api/portfolio/${activeResult.id}/trades.csv`}>CSV</ExportButton>} flush>
          <Table caption="Sugestão de compra" swatch={row => tickerColor(session.selection.indexOf((row as { ticker: string }).ticker))} columns={[{ key: "ticker", label: "Ativo" }, { key: "price", label: "Preço unit.", unit: "money" }, { key: "quote_date", label: "Data cotação" }, { key: "shares", label: "Cotas", unit: "integer" }, { key: "effective_value", label: "Valor efetivo", unit: "money" }, { key: "suggested_value", label: "Valor sugerido", unit: "money" }, { key: "suggested_weight", label: "Peso sugerido", unit: "fraction" }, { key: "effective_weight", label: "Peso efetivo", unit: "fraction" }]} rows={activeResult.trade_suggestions} />
          {activeResult.trade_totals && <Metrics items={[{ label: "Total alocado efetivo", value: activeResult.trade_totals.allocated, unit: "money" }, { label: "Saldo restante (caixa)", value: activeResult.trade_totals.cash, unit: "money" }, { label: "Eficiência da alocação", value: activeResult.trade_totals.efficiency, unit: "fraction" }]} />}
          <p className="pane-foot">Último fechamento bruto por ativo (pode haver atraso); confirme o preço de execução. Retornos usam preços ajustados. Arredondamento para baixo, sem completar com o caixa.</p>
        </Pane>
        {activeResult.charts.frontier && <Pane id="pf-fronteira" title="Fronteira eficiente" className="span-4"><Chart figure={activeResult.charts.frontier} height={300} /></Pane>}
        {activeResult.charts.drawdown && <Pane id="pf-drawdown" title="Drawdown" className="span-8"><Chart figure={activeResult.charts.drawdown} height={260} /></Pane>}
        {activeResult.charts.drawdown && <Pane title="Máx. drawdown por ativo" className="span-4" flush>
          <Table caption="Máximo drawdown por ativo" swatch={row => tickerColor(session.selection.indexOf((row as { ticker: string }).ticker))} columns={[{ key: "ticker", label: "Ativo" }, { key: "max_drawdown", label: "Máx. DD", unit: "percent" }, { key: "date", label: "Data" }]} rows={activeResult.asset_drawdowns} />
        </Pane>}
        <Pane id="pf-mensal" title="Retornos mensais" className="span-12" meta="% ao mês · + ganho · − perda · intensidade acompanha o movimento" flush>
          <MonthlyReturns rows={activeResult.monthly_returns} />
        </Pane>
        {activeResult.stress && <Pane id="pf-stress" title="Stress test · crises históricas" className="span-12" actions={<label className="inline-check"><input type="checkbox" checked={longHistory} disabled={stressBusy} onChange={event => void loadStress(event.target.checked)} />Histórico longo (desde 2008)</label>}>
          {stressBusy ? <Loading>Carregando histórico de crises…</Loading> : stressResult && <div className={stressResult.chart ? "split" : "solo"}>
            <div>
              {stressResult.warnings.map((warning, index) => <Notice tone="warning" key={index}>{warning}</Notice>)}
              {stressResult.rows.length === 0 && <Notice>Não há pelo menos 5 pregões completos nas crises para todos os ativos. Ative o histórico longo para incluir crises mais antigas.</Notice>}
              {stressResult.rows.length > 0 && <Table caption="Stress test" rowKey="crisis" columns={[{ key: "crisis", label: "Crise", unit: "text" }, { key: "period", label: "Período", unit: "text" }, { key: "portfolio", label: "Portfólio", unit: "fraction" }, { key: "benchmark", label: "IBOV", unit: "fraction" }, { key: "difference_pp", label: "vs IBOV p.p.", unit: "number", signed: true }]} rows={stressResult.rows} />}
              <p className="pane-foot">Pesos atuais rebalanceados diariamente, sem taxas ou impostos. Usa o lookback selecionado; histórico longo é opcional.</p>
            </div>
            {stressResult.chart && <Chart figure={stressResult.chart} height={260} />}
          </div>}
        </Pane>}
        {activeResult.regime && <Pane id="pf-risco" title="Regime · 21 pregões" className="span-4">
          <Metrics items={[{ label: "Regime", value: activeResult.regime.label }, { label: "Portfólio 21d", value: activeResult.regime.portfolio, unit: "percent" }, { label: "vs IBOV 21d (p.p.)", value: activeResult.regime.difference_pp, unit: "number" }, { label: "Volatilidade 21d a.a.", value: activeResult.regime.volatility, unit: "percent" }]} />
          <Notice tone="warning">Sugestão heurística: {activeResult.regime.advice}</Notice>
          <p className="pane-foot">Regra simples sobre os últimos 21 pregões, sem previsão e sem recomendação.</p>
        </Pane>}
        {activeResult.concentration && <Pane id={activeResult.regime ? undefined : "pf-risco"} title="Concentração" className="span-4">
          <Metrics items={[{ label: "HHI", value: activeResult.concentration.hhi, unit: "integer", help: activeResult.concentration.level }, { label: "Ativos efetivos", value: `${activeResult.concentration.effective_assets}/${activeResult.concentration.selected_assets}` }, { label: "Maior peso", value: activeResult.concentration.largest_weight, unit: "percent", status: activeResult.concentration.largest_signal, help: activeResult.concentration.largest_status }]} />
          <Notice tone={activeResult.concentration.diagnostic_status}>{activeResult.concentration.diagnosis}</Notice>
        </Pane>}
        {activeResult.health && <Pane title="Diagnóstico heurístico" className="span-4">
          <Metrics items={[{ label: "Indicador heurístico", value: activeResult.health.score == null ? null : `${activeResult.health.score}/100` }, { label: "Diagnóstico", value: activeResult.health.label }, { label: "Cobertura", value: activeResult.health.coverage, unit: "percent" }]} />
          {activeResult.health.diagnostics.map((row, index) => <Notice key={index} tone={row.status}>{row.text}</Notice>)}
        </Pane>}
        {activeResult.charts.rolling_beta && <Pane id="pf-moveis" title="Beta móvel 60d vs IBOV" className="span-6"><Chart figure={activeResult.charts.rolling_beta} height={240} /></Pane>}
        {activeResult.charts.rolling_sharpe && <Pane id={activeResult.charts.rolling_beta ? undefined : "pf-moveis"} title="Sharpe móvel 60d anualizado" className="span-6"><Chart figure={activeResult.charts.rolling_sharpe} height={240} /></Pane>}
        <div className="span-12 board-foot">
          <p>Fontes: {activeResult.source.adjusted_prices}; {activeResult.source.trade_quotes}; Selic: {activeResult.source.selic_source} ({(activeResult.source.selic_annual * 100).toLocaleString("pt-BR", { maximumFractionDigits: 2 })}% a.a.); taxa livre de risco histórica: {activeResult.source.historical_rf}.</p>
          <p>Próximo <Link href="/simulacao"><kbd>F3</kbd> Simulação</Link> projeta trajetórias Monte Carlo com estes pesos · <Link href="/noticias"><kbd>F4</kbd> Notícias</Link> lê o tom das notícias da carteira.</p>
        </div>
      </>}
    </div>
  </div>;
}
