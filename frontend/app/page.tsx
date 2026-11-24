"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import "./home.css";
import { MAX_SELECTION, SELECTION_FULL_MESSAGE, useApp } from "../components/AppProvider";
import type { SelectionUpdate } from "../components/AppProvider";
import HomeResults from "../components/HomeResults";
import StockPicker from "../components/StockPicker";
import { Headlines, Macro, Monitor, SectorMap } from "./HomeBoard";
import { Idle, Loading, Notice, Pane } from "../components/ui";
import { analyzeHome, currentAnalysis, refreshConsensus, refreshFundamentus, suggestions } from "../lib/home";
import type { HomeResult } from "../lib/home";

type AnalysisAction = "analyze" | "consensus" | "source";
type QuickAnalysis = "stay" | "scroll";

export default function HomePage() {
  const { session, stocks, loading, error: appError, setSelection, toggleFavorite, refreshSession } = useApp();
  const [result, setResult] = useState<HomeResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [mapNote, setMapNote] = useState<{ text: string; failed?: boolean } | null>(null);
  const [retryAction, setRetryAction] = useState<AnalysisAction>("analyze");
  const [sectorUndo, setSectorUndo] = useState<{ tickers: string[]; revision: number } | null>(null);
  const [quickAnalysis, setQuickAnalysis] = useState<{ revision: number; mode: QuickAnalysis } | null>(null);
  const generation = useRef(0);
  const selectionPending = useRef(false);
  const results = useRef<HTMLDivElement>(null);
  const revision = useRef(session?.revision);
  revision.current = session?.revision;

  useEffect(() => {
    generation.current++;
    setBusy(false);
    setAnalysisError(null);
    return () => { generation.current++; };
  }, [session?.revision]);

  useEffect(() => {
    if (!session || !quickAnalysis || session.revision < quickAnalysis.revision) return;
    setQuickAnalysis(null);
    if (session.revision === quickAnalysis.revision) void run("analyze", quickAnalysis.mode === "scroll");
  }, [session, quickAnalysis]);

  useEffect(() => {
    if (!session) return;
    let active = true;
    const request = generation.current;
    queueMicrotask(() => {
      if (!active) return;
      const url = new URL(window.location.href);
      const handoff = url.searchParams.get("analisar");
      const unavailable = url.searchParams.get("unavailable");
      if (handoff !== null || unavailable !== null) {
        url.searchParams.delete("analisar");
        url.searchParams.delete("unavailable");
        window.history.replaceState(null, "", url);
      }
      if (unavailable) setNotice(`${unavailable} está fora da lista local B3. Escolha outra ação para analisar.`);
      if (handoff && handoff === session.selection.join(",")) {
        void run("analyze", true);
        return;
      }
      if (quickAnalysis?.revision === session.revision) return;
      currentAnalysis().then(saved => {
        if (active && saved && request === generation.current && saved.revision === revision.current) setResult(saved);
      }).catch(() => {});
    });
    return () => { active = false; };
  }, [session?.revision]);

  async function editSelection(update: SelectionUpdate, analyze?: QuickAnalysis) {
    if (selectionPending.current) return;
    selectionPending.current = true;
    setSelecting(true);
    setSelectionError(null);
    try {
      const updated = await setSelection(update);
      setSectorUndo(null);
      setMapNote(null);
      if (analyze) setQuickAnalysis({ revision: updated.revision, mode: analyze });
    } catch (cause) {
      setSelectionError((cause as Error).message);
    } finally {
      selectionPending.current = false;
      setSelecting(false);
    }
  }

  function addStock(ticker: string, analyze?: QuickAnalysis) {
    void editSelection(current => {
      if (current.includes(ticker)) return [...current];
      if (current.length >= MAX_SELECTION) throw new Error(SELECTION_FULL_MESSAGE);
      return [...current, ticker];
    }, analyze);
  }

  async function selectSector(sector: string) {
    if (selectionPending.current) return;
    selectionPending.current = true;
    setSelecting(true);
    setMapNote(null);
    setSelectionError(null);
    try {
      const options = await suggestions([sector]);
      if (!options.autopick.length) { setMapNote({ text: "Nenhuma ação disponível neste setor." }); return; }
      let before: string[] = [];
      const updated = await setSelection(current => {
        before = [...current];
        return options.autopick;
      });
      setSectorUndo(before.join(",") === updated.selection.join(",") ? null : { tickers: before, revision: updated.revision });
      setMapNote({ text: `${updated.selection.length} ações mais líquidas de ${sector}, entre ${options.count} disponíveis.` });
    } catch (cause) {
      setMapNote({ text: (cause as Error).message, failed: true });
    } finally {
      selectionPending.current = false;
      setSelecting(false);
    }
  }

  async function run(action: AnalysisAction = "analyze", showResult = false) {
    if (!session || selectionPending.current) return;
    const request = ++generation.current;
    const rev = session.revision;
    const tickers = session.selection;
    const isCurrent = () => request === generation.current && rev === revision.current;
    setRetryAction(action);
    setBusy(true);
    setAnalysisError(null);
    if (showResult) results.current?.scrollIntoView({ block: "nearest" });
    let refreshingSource = action === "source";
    if (refreshingSource) setMapNote(null);
    try {
      if (action === "source") {
        await refreshFundamentus();
        if (!isCurrent()) return;
        setMapNote({ text: "Dados da Fundamentus atualizados." });
        refreshingSource = false;
        if (result?.revision !== rev) return;
      }
      if (action === "consensus") {
        const consensus = await refreshConsensus(tickers, rev);
        if (isCurrent()) setResult(current => current ? { ...current, consensus } : current);
      } else if (tickers.length) {
        const value = await analyzeHome(tickers, rev);
        if (isCurrent()) setResult(value);
      }
    } catch (cause) {
      if (isCurrent()) {
        const text = (cause as Error).message;
        if (refreshingSource) setMapNote({ text, failed: true });
        else setAnalysisError(text);
      }
    } finally {
      if (isCurrent()) setBusy(false);
    }
  }

  async function favorite(ticker: string) {
    try { await toggleFavorite(ticker); }
    catch (cause) { setSelectionError((cause as Error).message); }
  }

  if (loading || !session) return <div className="desk">{appError ? <Notice tone="negative">{appError}<button onClick={() => void refreshSession()}>Tentar novamente</button></Notice> : <Loading>Carregando sessão e lista de ações da B3…</Loading>}</div>;
  const selection = session.selection;
  const ready = result?.revision === session.revision && result.requested_tickers.join(",") === selection.join(",") ? result : null;
  const disabled = busy || selecting;
  const analyzeLabel = busy ? "Analisando…" : ready ? "Atualizar análise" : selection.length === 1 ? `Analisar ${selection[0]}` : `Comparar ${selection.length} ativos`;
  const undo = sectorUndo?.revision === session.revision ? sectorUndo : null;

  return <div className="desk desk-home">
    <div className="home-main">
    <Monitor selection={selection} stocks={stocks} watchlist={session.watchlist} ready={ready} busy={disabled}
      picker={<>
        <StockPicker selection={selection} action={selection.length ? "Adicionar" : "Analisar"} disabled={disabled || selection.length >= MAX_SELECTION} onChoose={ticker => addStock(ticker, selection.length === 0 ? "stay" : undefined)} />
        {!!selection.length && <button type="button" className={ready ? undefined : "primary"} disabled={disabled} aria-busy={busy} onClick={() => void run("analyze", true)}>{analyzeLabel}</button>}
      </>}
      onQuick={ticker => addStock(ticker, "scroll")} onFavorite={ticker => void favorite(ticker)}
      onAdd={ticker => addStock(ticker, selection.length === 0 ? "scroll" : undefined)}
      onRemove={ticker => void editSelection(current => current.filter(item => item !== ticker))}
      notices={<>
        {notice && <Notice>{notice}</Notice>}
        {selectionError && <Notice tone="negative">{selectionError}</Notice>}
        {selecting && <p className="picker-status" role="status">Atualizando a seleção…</p>}
        {session.restored && !!session.restoration?.invalid_count && <Notice tone="warning">{session.restoration.invalid_count} ações da carteira salva estão fora da lista B3 e foram removidas.</Notice>}
        {session.restored && !!session.restoration?.truncated_count && <Notice>Das {session.restoration.original_count} ações salvas, foram carregadas as 20 primeiras válidas.</Notice>}
      </>}
    />
    {(selection.length > 0 || busy || analysisError) && <div className="home-result" id="resultado-analise" ref={results}>
      {analysisError && <Notice tone="negative">{analysisError}<button disabled={disabled} onClick={() => void run(retryAction)}>Tentar novamente</button></Notice>}
      {ready ? <>
        {ready.missing_tickers.length > 0 && <Notice tone="warning">Fundamentus indisponível para {ready.missing_tickers.join(", ")}.</Notice>}
        <HomeResults ready={ready} selection={selection} busy={busy} onRefreshConsensus={() => void run("consensus")} />
        <div className="flow-links">
          <Link className="button-link" href="/portfolio">{selection.length === 1 ? "Montar carteira com esta ação" : "Montar carteira com estas ações"} <span aria-hidden="true">→</span></Link>
          <Link className="button-link" href="/screener">Encontrar mais ações <span aria-hidden="true">→</span></Link>
        </div>
      </> : <Pane title="Resultado da análise" className="pane-idle">
        {busy ? <Loading>Buscando indicadores, comparação setorial e preços-alvo para {selection.join(", ")}…</Loading>
          : !analysisError && <Idle>{result ? "A seleção mudou. Faça uma nova análise para ver os indicadores atuais." : "Consulte os indicadores e os preços-alvo da seleção acima."}</Idle>}
      </Pane>}
    </div>}
    <SectorMap stocks={stocks} selection={selection} busy={disabled} onSector={sector => void selectSector(sector)}
      onAdd={ticker => addStock(ticker, selection.length === 0 ? "scroll" : undefined)} onRefresh={() => void run("source")}
      note={mapNote && <Notice tone={mapNote.failed ? "negative" : "info"}>{mapNote.text} {undo && <button type="button" disabled={disabled} onClick={() => void editSelection(undo.tickers)}>Desfazer</button>}</Notice>} />
    </div>
    <div className="home-rail">
      <Macro />
      <Headlines />
    </div>
  </div>;
}
