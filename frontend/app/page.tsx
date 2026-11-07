"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import "./home.css";
import { MAX_SELECTION, useApp } from "../components/AppProvider";
import type { SelectionUpdate } from "../components/AppProvider";
import HomeResults from "../components/HomeResults";
import { Headlines, Macro, Monitor, SectorMap } from "./HomeBoard";
import { Idle, Loading, Notice, Pane } from "../components/ui";
import { analyzeHome, currentAnalysis, refreshConsensus, refreshFundamentus, suggestions } from "../lib/home";
import type { HomeResult, Suggestions } from "../lib/home";

export default function HomePage() {
  const {session,stocks,loading,error:appError,setSelection,toggleFavorite,refreshSession} = useApp();
  const [sectors,setSectors] = useState<string[]>([]);
  const [suggested,setSuggested] = useState<Suggestions|null>(null);
  const [result,setResult] = useState<HomeResult|null>(null);
  const [busy,setBusy] = useState(false);
  const [sectorBusy,setSectorBusy] = useState(false);
  const [error,setError] = useState<string|null>(null);
  const [notice,setNotice] = useState<string|null>(null);
  const [retryAction,setRetryAction] = useState<"analyze"|"consensus"|"source">("analyze");
  const [stale,setStale] = useState(false);
  const [sectorUndo,setSectorUndo] = useState<string[]|null>(null);
  const pendingAnalysis = useRef<string[]|null>(null);
  const generation = useRef(0);
  const sectorGeneration = useRef(0);
  const analyzedRevision = useRef<number|null>(null);
  const revision = useRef(session?.revision);
  revision.current = session?.revision;
  useEffect(() => {generation.current++;analyzedRevision.current=null;setResult(null);setBusy(false);},[session?.revision]);
  useEffect(() => {
    // Quick-start runs after the new selection is rendered, so the revision reset above does not discard it.
    if(!pendingAnalysis.current || !session || JSON.stringify(session.selection)!==JSON.stringify(pendingAnalysis.current))return;
    pendingAnalysis.current=null;
    void run();
  },[session]);
  useEffect(() => {
    const unavailable=new URLSearchParams(window.location.search).get("unavailable");
    if(unavailable)setNotice(`${unavailable} está fora da lista local B3. A seleção foi removida; escolha outro ativo para analisar.`);
    let active=true;
    const request=++sectorGeneration.current;
    suggestions([]).then(value => {if(active && request===sectorGeneration.current)setSuggested(value);}).catch(exc => {if(active && request===sectorGeneration.current)setError(String(exc.message));});
    return () => {active=false;generation.current++;sectorGeneration.current++;};
  },[]);
  useEffect(() => {
    const request=generation.current;
    let active=true;
    currentAnalysis().then(saved => {
      if(active && saved && request===generation.current && saved.revision===session?.revision && JSON.stringify(saved.requested_tickers)===JSON.stringify(session?.selection ?? []))setResult(saved);
    }).catch(() => {});
    return () => {active=false;};
  },[]);

  async function updateSelection(tickers:SelectionUpdate) {
    if(result)setStale(true);generation.current++;analyzedRevision.current=null;setResult(null);setBusy(false);setError(null);setSectorUndo(null);
    await setSelection(tickers);
  }
  async function changeSelection(tickers:SelectionUpdate) {
    try {await updateSelection(tickers);} catch(exc) {setError(exc instanceof Error ? exc.message : String(exc));}
  }
  async function startQuickAnalysis(ticker:string) {
    generation.current++;analyzedRevision.current=null;setResult(null);setBusy(false);setError(null);setSectorUndo(null);
    pendingAnalysis.current=[ticker];
    try {await setSelection([ticker]);} catch(exc) {pendingAnalysis.current=null;setError((exc as Error).message);}
  }
  async function undoSectorChange() {
    const previous=sectorUndo;
    if(!previous)return;
    setSectorUndo(null);
    await changeSelection(previous);
    setNotice(`Seleção anterior restaurada: ${previous.length} ${previous.length===1 ? "ativo":"ativos"}.`);
  }
  async function changeSectors(next:string[]) {
    const gen=++sectorGeneration.current;
    const previous=[...(session?.selection ?? [])];
    setSectors(next);setSectorBusy(true);setError(null);
    try {
      const value=await suggestions(next);
      if(gen!==sectorGeneration.current)return;
      setSuggested(value);
      if(next.length && !next.includes("Todos")) {
        await changeSelection(value.autopick);
        setNotice(previous.length
          ? `${next.join(", ")}: seleção de ${previous.length} ${previous.length===1 ? "ativo":"ativos"} substituída pelos ${value.autopick.length} mais líquidos de ${value.count}.`
          : `${next.join(", ")}: ${value.autopick.length} ativos mais líquidos de ${value.count} selecionados.`);
        if(previous.length)setSectorUndo(previous);
      }
    } catch(exc) {if(gen===sectorGeneration.current)setError((exc as Error).message);}
    finally {if(gen===sectorGeneration.current)setSectorBusy(false);}
  }
  async function run(action:"analyze"|"consensus"|"source"="analyze") {
    if(!session)return;
    const gen=++generation.current;
    const rev=session.revision;
    const tickers=[...session.selection];
    setRetryAction(action);setBusy(true);setError(null);setStale(false);
    if(action==="analyze")analyzedRevision.current=rev;
    if(action!=="consensus")setResult(null);
    try {
      if(action==="source") {
        const value=await refreshFundamentus();
        if(gen!==generation.current || rev!==revision.current)return;
        setNotice(`Cache Fundamentus atualizado (${value.removed} entradas removidas). ${value.scope}.`);
        const options=await suggestions(sectors);
        if(gen!==generation.current || rev!==revision.current)return;
        setSuggested(options);
        if(analyzedRevision.current!==rev)return;
      }
      if(action==="consensus") {
        const value=await refreshConsensus(tickers,rev);
        if(gen===generation.current && rev===revision.current)setResult(current => current ? {...current,consensus:value}:current);
      } else if(tickers.length) {
        const value=await analyzeHome(tickers,rev);
        if(gen===generation.current && rev===revision.current)setResult(value);
      }
    } catch(exc) {if(gen===generation.current && rev===revision.current)setError((exc as Error).message);}
    finally {if(gen===generation.current)setBusy(false);}
  }
  async function favorite(ticker:string) {
    try {await toggleFavorite(ticker);} catch(exc) {setError((exc as Error).message);}
  }
  const selection=session?.selection ?? [];
  const ready=session && result && result.revision===session.revision && JSON.stringify(result.requested_tickers)===JSON.stringify(selection) ? result:null;
  if(loading || !session) return <div className="desk">{appError ? <Notice tone="negative">{appError}<button onClick={() => void refreshSession()}>Tentar novamente</button></Notice>:<Loading>Carregando sessão e lista de ações da B3…</Loading>}</div>;
  const analyzeLabel=busy ? "Analisando…":`Analisar ${selection.length || ""}`.trim();
  const resultSlot = ready ? <HomeResults ready={ready} selection={selection} busy={busy} onRefreshConsensus={()=>void run("consensus")}/> : <Pane title="Análise" area="res" meta={selection.length ? `${selection.length} ${selection.length===1 ? "ativo":"ativos"} na fila`:"aguardando seleção"} className="pane-idle">
    {busy ? <Loading>Buscando indicadores, comparação setorial e consensos de {selection.join(", ")}…</Loading>
      : error ? null
      : selection.length ? <Idle action={<p className="idle-keys"><kbd>Analisar</kbd> no Monitor busca indicadores, saúde financeira, preço-alvo e síntese setorial.</p>}>{stale ? "A seleção mudou desde a última análise. Analise de novo para atualizar os indicadores." : `Seleção pronta: ${selection.join(", ")}.`}</Idle>
      : <Idle action={<ol className="idle-steps"><li><kbd>Ctrl K</kbd> ou <kbd>/</kbd> abre a busca; <kbd>Enter</kbd> adiciona o ativo.</li><li>Ou clique em um setor no mapa para pegar os mais líquidos.</li><li><kbd>Analisar</kbd> preenche este painel com indicadores e comparação.</li><li><kbd>F2</kbd> leva a seleção para a Carteira.</li></ol>}>Nenhum ativo na seleção.</Idle>}
    {error && <Notice tone="negative">{error}<button disabled={busy} onClick={()=>void (selection.length ? run(retryAction) : suggestions(sectors).then(setSuggested).catch(exc=>setError(exc.message)))}>Tentar novamente</button></Notice>}
  </Pane>;
  return <div className={`desk desk-home${ready ? " is-analyzed":""}`}>
    <Monitor selection={selection} stocks={stocks} watchlist={session.watchlist} ready={ready} busy={busy || sectorBusy}
      analyze={<button type="button" className="primary" disabled={busy || sectorBusy || !selection.length} aria-busy={busy} onClick={()=>void run()}>{analyzeLabel}</button>}
      onQuick={ticker=>void startQuickAnalysis(ticker)} onFavorite={ticker=>void favorite(ticker)}
      onAdd={ticker=>{if(selection.length<MAX_SELECTION)void changeSelection(current=>Array.from(new Set([...current,ticker])));}}
      onRemove={ticker=>void changeSelection(current=>current.filter(item=>item!==ticker))}
      notices={<>
        {session.restored && <Notice>Seleção da carteira salva carregada. Revise e clique em Analisar.</Notice>}
        {session.restored && !!session.restoration?.invalid_count && <Notice tone="warning">{session.restoration.invalid_count} ação(ões) da carteira salva não estão na lista B3 e foram removidas.</Notice>}
        {session.restored && !!session.restoration?.truncated_count && <Notice>A seleção salva tinha {session.restoration.original_count} ações; carregamos as 20 primeiras válidas.</Notice>}
        {ready && !!ready.missing_tickers.length && <Notice tone="warning">Fundamentus indisponível para {ready.missing_tickers.join(", ")}.</Notice>}
      </>}/>
    {resultSlot}
    <SectorMap stocks={stocks} selection={selection} busy={busy || sectorBusy} active={sectors} available={suggested?.count ?? stocks.length} onSector={sector=>void changeSectors([sector])} onAdd={ticker=>{if(selection.length<MAX_SELECTION)void changeSelection(current=>Array.from(new Set([...current,ticker])));}} onReset={()=>void changeSectors([])}
      onRefresh={()=>void run("source")}
      note={notice ? <Notice tone="positive">{notice}{sectorUndo && <> <button type="button" onClick={()=>void undoSectorChange()}>Desfazer</button></>}</Notice> : sectorBusy ? <Loading>Ordenando ações do setor por liquidez…</Loading> : null}/>
    <Macro/>
    <Headlines/>
    {ready && <p className="desk-next">Próximo <Link href="/portfolio"><kbd>F2</kbd> Carteira</Link> com {selection.slice(0,4).join(", ")}{selection.length>4 ? "…":""} · <Link href="/screener"><kbd>F5</kbd> Screener</Link> para achar mais ativos</p>}
  </div>;
}
