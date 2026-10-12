"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import "./home.css";
import { useApp } from "../components/AppProvider";
import type { SelectionUpdate } from "../components/AppProvider";
import SelectionSummary from "./SelectionSummary";
import HomeResults from "../components/HomeResults";
import { Loading, Notice, PageHeader, TickerPicker } from "../components/ui";
import { analyzeHome, currentAnalysis, quickSectors, quickTickers, refreshConsensus, refreshFundamentus, suggestions } from "../lib/home";
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
  const [editRequested,setEditRequested] = useState(false);
  const [stale,setStale] = useState(false);
  const panel = useRef<HTMLElement>(null);
  const pendingAnalysis = useRef<string[]|null>(null);
  const generation = useRef(0);
  const sectorGeneration = useRef(0);
  const analyzedRevision = useRef<number|null>(null);
  const revision = useRef(session?.revision);
  revision.current = session?.revision;
  useEffect(() => {generation.current++;analyzedRevision.current=null;setResult(null);setBusy(false);},[session?.revision]);
  useEffect(() => {if(editRequested)panel.current?.querySelector<HTMLInputElement>("input[type=search]")?.focus();},[editRequested]);
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

  async function changeSelection(tickers:SelectionUpdate) {
    if(result)setStale(true);generation.current++;analyzedRevision.current=null;setResult(null);setBusy(false);setError(null);setEditRequested(false);
    try {await setSelection(tickers);} catch(exc) {setError((exc as Error).message);}
  }
  async function startQuickAnalysis(ticker:string) {
    generation.current++;analyzedRevision.current=null;setResult(null);setBusy(false);setError(null);setEditRequested(false);
    pendingAnalysis.current=[ticker];
    try {await setSelection([ticker]);} catch(exc) {pendingAnalysis.current=null;setError((exc as Error).message);}
  }
  async function changeSectors(next:string[]) {
    const gen=++sectorGeneration.current;
    setSectors(next);setSectorBusy(true);setError(null);
    try {
      const value=await suggestions(next);
      if(gen!==sectorGeneration.current)return;
      setSuggested(value);
      if(next.length && !next.includes("Todos")) {
        await changeSelection(value.autopick);
        setNotice(`Selecionamos os ${value.autopick.length} ativos mais líquidos de ${value.count} disponíveis. Você pode adicionar outros pelo campo abaixo.`);
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
    if(action!=="consensus")setEditRequested(false);
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
  if(loading || !session) return <div className="home-page"><PageHeader title="Fundamentos" description="Análise quantitativa · B3" icon="home"/>{appError ? <Notice tone="negative">{appError}<button onClick={() => void refreshSession()}>Tentar novamente</button></Notice>:<Loading>Carregando sessão e lista de ações da B3…</Loading>}</div>;
  const stockMap=new Map(stocks.map(stock => [stock.ticker,stock]));
  const ordered=[...(suggested?.tickers ?? []),...stocks.map(stock=>stock.ticker)].filter((ticker,index,array)=>array.indexOf(ticker)===index).map(ticker=>stockMap.get(ticker)).filter((stock):stock is typeof stocks[number]=>!!stock);
  const popularSectors=new Set(quickSectors.map(([,sector])=>sector));
  const allSectors=Array.from(new Set(stocks.map(stock=>stock.sector).filter(Boolean))).sort();
  const sectorOptions=["Todos",...allSectors.filter(sector=>popularSectors.has(sector)),...allSectors.filter(sector=>!popularSectors.has(sector))];
  const mode:"edit"|"collapsed"=ready && !editRequested ? "collapsed":"edit";
  return <div className="home-page">
    <PageHeader title="Fundamentos" description="Explore fundamentos, compare empresas e aprofunde sua análise da B3." icon="home"/>
    {session.restored && <Notice>Seleção da carteira salva carregada. Revise os ativos e clique em Analisar para atualizar os fundamentos.</Notice>}
    {session.restored && !!session.restoration?.invalid_count && <Notice tone="warning">{session.restoration.invalid_count} ação(ões) da carteira salva não estão na lista B3 e foram removidas da seleção.</Notice>}
    {session.restored && !!session.restoration?.truncated_count && <Notice>A seleção salva tinha {session.restoration.original_count} ações. Carregamos as primeiras 20 ações válidas para manter a análise estável.</Notice>}
    {notice && <Notice tone="positive">{notice}</Notice>}
    <div className={`home-desk${!selection.length ? " is-empty" : ""}`}>
      <section ref={panel} className="task-panel home-task" {...(mode==="collapsed" ? {"aria-label":"Seleção analisada"}:{"aria-labelledby":"home-task-title"})}>
        {mode==="collapsed" ? <SelectionSummary tickers={selection} onEdit={()=>setEditRequested(true)}/> : <>
        <h2 id="home-task-title" className="ui-section-heading">Selecione ativos para analisar</h2>
        <p className="home-lead">Escolha até 20 ações da B3 para ver indicadores, saúde financeira, preço-alvo e síntese setorial.</p>
        <TickerPicker value={selection} onChange={changeSelection} stocks={ordered} max={20} label="Buscar ações"/>
        {!!selection.length && <>
          <div className="actions"><button className="primary" disabled={busy || sectorBusy} aria-busy={busy} onClick={()=>void run()}>{busy ? "Analisando…":"Analisar"}</button><button disabled={busy} onClick={()=>void changeSelection([])}>Limpar seleção</button></div>
          <p className="caption">A seleção não é uma carteira analisada. Use <Link href="/portfolio">Portfólio</Link> para analisar a carteira completa.</p>
          {!ready && !busy && <Notice>{stale ? "A seleção mudou. Clique em Analisar para atualizar os indicadores." : "Clique em Analisar para buscar os indicadores das ações selecionadas."}</Notice>}
        </>}
        {sectorBusy && <Loading>Ordenando ações por liquidez…</Loading>}
        {!selection.length && <div className="home-quick">
          <h3>Ou escolha uma ação líquida</h3>
          <div className="quick-start-grid">{quickTickers.map(([ticker,sector])=><button className="quick-start-button" key={ticker} onClick={()=>void startQuickAnalysis(ticker)}><strong>{ticker}</strong><span className="quick-start-sector">{sector}</span></button>)}</div>
        </div>}
        <div className="home-source" role="status"><span><strong>{suggested?.count ?? stocks.length} ações</strong> disponíveis em {sectors.length && !sectors.includes("Todos") ? sectors.join(", "):"todos os setores"}</span><span>Lista local B3 · Fundamentus · ordenação por liquidez</span></div>
        </>}
      </section>
      {!selection.length && <aside className="home-guide" aria-labelledby="home-guide-title">
        <h2 id="home-guide-title">Como funciona</h2>
        <ol>
          <li><strong>Escolha</strong> Filtre por setor ou pesquise diretamente pelo ticker.</li>
          <li><strong>Compare</strong> Compare fundamentos, rentabilidade, crescimento e endividamento lado a lado.</li>
          <li><strong>Aprofunde</strong> Carregue a análise para comparar os preços-alvo e o consenso dos ativos.</li>
        </ol>
        <p>A seleção não é uma carteira analisada. Use <Link href="/portfolio">Portfólio</Link> para analisar a carteira completa.</p>
      </aside>}
    </div>
    <details className="home-tools">
      <summary>Setores e favoritos</summary>
      <div className="home-tools-grid">
        <div>
          <h3>Setores</h3>
          <fieldset className="checkbox-group sector-picker">
            <legend>Escolha um ou mais setores</legend>
            <div className="checkbox-group-options sector-picker-options">{sectorOptions.map(sector=>{
              const checked=sector==="Todos" ? !sectors.length || sectors.includes("Todos"):sectors.includes(sector);
              return <label key={sector}><input type="checkbox" checked={checked} onChange={()=>void changeSectors(sector==="Todos" ? []:checked ? sectors.filter(item=>item!==sector):[...sectors.filter(item=>item!=="Todos"),sector])} />{sector}</label>;
            })}</div>
          </fieldset>
          <p>Nenhum setor ou Todos inclui toda a lista B3. Os setores mais buscados aparecem primeiro.</p>
          <button disabled={busy} onClick={()=>void run("source")}>Atualizar dados Fundamentus</button>
        </div>
        <div>
          <h3>Favoritos</h3>
          {!session.watchlist.length && <p>Nenhum favorito ainda. Depois de analisar, use Favoritar para guardar ativos aqui.</p>}
          {session.watchlist.map(ticker=>{const full=selection.length>=20 && !selection.includes(ticker);return <div className="home-favorite-row" key={ticker}>
            {stockMap.has(ticker) ? <><button aria-disabled={full} onClick={()=>{if(!full)void changeSelection(current=>Array.from(new Set([...current,ticker])))}}>{ticker}</button>{full && <span>Seleção cheia</span>}</>:<span>{ticker} · fora da lista B3</span>}
            <button aria-label={`Remover ${ticker} dos favoritos`} onClick={()=>void favorite(ticker)}>Remover favorito</button>
          </div>;})}
        </div>
      </div>
    </details>
    {busy && <Loading>Buscando indicadores, comparação setorial e consensos dos ativos selecionados…</Loading>}
    {error && <Notice tone="negative">{error}<button disabled={busy} onClick={()=>void (selection.length ? run(retryAction) : suggestions(sectors).then(setSuggested).catch(exc=>setError(exc.message)))}>Tentar novamente</button></Notice>}
    {ready && <HomeResults ready={ready} selection={selection} watchlist={session.watchlist} busy={busy} onFavorite={ticker=>void favorite(ticker)} onRefreshConsensus={()=>void run("consensus")}/>}
  </div>;
}
