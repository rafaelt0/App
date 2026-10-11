"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useApp } from "../components/AppProvider";
import type { SelectionUpdate } from "../components/AppProvider";
import Chart from "../components/Chart";
import { ExportButton, Loading, Metrics, Notice, PageHeader, Section, Table, TickerPicker } from "../components/ui";
import type { Column } from "../components/ui";
import { analyzeHome, consensusColumns, identityColumns, marketColumns, quickSectors, quickTickers, refreshConsensus, refreshFundamentus, suggestions } from "../lib/home";
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
  const [comparison,setComparison] = useState("pl");
  const [retryAction,setRetryAction] = useState<"analyze"|"consensus"|"source">("analyze");
  const generation = useRef(0);
  const sectorGeneration = useRef(0);
  const analyzedRevision = useRef<number|null>(null);
  const revision = useRef(session?.revision);
  revision.current = session?.revision;
  useEffect(() => {generation.current++;analyzedRevision.current=null;setResult(null);setBusy(false);},[session?.revision]);
  useEffect(() => {
    const unavailable=new URLSearchParams(window.location.search).get("unavailable");
    if(unavailable)setNotice(`${unavailable} está fora da lista local B3. A seleção foi removida; escolha outro ativo para analisar.`);
    let active=true;
    const request=++sectorGeneration.current;
    suggestions([]).then(value => {if(active && request===sectorGeneration.current)setSuggested(value);}).catch(exc => {if(active && request===sectorGeneration.current)setError(String(exc.message));});
    return () => {active=false;generation.current++;sectorGeneration.current++;};
  },[]);

  async function changeSelection(tickers:SelectionUpdate) {
    generation.current++;analyzedRevision.current=null;setResult(null);setBusy(false);setError(null);
    try {await setSelection(tickers);} catch(exc) {setError((exc as Error).message);}
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
    setRetryAction(action);setBusy(true);setError(null);
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
  if(loading || !session) return <><PageHeader title="Fundamentos" description="Análise quantitativa · B3" icon="home"/>{appError ? <Notice tone="negative">{appError}<button onClick={() => void refreshSession()}>Tentar novamente</button></Notice>:<Loading/>}</>;
  const selection=session.selection;
  const ready=result && result.revision===session.revision && JSON.stringify(result.requested_tickers)===JSON.stringify(selection) ? result:null;
  const stockMap=new Map(stocks.map(stock => [stock.ticker,stock]));
  const ordered=[...(suggested?.tickers ?? []),...stocks.map(stock=>stock.ticker)].filter((ticker,index,array)=>array.indexOf(ticker)===index).map(ticker=>stockMap.get(ticker)).filter((stock):stock is typeof stocks[number]=>!!stock);
  const sectorOptions=["Todos",...Array.from(new Set(stocks.map(stock=>stock.sector).filter(Boolean))).sort()];
  const companies=ready?.companies ?? [];
  const single=selection.length===1;
  const choice=ready?.comparison_indicators.find(item=>item.key===comparison);
  const chartCompanies=companies.filter(company=>company.indicators_available);
  return <>
    <PageHeader title="Fundamentos" description="Explore fundamentos, compare empresas e aprofunde sua análise da B3." icon="home"/>
    {session.restored && <Notice>Seleção da carteira salva carregada. Revise os ativos e clique em Analisar para atualizar os fundamentos.</Notice>}
    {session.restored && !!session.restoration?.invalid_count && <Notice tone="warning">{session.restoration.invalid_count} ação(ões) da carteira salva não estão na lista B3 e foram removidas da seleção.</Notice>}
    {session.restored && !!session.restoration?.truncated_count && <Notice>A seleção salva tinha {session.restoration.original_count} ações. Carregamos as primeiras 20 ações válidas para manter a análise estável.</Notice>}
    {notice && <Notice tone="positive">{notice}</Notice>}
    <section className="task-panel">
      <h2 className="ui-section-heading">Selecione ativos para analisar</h2>
      <TickerPicker value={selection} onChange={changeSelection} stocks={ordered} max={20}/>
      {!!selection.length && <><div className="actions"><button className="primary" disabled={busy || sectorBusy} onClick={()=>void run()}>Analisar</button><button onClick={()=>void changeSelection([])}>Limpar</button></div>{!ready && !busy && <Notice>Clique em Analisar para buscar os indicadores das ações selecionadas.</Notice>}</>}
      <p>A seleção não é uma carteira analisada. Use <Link href="/portfolio">Portfólio</Link> para analisar a carteira completa.</p>
      <div className="home-source" role="status"><strong>{suggested?.count ?? stocks.length} ações</strong> disponíveis em {sectors.length && !sectors.includes("Todos") ? sectors.join(", "):"todos os setores"} · Lista local B3 · Fundamentus · ordenação por liquidez</div>
      {sectorBusy && <Loading>Ordenando ações por liquidez…</Loading>}
      {!selection.length && <>
        <p>Pesquise pelo ticker ou nome da empresa ou use a seleção rápida abaixo.</p>
        <h3>Seleção rápida · ações líquidas</h3>
        <div className="quick-start-grid">{quickTickers.map(([ticker,sector])=><button className="quick-start-button" key={ticker} onClick={()=>void changeSelection([ticker])}><strong>{ticker}</strong><span className="quick-start-sector">{sector}</span></button>)}</div>
      </>}
    </section>
    <details className="home-tools">
      <summary>Setores e favoritos</summary>
      <div className="home-tools-grid">
        <div>
          <h3>Setores</h3>
          <label>Escolha um ou mais setores<select multiple value={sectors} onChange={event=>void changeSectors(Array.from(event.target.selectedOptions,option=>option.value))}>{sectorOptions.map(sector=><option key={sector}>{sector}</option>)}</select></label>
          <p>Nenhum setor ou Todos inclui toda a lista B3.</p>
          {!selection.length && <><h3>Explorar · setores</h3><div className="actions">{quickSectors.map(([label,sector])=><button key={label} onClick={()=>void changeSectors([sector])}>{label}</button>)}</div></>}
          <button disabled={busy} onClick={()=>void run("source")}>Atualizar dados Fundamentus</button>
        </div>
        <div>
          <h3>Favoritos</h3>
          {!session.watchlist.length && <p>Nenhum favorito ainda.</p>}
          {session.watchlist.map(ticker=><div className="actions" key={ticker}>
            {stockMap.has(ticker) ? <button disabled={selection.length>=20 && !selection.includes(ticker)} onClick={()=>void changeSelection(current=>Array.from(new Set([...current,ticker])))}>{ticker}</button>:<span>{ticker} · fora da lista B3</span>}
            <button aria-label={`Remover ${ticker} dos favoritos`} onClick={()=>void favorite(ticker)}>Remover</button>
          </div>)}
        </div>
      </div>
    </details>
    {!selection.length && <details className="home-help">
      <summary>Como funciona</summary>
      <ol>
        <li><strong>Escolha</strong> · Filtre por setor ou pesquise diretamente pelo ticker.</li>
        <li><strong>Compare</strong> · Compare fundamentos, rentabilidade, crescimento e endividamento lado a lado.</li>
        <li><strong>Aprofunde</strong> · Carregue a análise para comparar os preços-alvo e o consenso dos ativos.</li>
      </ol>
    </details>}
    {busy && <Loading>Buscando indicadores, comparação setorial e consensos dos ativos selecionados…</Loading>}
    {error && <Notice tone="negative">{error}<button disabled={busy} onClick={()=>void (selection.length ? run(retryAction) : suggestions(sectors).then(setSuggested).catch(exc=>setError(exc.message)))}>Tentar novamente</button></Notice>}
    {ready && <>
      {!!ready.missing_tickers.length && <Notice tone="warning">Dados Fundamentus indisponíveis para {ready.missing_tickers.join(", ")}. Os dados disponíveis foram preservados e os caches da fonte invalidados.</Notice>}
      <ExportButton path={`/api/home/${ready.id}/fundamentals.csv`}>Exportar dados fundamentalistas (CSV bruto)</ExportButton>
      <Section title="Setor">
        {single ? companies.map(company=>company.available ? <div className="identity-metrics" key={company.ticker}><Metrics items={[{label:"Empresa",value:company.identity.company},{label:"Setor",value:company.identity.sector},{label:"Subsetor",value:company.identity.subsector}]}/></div>:<Notice key={company.ticker}>Sem dados de setor para {company.ticker}.</Notice>):<Table columns={identityColumns} rows={companies.map(company=>({ticker:company.ticker,...company.identity}))} rowKey="ticker"/>}
      </Section>
      <Section title="Informações de Mercado">
        {single ? companies.map(company=>company.available ? <div key={company.ticker}><Metrics items={marketColumns.filter(column=>column.unit).map(column=>({label:column.label,value:company.market[column.key as keyof typeof company.market] ?? 0,unit:column.unit,digits:column.digits}))}/><p>Última cotação registrada: {company.market.last_quote_date ?? "—"} para {company.ticker}</p></div>:<Notice key={company.ticker}>Sem dados de mercado para {company.ticker}.</Notice>):<Table columns={marketColumns} rows={companies.map(company=>({ticker:company.ticker,...company.market}))} rowKey="ticker"/>}
      </Section>
      <Section title="Indicadores Financeiros">
        <div className="actions">{companies.map(company=><button key={company.ticker} onClick={()=>void favorite(company.ticker)}>{session.watchlist.includes(company.ticker) ? "★ Favoritado":"☆ Favoritar"} · {company.ticker}</button>)}</div>
        {single ? companies.map(company=>company.indicators_available ? <div key={company.ticker}>{Object.entries(company.metric_groups).map(([group,metrics])=><div key={group}><h3>{group}</h3><Metrics items={metrics.map(metric=>({...metric,value:metric.value ?? (metric.label==="EV/EBITDA" ? null:0)}))}/><div className="actions">{metrics.filter(metric=>metric.classification).map(metric=><span key={metric.label} className={`signal-${metric.status}`}>{metric.label}: {metric.classification}</span>)}</div></div>)}</div>:<Notice key={company.ticker}>Sem indicadores para {company.ticker}.</Notice>):<><p>Rentabilidade, margens e crescimento em % · múltiplos em × · dados ausentes: —.</p><Table columns={[{key:"ticker",label:"Ticker"},...ready.comparison_indicators.map(item=>({key:item.key,label:item.label,unit:item.unit,digits:2} as Column))]} rows={companies.map(company=>({ticker:company.ticker,...company.indicators}))} rowKey="ticker"/></>}
      </Section>
      <Section title="Saúde Financeira"><p>Endividamento e liquidez da empresa. Dívida/PL acima de 3× e liquidez abaixo de 1× são sinais de alerta.</p>{companies.map(company=><div key={company.ticker}><h3>{company.ticker}</h3>{company.health.metrics.length ? <><Metrics items={company.health.metrics}/>{company.health.diagnostics.map(diag=><Notice key={diag.text} tone={diag.status}>{diag.text}</Notice>)}</>:<Notice>Dados de endividamento não disponíveis via Fundamentus para este ticker.</Notice>}</div>)}</Section>
      {!single && <Section title="Comparação Gráfica de Múltiplos"><label>Indicador comparativo<select value={comparison} onChange={event=>setComparison(event.target.value)}>{ready.comparison_indicators.map(item=><option value={item.key} key={item.key}>{item.label}</option>)}</select></label><Chart figure={{data:[{type:"bar",x:chartCompanies.map(company=>company.ticker),y:chartCompanies.map(company=>company.indicators[comparison]),text:chartCompanies.map(company=>{const value=company.indicators[comparison];return value===null ? "N/D":!company.indicator_available[comparison] ? "N/A":`${value.toFixed(2)}${choice?.unit==="percent" ? "%":""}`;}),textposition:"outside",cliponaxis:false,marker:{color:"#8b93ff"}}],layout:{height:350,title:{text:`Comparativo de ${choice?.label}`},yaxis:{title:{text:choice?.unit==="percent" ? `${choice.label} (%)`:choice?.label}}}}}/></Section>}
      <Section title="Preço-alvo e consenso"><button disabled={busy} onClick={()=>void run("consensus")}>Atualizar consensos</button><Table columns={consensusColumns} rows={ready.consensus.rows} rowKey="ticker"/>{ready.consensus.failed_tickers.length ? <Notice tone="warning">Consenso indisponível para: {ready.consensus.failed_tickers.join(", ")}.</Notice>:!ready.consensus.has_targets && <Notice>O Yahoo Finance não informa preços-alvo para as ações selecionadas.</Notice>}<p>Fonte: Yahoo Finance. Potencial calculado sobre o alvo médio; opiniões e recomendações são agregadas pela fonte e não constituem recomendação de investimento.</p></Section>
      <Section title="Síntese do Analista">{companies.map(company=><article className="analyst-synthesis-card" key={company.ticker}>
        <h3>{company.ticker} · {company.identity.company ?? "—"}</h3><span className="analyst-synthesis-sector">{company.synthesis.sector ?? "Setor indisponível"}</span>
        <p>{company.synthesis.favorable_categories}/{company.synthesis.categorias_validas} áreas favoráveis · {company.synthesis.indicadores_validos} indicadores elegíveis</p>
        {company.synthesis.categorias_validas<2 && <Notice>Dados insuficientes para uma conclusão setorial: são necessárias pelo menos duas áreas com indicadores e no mínimo três observações.</Notice>}
        <div className="mcard-grid">{Object.entries(company.synthesis.categorias).map(([name,category])=><div className={`analyst-category ${category.veredicto==="Favorável" ? "is-positive":category.veredicto==="Desfavorável" ? "is-negative":"is-neutral"}`} key={name}><h4>{name==="Yield" ? "Dividendos":name}</h4><span className="analyst-category-status">{category.veredicto}</span><div className="analyst-synthesis-meter"><span style={{width:`${category.percentil}%`}}/></div><p>Percentil {category.percentil.toFixed(1)} · {category.indicadores_validos} indicadores</p></div>)}</div>
        {["Favorável","Neutro","Desfavorável"].map(verdict=>{const details=company.synthesis.details.filter(detail=>detail.verdict===verdict);return details.length ? <details className="analyst-synthesis-details" key={verdict}><summary>{verdict} · {details.length} indicadores</summary><Table columns={[{key:"label",label:"Indicador"},{key:"value",label:"Valor"},{key:"unit",label:"Unidade"},{key:"median",label:"Mediana do setor"},{key:"mean",label:"Média do setor"},{key:"percentile",label:"Percentil",unit:"number",digits:1},{key:"observations",label:"Observações",unit:"integer"}]} rows={details.map(detail=>({...detail,unit:detail.unit==="percent" ? "%":"×"}))} rowKey="label"/><p>Percentil é a proporção de observações estritamente superadas; inclui o próprio ticker. Empates não contam como superados.</p></details>:null;})}
        <p>Fonte: Fundamentus · comparação setorial. Áreas e percentis não constituem recomendação de investimento.</p>
      </article>)}</Section>
      <div className="next-step-card"><div className="next-step-eyebrow">Próximo passo</div><h2>Analise sua carteira</h2><p>Monte e otimize a carteira com {selection.slice(0,3).join(", ")}{selection.length>3 ? "…":""}.</p><Link href="/portfolio">Abrir Portfolio</Link></div>
    </>}
  </>;
}
