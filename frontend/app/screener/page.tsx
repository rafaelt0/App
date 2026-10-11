"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useApp } from "../../components/AppProvider";
import { ExportButton, Loading, Metrics, Notice, PageHeader, Section, Table } from "../../components/ui";
import { refreshFundamentus } from "../../lib/home";
import { applyPreset, canonicalCriteria, defaults, editControl, queryScreener, screenerConfig } from "../../lib/screener";
import type { Config, Controls, Result, Sort } from "../../lib/screener";

export default function ScreenerPage() {
  const {session,stocks,loading,error:appError,setSelection,toggleFavorite,refreshSession} = useApp();
  const router=useRouter();
  const [config,setConfig]=useState<Config|null>(null);
  const [controls,setControls]=useState<Controls>(defaults);
  const [sort,setSort]=useState<Sort>({field:"liq2m",direction:"desc"});
  const [result,setResult]=useState<Result|null>(null);
  const [limit,setLimit]=useState(25);
  const [visible,setVisible]=useState<string[]|null>(null);
  const [selected,setSelected]=useState<string|null>(null);
  const [busy,setBusy]=useState(false);
  const [actionBusy,setActionBusy]=useState(false);
  const [error,setError]=useState<string|null>(null);
  const [notice,setNotice]=useState<string|null>(null);
  const [reload,setReload]=useState(0);
  const generation=useRef(0);
  const orderedTickers=useRef("");
  const revision=useRef(session?.revision);
  revision.current=session?.revision;
  useEffect(()=>{
    let active=true;
    screenerConfig().then(value=>{if(active)setConfig(value);}).catch(exc=>{if(active)setError(exc.message);});
    return ()=>{active=false;generation.current++;};
  },[reload]);
  useEffect(()=>{
    if(!session || !config)return;
    const gen=++generation.current;
    const rev=session.revision;
    setResult(null);setBusy(true);setError(null);
    if(controls.pl_enabled && controls.pl_min>controls.pl_max) {setError("O P/L mínimo não pode ser maior que o P/L máximo.");setBusy(false);return;}
    const criteria=canonicalCriteria(controls);
    if(Object.values(criteria).some(value=>typeof value==="number" && !Number.isFinite(value))) {
      setError("Preencha todos os limites dos critérios ativos com números válidos.");setBusy(false);return;
    }
    const timer=setTimeout(()=>{
      queryScreener(criteria,sort).then(value=>{
        if(gen!==generation.current || revision.current!==rev)return;
        const identity=value.rows.map(row=>String(row.ticker)).join("|");
        if(identity!==orderedTickers.current)setSelected(null);
        orderedTickers.current=identity;
        setResult(value);
        setLimit(25);
        setVisible(current=>current===null ? value.default_columns:current.filter(key=>value.available_columns.some(column=>column.key===key)));
      }).catch(exc=>{if(gen===generation.current && revision.current===rev)setError(exc.message);}).finally(()=>{if(gen===generation.current)setBusy(false);});
    },200);
    return ()=>{clearTimeout(timer);generation.current++;};
  },[config,controls,sort,reload,session?.revision]);
  async function sourceRefresh() {
    generation.current++;setResult(null);setSelected(null);setActionBusy(true);setError(null);
    try {const response=await refreshFundamentus();setNotice(`Cache Fundamentus atualizado (${response.removed} entradas removidas). ${response.scope}.`);setReload(value=>value+1);}
    catch(exc){setError((exc as Error).message);}
    finally{setActionBusy(false);}
  }
  async function handoff() {
    if(!selected)return;
    setActionBusy(true);setError(null);
    try {
      const known=stocks.some(stock=>stock.ticker===selected);
      await setSelection(known ? [selected]:[]);
      router.push(known ? "/":`/?unavailable=${encodeURIComponent(selected)}`);
    } catch(exc){setError((exc as Error).message);}finally{setActionBusy(false);}
  }
  async function favorite() {
    if(!selected)return;
    setActionBusy(true);setError(null);
    try{await toggleFavorite(selected);}catch(exc){setError((exc as Error).message);}finally{setActionBusy(false);}
  }
  function edit<K extends keyof Controls>(key:K,value:Controls[K]) {setControls(current=>editControl(current,key,value));}
  if(loading || !session)return <><PageHeader title="Screener de ações" description="Explore ações da B3 com critérios claros e ajustáveis." icon="filter"/>{appError ? <Notice tone="negative">{appError}<button onClick={()=>void refreshSession()}>Tentar novamente</button></Notice>:<Loading/>}</>;
  const selectedCurrent=selected && result?.rows.some(row=>row.ticker===selected) ? selected:null;
  return <>
    <PageHeader title="Screener de ações" description="Explore ações listadas na B3 com critérios claros e ajustáveis. Dados da Fundamentus; consulte a coleta exibida abaixo." icon="filter"/>
    {notice && <Notice tone="positive">{notice}</Notice>}
    <div className="page-columns"><aside className="panel">
      <label>Ponto de partida<select value={controls.preset} onChange={event=>{const preset=config?.presets.find(item=>item.name===event.target.value);if(preset)setControls(current=>applyPreset(current,preset.name,preset.criteria));}}>{config?.presets.map(preset=><option key={preset.name}>{preset.name}</option>)}</select></label>
      <p>{config?.presets.find(preset=>preset.name===controls.preset)?.description}</p>
      <button onClick={()=>setControls({...defaults})}>Redefinir para Explorar B3</button>
      <h2>Personalize critérios</h2>
      <fieldset><legend>Valuation</legend><label><input type="checkbox" checked={controls.pl_enabled} onChange={event=>edit("pl_enabled",event.target.checked)}/>Aplicar faixa de P/L</label>
        <label>P/L mínimo<input type="number" min={-50} max={200} step={.5} value={controls.pl_min} disabled={!controls.pl_enabled} onChange={event=>edit("pl_min",event.target.valueAsNumber)}/></label>
        <label>P/L máximo<input type="number" min={-50} max={200} step={.5} value={controls.pl_max} disabled={!controls.pl_enabled} onChange={event=>edit("pl_max",event.target.valueAsNumber)}/></label>
        {controls.pl_enabled && controls.pl_min_exclusive && <p>Mínimo exclusivo: P/L &gt; {controls.pl_min}. Editar o mínimo torna a comparação inclusiva.</p>}
      </fieldset>
      <fieldset><legend>Qualidade</legend><label><input type="checkbox" checked={controls.roe_enabled} onChange={event=>edit("roe_enabled",event.target.checked)}/>Aplicar ROE mínimo</label><label>ROE mínimo (%)<input type="number" min={0} max={80} step={1} value={controls.roe_min} disabled={!controls.roe_enabled} onChange={event=>edit("roe_min",event.target.valueAsNumber)}/></label></fieldset>
      <fieldset><legend>Crescimento</legend><label><input type="checkbox" checked={controls.c5y_enabled} onChange={event=>edit("c5y_enabled",event.target.checked)}/>Aplicar crescimento da receita (5 anos)</label><label>Crescimento mínimo (%)<input type="number" min={0} max={100} step={1} value={controls.c5y_min} disabled={!controls.c5y_enabled} onChange={event=>edit("c5y_min",event.target.valueAsNumber)}/></label></fieldset>
      <fieldset><legend>Dividendos</legend><label><input type="checkbox" checked={controls.dy_enabled} onChange={event=>edit("dy_enabled",event.target.checked)}/>Aplicar dividend yield mínimo</label><label>Dividend yield mínimo (%)<input type="number" min={0} max={30} step={.5} value={controls.dy_min} disabled={!controls.dy_enabled} onChange={event=>edit("dy_min",event.target.valueAsNumber)}/></label></fieldset>
      <fieldset><legend>Liquidez</legend><label><input type="checkbox" checked={controls.liq2m_enabled} onChange={event=>edit("liq2m_enabled",event.target.checked)}/>Aplicar liquidez mínima</label><label>Liquidez média 2 meses mínima (R$)<input type="range" min={0} max={100000000} step={100000} value={controls.liq2m_min} disabled={!controls.liq2m_enabled} onChange={event=>edit("liq2m_min",event.target.valueAsNumber)}/><output>{(controls.liq2m_min/1000000).toLocaleString("pt-BR")} {controls.liq2m_min===1000000 ? "milhão":"milhões"}</output></label></fieldset>
      <label>Ordenar resultados<select value={sort.field} onChange={event=>{const value=config?.sorts.find(option=>option.field===event.target.value);if(value)setSort(value);}}>{config?.sorts.map(option=><option key={option.field} value={option.field}>{option.label}</option>)}</select></label>
      <button disabled={actionBusy} onClick={()=>void sourceRefresh()}>Atualizar dados Fundamentus</button>
    </aside><div>
      {(busy || actionBusy || !config) && !error && <Loading>Carregando dados da B3…</Loading>}
      {error && <Notice tone="negative">{error}<button disabled={actionBusy} onClick={()=>setReload(value=>value+1)}>Tentar novamente</button></Notice>}
      {result && <Section title="Resultados">
        <p>{result.matched_count} de {result.total_count} ativos correspondem aos filtros · {result.active_filters.join(" · ") || "sem critérios ativos"}</p>
        <p>{result.fetched_at ? `Fundamentus coletado em ${new Date(result.fetched_at).toLocaleString("pt-BR")} · há ${result.age_minutes} min`:"Horário da coleta indisponível; a idade deste dado não pode ser confirmada."}</p>
        <Metrics items={[{label:"Ações correspondentes",value:result.matched_count,unit:"integer"},{label:"DY médio",value:result.metrics.mean_dy,unit:"percent",digits:2},{label:"ROE médio",value:result.metrics.mean_roe,unit:"percent",digits:2}]}/>
        {!result.matched_count ? <Notice tone="warning">Nenhuma ação encontrada. Reduza os limites ou desative critérios; dados ausentes não são considerados aprovação.</Notice>:<>
          <fieldset className="checkbox-group"><legend>Colunas exibidas</legend><div className="checkbox-group-options">{result.available_columns.map(column=><label key={column.key}><input type="checkbox" checked={visible?.includes(column.key) ?? false} onChange={event=>setVisible(current=>{const next=new Set(current ?? []);if(event.target.checked)next.add(column.key);else next.delete(column.key);return result.available_columns.map(item=>item.key).filter(key=>next.has(key));})}/>{column.label}</label>)}</div></fieldset>
          <p>Selecione uma linha para abrir a análise de preço-alvo ou gerenciar favoritos. As colunas exibidas não alteram a seleção nem o CSV.</p>
          {selectedCurrent && <div className="panel"><strong>Ação selecionada: {selectedCurrent}</strong><div className="actions"><button disabled={actionBusy} onClick={()=>void handoff()}>Ver preço-alvo</button><button disabled={actionBusy} onClick={()=>void favorite()}>{session.watchlist.includes(selectedCurrent) ? "★ Favoritado":"☆ Favoritar"}</button></div><p>Abrir Home substitui a seleção; clique em Analisar para buscar fundamentos e consenso.</p></div>}
          <Table columns={[{key:"ticker",label:"Papel"},...result.available_columns.filter(column=>visible?.includes(column.key))]} rows={result.rows.slice(0,limit)} rowKey="ticker" selectedKey={selectedCurrent ?? undefined} onSelect={row=>setSelected(String(row.ticker))}/>
          {result.rows.length>limit ? <button onClick={()=>setLimit(previous=>previous+25)}>Ver mais {Math.min(25,result.rows.length-limit)} ações ({result.rows.length-limit} restantes)</button>:result.rows.length>25 ? <p className="caption">Todas as {result.rows.length} ações exibidas.</p>:null}
          <ExportButton path={`/api/screener/${result.id}/results.csv`}>Exportar todos os {result.matched_count} resultados como CSV</ExportButton>
        </>}
      </Section>}
    </div></div>
    <hr/><p>Fonte: Fundamentus (fundamentus.com.br). Dados com atraso. Esta triagem não é recomendação de investimento; verifique os indicadores e faça sua própria análise.</p>
  </>;
}
