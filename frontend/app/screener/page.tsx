"use client";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import "./screener.css";
import { useApp } from "../../components/AppProvider";
import { ExportButton, Icon, Loading, Metrics, Notice, PageHeader, Section } from "../../components/ui";
import { refreshFundamentus } from "../../lib/home";
import { activeCriteria, applyPreset, canonicalCriteria, criterionDefs, defaults, editControl, queryScreener, screenerConfig, sortKey } from "../../lib/screener";
import type { Config, Controls, CriterionDef, CriterionKey, Result, ScreenerColumn, Sort } from "../../lib/screener";
import ScreenerTable from "./ScreenerTable";

const liquidityShortcuts = [1_000_000, 2_000_000, 5_000_000, 20_000_000];

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
  const barRef=useRef<HTMLDivElement>(null);
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
    setBusy(true);setError(null);
    if(controls.pl_enabled && controls.pl_min>controls.pl_max) {setResult(null);setError("O P/L mínimo não pode ser maior que o P/L máximo.");setBusy(false);return;}
    const criteria=canonicalCriteria(controls);
    if(Object.values(criteria).some(value=>typeof value==="number" && !Number.isFinite(value))) {
      setResult(null);setError("Preencha todos os limites dos critérios ativos com números válidos.");setBusy(false);return;
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
      }).catch(exc=>{if(gen===generation.current && revision.current===rev){setResult(null);setError(exc.message);}}).finally(()=>{if(gen===generation.current)setBusy(false);});
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
  function sortBy(field:string) {
    setSort(current => current.field===field ? {field,direction:current.direction==="asc" ? "desc":"asc"} : {field,direction:"desc"});
  }
  function columnLabel(field:string) {
    return config?.columns.find(column=>column.key===field)?.label ?? field;
  }
  function sortLabel(option:Sort) {
    return option.label ?? `${columnLabel(option.field)} (${option.direction==="asc" ? "menor primeiro":"maior primeiro"})`;
  }
  async function favorite() {
    if(!selected)return;
    setActionBusy(true);setError(null);
    try{await toggleFavorite(selected);}catch(exc){setError((exc as Error).message);}finally{setActionBusy(false);}
  }
  function removeCriterion(key:CriterionDef["enabled"]) {edit(key,false);barRef.current?.focus();}
  function edit<K extends keyof Controls>(key:K,value:Controls[K]) {setControls(current=>editControl(current,key,value));}
  if(loading || !session)return <div className="screener-page"><PageHeader title="Screener de ações" description="Explore ações da B3 com critérios claros e ajustáveis." icon="filter"/>{appError ? <Notice tone="negative">{appError}<button onClick={()=>void refreshSession()}>Tentar novamente</button></Notice>:<Loading/>}</div>;
  const selectedCurrent=selected && result?.rows.some(row=>row.ticker===selected) ? selected:null;
  const preset=config?.presets.find(item=>item.name===controls.preset);
  const plInvalid=controls.pl_enabled && controls.pl_min>controls.pl_max;
  const liquidity=controls.liq2m_min.toLocaleString("pt-BR",{style:"currency",currency:"BRL",maximumFractionDigits:0});
  const inputs:Record<CriterionKey,ReactNode>={
    pl:<>
      <div className="scr-pair">
        <label>P/L mínimo (×)<input type="number" inputMode="decimal" min={-50} max={200} step={.5} value={Number.isNaN(controls.pl_min) ? "":controls.pl_min} aria-invalid={plInvalid} aria-describedby={plInvalid ? "scr-pl-error":undefined} onChange={event=>edit("pl_min",event.target.valueAsNumber)}/></label>
        <label>P/L máximo (×)<input type="number" inputMode="decimal" min={-50} max={200} step={.5} value={Number.isNaN(controls.pl_max) ? "":controls.pl_max} aria-invalid={plInvalid} aria-describedby={plInvalid ? "scr-pl-error":undefined} onChange={event=>edit("pl_max",event.target.valueAsNumber)}/></label>
      </div>
      {plInvalid && <p className="scr-field-error" id="scr-pl-error">⚠ O mínimo não pode ser maior que o máximo.</p>}
      {controls.pl_min_exclusive && <p className="caption">Mínimo exclusivo: P/L &gt; {controls.pl_min}. Editar o mínimo torna a comparação inclusiva.</p>}
    </>,
    roe:<label>ROE mínimo (%)<input type="number" inputMode="decimal" min={0} max={80} step={1} value={Number.isNaN(controls.roe_min) ? "":controls.roe_min} onChange={event=>edit("roe_min",event.target.valueAsNumber)}/></label>,
    c5y:<label>Crescimento mínimo (%)<input type="number" inputMode="decimal" min={0} max={100} step={1} value={Number.isNaN(controls.c5y_min) ? "":controls.c5y_min} onChange={event=>edit("c5y_min",event.target.valueAsNumber)}/></label>,
    dy:<label>Dividend yield mínimo (%)<input type="number" inputMode="decimal" min={0} max={30} step={.5} value={Number.isNaN(controls.dy_min) ? "":controls.dy_min} onChange={event=>edit("dy_min",event.target.valueAsNumber)}/></label>,
    liq2m:<>
      <label>Liquidez média 2 meses mínima<input type="range" min={0} max={100000000} step={100000} value={Number.isNaN(controls.liq2m_min) ? 0:controls.liq2m_min} onChange={event=>edit("liq2m_min",event.target.valueAsNumber)}/><output className="scr-output">{liquidity}</output></label>
      <div className="scr-quick-amounts" role="group" aria-label="Valores rápidos de liquidez">{liquidityShortcuts.map(amount=><button type="button" key={amount} aria-pressed={controls.liq2m_min===amount} onClick={()=>edit("liq2m_min",amount)}>R$ {amount/1_000_000} mi</button>)}</div>
    </>,
  };
  const chips=activeCriteria(controls);
  const presetSortKeys=new Set((config?.sorts ?? []).map(sortKey));
  const sortOptions:Sort[]=config ? [...config.sorts,...(presetSortKeys.has(sortKey(sort)) ? []:[sort])]:[];
  const columnCount=visible?.length ?? 0;
  return <div className="screener-page">
    <PageHeader title="Screener de ações" description="Explore ações listadas na B3 com critérios claros e ajustáveis. Dados da Fundamentus; consulte a coleta exibida abaixo." icon="filter"/>
    {notice && <Notice tone="positive">{notice}</Notice>}
    <div className="scr-workspace">
    <section className="task-panel scr-filters" aria-labelledby="scr-filters-title">
      <div className="task-heading"><span className="task-step" aria-hidden="true">01</span><div className="task-heading-content"><h2 id="scr-filters-title">Defina os critérios</h2><p>Escolha um ponto de partida e ajuste os limites da busca.</p></div></div>
      <div className="scr-setup">
        <label>Ponto de partida<select value={config ? controls.preset:""} disabled={!config} onChange={event=>{const next=config?.presets.find(item=>item.name===event.target.value);if(next)setControls(current=>applyPreset(current,next.name,next.criteria));}}>{config ? config.presets.map(item=><option key={item.name}>{item.name}</option>):<option value="">Carregando…</option>}</select></label>
        <label>Ordenar resultados<select value={config ? sortKey(sort):""} disabled={!config} onChange={event=>{const [field,direction]=event.target.value.split(":");setSort({field,direction:direction as Sort["direction"]});}}>{config ? sortOptions.map(option=><option key={sortKey(option)} value={sortKey(option)}>{sortLabel(option)}</option>):<option value="">Carregando…</option>}</select></label>
      </div>
      {preset?.description && <p className="caption scr-preset-note">{preset.description}</p>}
      <div className="scr-criteria">
        {criterionDefs.map(def=>{
          const enabled=controls[def.enabled];
          return <fieldset className="scr-criterion" data-active={enabled} key={def.key}><legend>{def.group}{enabled && <span className="scr-state">Ativo</span>}</legend>
            <label><input type="checkbox" checked={enabled} onChange={event=>edit(def.enabled,event.target.checked)}/>{def.toggle}</label>
            {enabled && inputs[def.key]}
          </fieldset>;
        })}
      </div>
      <div className="actions scr-filters-actions">
        <button disabled={actionBusy} onClick={()=>setControls({...defaults})}>Redefinir para Explorar B3</button>
        <button disabled={actionBusy} onClick={()=>void sourceRefresh()}>{actionBusy ? "Atualizando…":"Atualizar dados Fundamentus"}</button>
        <span className="caption">Os resultados são atualizados automaticamente ao alterar os critérios.</span>
      </div>
    </section>
    <div className="scr-results" aria-busy={busy || actionBusy}>
      <div className="scr-active context-strip" tabIndex={-1} ref={barRef}>
        <p className="scr-chips-label" id="scr-active-label">Critérios ativos</p>
        {chips.length ? <ul className="scr-chips" aria-labelledby="scr-active-label">{chips.map(chip=><li key={chip.key}><span>{chip.label}</span><button type="button" aria-label={`Remover critério ${chip.label}`} onClick={()=>removeCriterion(chip.enabled)}><span aria-hidden="true">×</span></button></li>)}</ul>:<span className="caption">Nenhum critério ativo; todos os ativos são exibidos.</span>}
        <p className="scr-count" role="status">{result && !busy ? <><strong>{result.matched_count}</strong> de {result.total_count} ações correspondem</>:error ? "Resultado indisponível":"Atualizando resultado…"}</p>
      </div>
      {(busy || actionBusy || !config) && !error && <Loading>{actionBusy ? "Atualizando dados da Fundamentus…":"Carregando dados da B3…"}</Loading>}
      {error && <Notice tone="negative"><strong>Não foi possível exibir os resultados.</strong> {error}{!plInvalid && <button disabled={actionBusy} onClick={()=>setReload(value=>value+1)}>Tentar novamente</button>}</Notice>}
      {result && <Section title="Resultados" description="Ajuste as colunas, selecione um ativo ou exporte a lista completa."><div className={busy ? "scr-stale":undefined}>
        <Metrics items={[{label:"Ações correspondentes",value:result.matched_count,unit:"integer",help:`de ${result.total_count} ativos analisados`},{label:"DY médio",value:result.metrics.mean_dy,unit:"percent",digits:2},{label:"ROE médio",value:result.metrics.mean_roe,unit:"percent",digits:2}]}/>
        <div className="scr-context context-strip">
          <p className="caption">{result.fetched_at ? `Fundamentus coletado em ${new Date(result.fetched_at).toLocaleString("pt-BR")} · há ${result.age_minutes} min`:"Horário da coleta indisponível; a idade deste dado não pode ser confirmada."}</p>
        </div>
        {!result.matched_count ? <div className="empty-state-card"><span className="empty-state-icon"><Icon name="filter"/></span><h3 className="empty-state-title">Nenhuma ação encontrada</h3><div className="empty-state-message">Reduza os limites ou desative critérios. Dados ausentes não são considerados aprovação.</div><button className="primary" onClick={()=>setControls({...defaults})}>Redefinir para Explorar B3</button></div>:<>
          <details className="scr-columns"><summary>Colunas exibidas ({columnCount} de {result.available_columns.length})</summary><fieldset className="checkbox-group"><legend>Indicadores na tabela</legend><div className="checkbox-group-options">{result.available_columns.map(column=><label key={column.key}><input type="checkbox" checked={visible?.includes(column.key) ?? false} onChange={event=>setVisible(current=>{const next=new Set(current ?? []);if(event.target.checked)next.add(column.key);else next.delete(column.key);return result.available_columns.map(item=>item.key).filter(key=>next.has(key));})}/>{column.label}</label>)}</div></fieldset></details>
          <details className="scr-columns scr-glossary"><summary>Entenda os indicadores exibidos</summary>
            <dl>{result.available_columns.filter(column=>visible?.includes(column.key) && column.help).map(column=><div key={column.key}><dt>{column.label}</dt><dd>{column.help}</dd></div>)}</dl>
          </details>
          <div className="scr-selection" data-selected={selectedCurrent ? "true":"false"} role="region" aria-label="Ação selecionada" aria-live="polite">
            {selectedCurrent ? <>
              <strong>Ação selecionada: {selectedCurrent}</strong>
              <div className="actions"><button className="primary" disabled={actionBusy} onClick={()=>void handoff()}>Analisar em Fundamentos</button><button disabled={actionBusy} aria-pressed={session.watchlist.includes(selectedCurrent)} onClick={()=>void favorite()}>{session.watchlist.includes(selectedCurrent) ? "★ Favoritado":"☆ Favoritar"}</button><ExportButton path={`/api/screener/${result.id}/results.csv`}>Exportar todos os {result.matched_count} resultados (CSV)</ExportButton></div>
              <p className="caption">Abrir Fundamentos substitui a seleção atual; clique em Analisar para buscar indicadores, preço-alvo e consenso.</p>
            </>:<>
              <p className="caption">Selecione uma linha para abrir a análise de preço-alvo ou gerenciar favoritos. As colunas exibidas não alteram a seleção nem o CSV.</p>
              <div className="actions"><ExportButton path={`/api/screener/${result.id}/results.csv`}>Exportar todos os {result.matched_count} resultados (CSV)</ExportButton></div>
            </>}
          </div>
          <div className="scr-table">
            <ScreenerTable columns={result.available_columns.filter(column=>visible?.includes(column.key))} rows={result.rows.slice(0,limit)} sort={sort} onSort={sortBy} selectedKey={selectedCurrent ?? undefined} onSelect={setSelected}/>
          </div>
          <p className="caption" role="status">Exibindo {Math.min(limit,result.rows.length)} de {result.rows.length} ações.</p>
          {result.rows.length>limit && <button className="scr-more" onClick={()=>setLimit(previous=>previous+25)}>Ver mais {Math.min(25,result.rows.length-limit)} ações ({result.rows.length-limit} restantes)</button>}
        </>}
      </div></Section>}
    </div>
    </div>
    <hr/><p className="caption">Fonte: Fundamentus (fundamentus.com.br). Dados com atraso. Esta triagem não é recomendação de investimento; verifique os indicadores e faça sua própria análise.</p>
  </div>;
}
