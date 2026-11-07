"use client";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import Link from "next/link";
import "./screener.css";
import { MAX_SELECTION, SELECTION_FULL_MESSAGE, useApp } from "../../components/AppProvider";
import { ExportButton, formatCompactValue, formatValue, Idle, Loading, Notice, Pane } from "../../components/ui";
import { refreshFundamentus } from "../../lib/home";
import { activeCriteria, applyPreset, canonicalCriteria, criterionDefs, defaults, editControl, queryScreener, screenerConfig, sortKey } from "../../lib/screener";
import type { Config, Controls, CriterionDef, CriterionKey, Result, Sort } from "../../lib/screener";
import ScreenerTable from "./ScreenerTable";

const liquidityShortcuts = [1_000_000, 2_000_000, 5_000_000, 20_000_000];
const liquidityMax = 100_000_000;

export default function ScreenerPage() {
  const {session,stocks,loading,error:appError,setSelection,toggleFavorite,refreshSession} = useApp();
  const [config,setConfig]=useState<Config|null>(null);
  const [controls,setControls]=useState<Controls>(defaults);
  const [sort,setSort]=useState<Sort>({field:"liq2m",direction:"desc"});
  const [result,setResult]=useState<Result|null>(null);
  const [visible,setVisible]=useState<string[]|null>(null);
  const [selected,setSelected]=useState<string|null>(null);
  const [busy,setBusy]=useState(false);
  const [actionBusy,setActionBusy]=useState(false);
  const [error,setError]=useState<string|null>(null);
  const [notice,setNotice]=useState<string|null>(null);
  const [undoSelection,setUndoSelection]=useState<{before:string[];ticker:string;mode:"add"|"only"}|null>(null);
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
    if(controls.liq2m_enabled && (!Number.isFinite(controls.liq2m_min) || controls.liq2m_min<0 || controls.liq2m_min>liquidityMax)) {setResult(null);setError("A liquidez mínima deve estar entre R$ 0 e R$ 100 milhões.");setBusy(false);return;}
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
  async function handoff(mode: "add" | "only", ticker = selected) {
    if(!ticker)return;
    setActionBusy(true);setError(null);setUndoSelection(null);
    try {
      const known=stocks.some(stock=>stock.ticker===ticker);
      if(!known){setError(`${ticker} está fora da lista local B3 e não pode entrar na seleção.`);return;}
      const before=[...session!.selection];
      if(mode==="add" && before.includes(ticker)){setUndoSelection(null);setNotice(`${ticker} já está na seleção.`);return;}
      if(mode==="add" && before.length>=MAX_SELECTION){setError(SELECTION_FULL_MESSAGE);return;}
      await setSelection(mode==="only" ? [ticker]:current=>current.includes(ticker) ? [...current]:[...current,ticker]);
      setNotice(null);
      setUndoSelection({before,ticker,mode});
    } catch(exc){setError((exc as Error).message);}finally{setActionBusy(false);}
  }
  async function undoOnly() {
    if(!undoSelection)return;
    setActionBusy(true);setError(null);
    try {await setSelection(undoSelection.before);setUndoSelection(null);}
    catch(exc){setError((exc as Error).message);}finally{setActionBusy(false);}
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
  if(loading || !session)return <div className="desk">{appError ? <Notice tone="negative">{appError}<button onClick={()=>void refreshSession()}>Tentar novamente</button></Notice>:<Loading/>}</div>;
  const selectedCurrent=selected && result?.rows.some(row=>row.ticker===selected) ? selected:null;
  const preset=config?.presets.find(item=>item.name===controls.preset);
  const plInvalid=controls.pl_enabled && controls.pl_min>controls.pl_max;
  const liquidityInvalid=controls.liq2m_enabled && (!Number.isFinite(controls.liq2m_min) || controls.liq2m_min<0 || controls.liq2m_min>liquidityMax);
  const invalidCriteria=liquidityInvalid || (controls.pl_enabled && controls.pl_min>controls.pl_max) || Object.values(canonicalCriteria(controls)).some(value=>typeof value==="number" && !Number.isFinite(value));
  const liquidity=Number.isFinite(controls.liq2m_min) ? controls.liq2m_min.toLocaleString("pt-BR",{style:"currency",currency:"BRL",maximumFractionDigits:2}) : "Valor inválido";
  const compactLiquidity=Number.isFinite(controls.liq2m_min) ? formatCompactValue(controls.liq2m_min,"money",1) : "Valor inválido";
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
      <label htmlFor="scr-liquidity-range">Liquidez média em 2 meses<input id="scr-liquidity-range" type="range" min={0} max={liquidityMax} step="any" value={Number.isFinite(controls.liq2m_min) ? controls.liq2m_min:0} disabled={liquidityInvalid} aria-describedby={liquidityInvalid ? "scr-liquidity-error":undefined} aria-valuetext={liquidity} onChange={event=>edit("liq2m_min",event.currentTarget.valueAsNumber)}/></label>
      <label htmlFor="scr-liquidity-amount">Valor mínimo (R$)<input id="scr-liquidity-amount" type="number" inputMode="decimal" min={0} max={liquidityMax} step="any" value={Number.isNaN(controls.liq2m_min) ? "":controls.liq2m_min} aria-invalid={liquidityInvalid} aria-describedby={liquidityInvalid ? "scr-liquidity-error":undefined} onChange={event=>edit("liq2m_min",event.currentTarget.valueAsNumber)}/></label>
      <output className="scr-liquidity-readout" title={liquidity}>{compactLiquidity}</output>
      {liquidityInvalid && <p className="scr-field-error" id="scr-liquidity-error">O valor deve estar entre R$ 0 e R$ 100 milhões para consultar o screener.</p>}
      <div className="scr-quick-amounts" role="group" aria-label="Valores rápidos de liquidez">{liquidityShortcuts.map(amount=><button type="button" key={amount} aria-pressed={controls.liq2m_min===amount} onClick={()=>edit("liq2m_min",amount)}>R$ {amount/1_000_000} mi</button>)}</div>
    </>,
  };
  const chips=activeCriteria(controls);
  const presetSortKeys=new Set((config?.sorts ?? []).map(sortKey));
  const sortOptions:Sort[]=config ? [...config.sorts,...(presetSortKeys.has(sortKey(sort)) ? []:[sort])]:[];
  const columnCount=visible?.length ?? 0;
  const selectionSet=new Set(session.selection);
  return <div className="desk desk-split desk-screener">
    <Pane title="Filtros" className="pane-params" narrowCollapsed meta={chips.length ? `${chips.length} ${chips.length===1 ? "critério ativo":"critérios ativos"}` : "nenhum critério"}>
      <label>Ponto de partida<select value={config ? controls.preset:""} disabled={!config} onChange={event=>{const next=config?.presets.find(item=>item.name===event.target.value);if(next)setControls(current=>applyPreset(current,next.name,next.criteria));}}>{config ? config.presets.map(item=><option key={item.name}>{item.name}</option>):<option value="">Carregando…</option>}</select></label>
      {preset?.description && <p className="param-hint">{preset.description}</p>}
      <label>Ordenar resultados<select value={config ? sortKey(sort):""} disabled={!config} onChange={event=>{const [field,direction]=event.target.value.split(":");setSort({field,direction:direction as Sort["direction"]});}}>{config ? sortOptions.map(option=><option key={sortKey(option)} value={sortKey(option)}>{sortLabel(option)}</option>):<option value="">Carregando…</option>}</select></label>
      <h3 className="pane-sub">Critérios</h3>
      <ul className="criteria">{criterionDefs.map(def=>{
        const enabled=controls[def.enabled];
        return <li key={def.key} data-active={enabled}>
          <label><input type="checkbox" checked={enabled} onChange={event=>edit(def.enabled,event.target.checked)}/><span>{def.group}</span>{enabled && <small>{def.chip(controls)}</small>}<span className="visually-hidden">{def.toggle}</span></label>
          {enabled && <div className="criteria-fields">{inputs[def.key]}</div>}
        </li>;
      })}</ul>
      <div className="param-run">
        <button disabled={actionBusy} onClick={()=>setControls({...defaults})}>Redefinir para Explorar B3</button>
        <button disabled={actionBusy} title="Limpa o cache Fundamentus e busca os dados de novo" onClick={()=>void sourceRefresh()}>{actionBusy ? "Atualizando…":"Atualizar dados Fundamentus"}</button>
      </div>
      {result && <>
        <details className="disclosure"><summary>Colunas ({columnCount} de {result.available_columns.length})</summary><fieldset className="column-picker"><legend className="visually-hidden">Indicadores na tabela</legend>{result.available_columns.map(column=><label key={column.key}><input type="checkbox" checked={visible?.includes(column.key) ?? false} onChange={event=>setVisible(current=>{const next=new Set(current ?? []);if(event.target.checked)next.add(column.key);else next.delete(column.key);return result.available_columns.map(item=>item.key).filter(key=>next.has(key));})}/>{column.label}</label>)}</fieldset></details>
        <details className="disclosure"><summary>Entenda os indicadores</summary><dl className="glossary">{result.available_columns.filter(column=>visible?.includes(column.key) && column.help).map(column=><div key={column.key}><dt>{column.label}</dt><dd>{column.help}</dd></div>)}</dl></details>
      </>}
      <p className="param-hint">Resultados atualizam ao alterar os critérios. Fonte: Fundamentus, com atraso. Triagem, não recomendação de investimento.</p>
    </Pane>
    <Pane title="Resultados" className="pane-screener" flush
      meta={result && !busy && !invalidCriteria ? <span role="status"><b className="scr-count">{result.matched_count}</b> de {result.total_count} · DY médio {formatValue(result.metrics.mean_dy,"percent",2)} · ROE médio {formatValue(result.metrics.mean_roe,"percent",2)}{result.fetched_at ? ` · coleta há ${result.age_minutes} min`:" · horário da coleta indisponível"}</span> : <span role="status">{error || invalidCriteria ? "Resultado indisponível":"Atualizando…"}</span>}
      actions={result ? <ExportButton path={`/api/screener/${result.id}/results.csv`}>CSV {result.matched_count}</ExportButton> : undefined}>
      <div className="scr-active" tabIndex={-1} ref={barRef} aria-label="Critérios ativos">
        {chips.length ? <ul className="scr-chips">{chips.map(chip=><li key={chip.key} className="chip"><span title={chip.title}>{chip.label}</span><button type="button" aria-label={`Remover critério ${chip.label}`} onClick={()=>removeCriterion(chip.enabled)}>×</button></li>)}</ul>:<span className="muted">Nenhum critério ativo; todos os ativos são exibidos.</span>}
      </div>
      {notice && <Notice tone="positive">{notice}</Notice>}
      {undoSelection && <Notice tone="positive"><span className="scr-undo">{undoSelection.mode==="only" ? `Seleção substituída por ${undoSelection.ticker} (eram ${undoSelection.before.length} ativos).` : `${undoSelection.ticker} adicionada à seleção (${undoSelection.before.length+1}/${MAX_SELECTION}).`}</span><button disabled={actionBusy} onClick={()=>void undoOnly()}>Desfazer</button><Link className="button-link" href="/"><kbd>F1</kbd> Analisar no Início</Link></Notice>}
      {(busy || actionBusy || !config) && !error && <Loading>{actionBusy ? "Atualizando dados da Fundamentus…":"Carregando dados da B3…"}</Loading>}
      {error && <Notice tone="negative"><strong>Não foi possível exibir os resultados.</strong> {error}{!invalidCriteria && <button disabled={actionBusy} onClick={()=>setReload(value=>value+1)}>Tentar novamente</button>}</Notice>}
      {result && !invalidCriteria && (!result.matched_count ? <Idle action={<button className="primary" onClick={()=>setControls({...defaults})}>Redefinir para Explorar B3</button>}>Nenhuma ação encontrada. Reduza os limites ou desative critérios; dados ausentes não contam como aprovação.</Idle> : <>
        <div className="scr-actionbar" data-selected={selectedCurrent ? "true":"false"} role="region" aria-label="Ação selecionada" aria-live="polite">
          {selectedCurrent ? <>
            <b>{selectedCurrent}</b><span className="muted ell">{stocks.find(stock=>stock.ticker===selectedCurrent)?.company ?? ""}</span>
            <button className="primary" disabled={actionBusy || selectionSet.has(selectedCurrent)} onClick={()=>void handoff("add")}>{selectionSet.has(selectedCurrent) ? "Já na seleção":"Adicionar à seleção"}</button>
            <button disabled={actionBusy} onClick={()=>void handoff("only")}>Ver só este</button>
            <button disabled={actionBusy} aria-pressed={session.watchlist.includes(selectedCurrent)} onClick={()=>void favorite()}>{session.watchlist.includes(selectedCurrent) ? "★ Favorito":"☆ Favoritar"}</button>
          </> : <span className="muted">Clique em uma linha para adicionar à seleção · ↑↓ percorre · Enter adiciona · ✓ já está na seleção</span>}
        </div>
        <div className={`scr-scroll${busy ? " is-stale":""}`}>
          <ScreenerTable columns={result.available_columns.filter(column=>visible?.includes(column.key))} rows={result.rows} sort={sort} onSort={sortBy} selectedKey={selectedCurrent ?? undefined} onSelect={setSelected} onActivate={ticker=>{setSelected(ticker);void handoff("add",ticker);}} inSelection={selectionSet}/>
        </div>
        <p className="pane-foot">{result.rows.length} ações · {result.fetched_at ? `Fundamentus coletado em ${new Date(result.fetched_at).toLocaleString("pt-BR")}`:"horário da coleta indisponível; a idade do dado não pode ser confirmada"}</p>
      </>)}
    </Pane>
  </div>;
}
