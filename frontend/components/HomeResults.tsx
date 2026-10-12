"use client";
import Link from "next/link";
import { useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import Chart from "./Chart";
import { ExportButton, formatValue, Metrics, Notice, Table, ValueText } from "./ui";
import { PriceRange, TargetRange } from "./RangeBar";
import type { Column } from "./ui";
import { comparisonColors, consensusColumns, marketColumns } from "../lib/home";
import type { Company, HomeResult } from "../lib/home";

type Props = {
  ready: HomeResult;
  selection: string[];
  watchlist: string[];
  busy: boolean;
  onFavorite: (ticker: string) => void;
  onRefreshConsensus: () => void;
};

type TabContext = {
  ready: HomeResult;
  companies: Company[];
  single: boolean;
  busy: boolean;
  comparison: string;
  setComparison: (key: string) => void;
  onRefreshConsensus: () => void;
};

type ResultTab = {
  id: string;
  label: string;
  available: (ready: HomeResult, single: boolean) => boolean;
  render: (context: TabContext) => ReactNode;
};

const always = () => true;

const resultTabs: ResultTab[] = [
  {id:"visao-geral",label:"Visão geral",available:always,render:({ready,companies,single})=><>
    <section className="analysis-section home-panel-section" aria-labelledby="home-mercado-title"><h3 id="home-mercado-title">Informações de Mercado</h3>
      {single ? companies.map(company=>company.available ? <div key={company.ticker}><Metrics items={marketColumns.filter(column=>column.unit).map(column=>({label:column.label,value:company.market[column.key as keyof typeof company.market],unit:column.unit,digits:column.digits}))}/><p>Última cotação registrada: {company.market.last_quote_date ?? "—"} para {company.ticker}</p></div>:<Notice key={company.ticker}>Sem dados de mercado para {company.ticker}.</Notice>):<Table columns={marketColumns} rows={companies.map(company=>({ticker:company.ticker,...company.market}))} rowKey="ticker"/>}
    </section>
    <section className="analysis-section home-panel-section" aria-labelledby="home-indicadores-title"><h3 id="home-indicadores-title">Indicadores Financeiros</h3>
      {single ? companies.map(company=>company.indicators_available ? <div key={company.ticker}>{Object.entries(company.metric_groups).map(([group,metrics])=><div className="home-group" key={group}><h4>{group}</h4><Metrics items={metrics}/><div className="actions">{metrics.filter(metric=>metric.classification).map(metric=><span key={metric.label} className={`home-signal signal-${metric.status}`}>{metric.label}: {metric.classification}</span>)}</div></div>)}</div>:<Notice key={company.ticker}>Sem indicadores para {company.ticker}.</Notice>):<><p>Rentabilidade, margens e crescimento em % · múltiplos em × · dados ausentes: —.</p><Table columns={[{key:"ticker",label:"Ticker"},...ready.comparison_indicators.map(item=>({key:item.key,label:item.label,unit:item.unit,digits:2} as Column))]} rows={companies.map(company=>({ticker:company.ticker,...company.indicators}))} rowKey="ticker"/></>}
    </section>
  </>},
  {id:"saude",label:"Saúde financeira",available:always,render:({ready,companies})=><>
    <p>{ready.health_summary}</p>
    {companies.map(company=><div className="home-company-block" key={company.ticker}><h3>{company.ticker}{company.identity.company ? <> · {company.identity.company}</>:null}</h3>{company.health.metrics.length ? <><Metrics items={company.health.metrics}/>{company.health.diagnostics.map(diag=><Notice key={diag.text} tone={diag.status}>{diag.text}</Notice>)}</>:<Notice>Dados de endividamento não disponíveis via Fundamentus para este ticker.</Notice>}</div>)}
  </>},
  {id:"comparacao",label:"Comparação",available:(_,single)=>!single,render:({ready,companies,comparison,setComparison})=>{
    const choice=ready.comparison_indicators.find(item=>item.key===comparison);
    const chartCompanies=companies.filter(company=>company.indicators_available);
    return <>
      <label>Indicador comparativo<select value={comparison} onChange={event=>setComparison(event.target.value)}>{ready.comparison_indicators.map(item=><option value={item.key} key={item.key}>{item.label}</option>)}</select></label>
      <Chart figure={{data:[{type:"bar",x:chartCompanies.map(company=>company.ticker),y:chartCompanies.map(company=>company.indicators[comparison]),text:chartCompanies.map(company=>{const value=company.indicators[comparison];return value===null ? "—":`${value.toFixed(2)}${choice?.unit==="percent" ? "%":""}`;}),textposition:"outside",cliponaxis:false,marker:{color:chartCompanies.map((_,index)=>comparisonColors[index%comparisonColors.length])}}],layout:{height:350,title:{text:`Comparativo de ${choice?.label}`},yaxis:{title:{text:choice?.unit==="percent" ? `${choice.label} (%)`:choice?.label}}}}}/>
    </>;
  }},
  {id:"preco-alvo",label:"Preço-alvo",available:always,render:({ready,busy,onRefreshConsensus})=><>
    <div className="actions"><button type="button" disabled={busy} onClick={onRefreshConsensus}>Atualizar consensos</button></div>
    {ready.consensus.has_targets && <ul className="target-ladder" aria-label="Faixas de preço-alvo por ativo">{ready.consensus.rows.map(row=>{
      const upside=typeof row.upside==="number" && Number.isFinite(row.upside) ? row.upside:null;
      const tone=upside===null ? "neutral":upside>=0 ? "positive":"negative";
      return <li key={String(row.ticker)}>
        <span className="ticker-badge">{String(row.ticker)}</span>
        <TargetRange row={row}/>
        <span className={`delta-chip is-${tone}`}>{upside===null ? "—":<><span aria-hidden="true">{upside>=0 ? "▲":"▼"}</span>{upside>0 ? "+":""}{formatValue(upside,"percent",1)}</>}</span>
        <span className="target-ladder-meta">{typeof row.analyst_count==="number" ? `${row.analyst_count} ${row.analyst_count===1 ? "opinião":"opiniões"}`:"Opiniões —"}{typeof row.recommendation==="string" && row.recommendation ? ` · ${row.recommendation}`:""}</span>
      </li>;
    })}</ul>}
    {ready.consensus.has_targets && <p className="caption target-ladder-legend"><span className="legend-band" aria-hidden="true"/>Faixa entre alvo mínimo e máximo <span className="legend-mean" aria-hidden="true"/>Alvo médio <span className="legend-price" aria-hidden="true"/>Cotação de referência do Yahoo · potencial sobre o alvo médio</p>}
    <Table columns={consensusColumns} rows={ready.consensus.rows} rowKey="ticker"/>
    {ready.consensus.failed_tickers.length ? <Notice tone="warning">Consenso indisponível para: {ready.consensus.failed_tickers.join(", ")}.</Notice>:!ready.consensus.has_targets && <Notice>O Yahoo Finance não informa preços-alvo para as ações selecionadas.</Notice>}
    <p>Fonte: Yahoo Finance. Potencial calculado sobre o alvo médio; opiniões e recomendações são agregadas pela fonte e não constituem recomendação de investimento.</p>
  </>},
  {id:"sintese",label:"Síntese",available:always,render:({companies})=><>{companies.map(company=><article className="analyst-synthesis-card" key={company.ticker}>
    <h3>{company.ticker} · {company.identity.company ?? "—"}</h3><span className="analyst-synthesis-sector">{company.synthesis.sector ?? "Setor indisponível"}</span>
    <p>{company.synthesis.favorable_categories}/{company.synthesis.categorias_validas} áreas favoráveis · {company.synthesis.indicadores_validos} indicadores elegíveis</p>
    {company.synthesis.categorias_validas<2 && <Notice>Dados insuficientes para uma conclusão setorial: são necessárias pelo menos duas áreas com indicadores e no mínimo três observações.</Notice>}
    <div className="mcard-grid">{Object.entries(company.synthesis.categorias).map(([name,category])=><div className={`analyst-category ${category.veredicto==="Favorável" ? "is-positive":category.veredicto==="Desfavorável" ? "is-negative":"is-neutral"}`} key={name}><h4>{name==="Yield" ? "Dividendos":name}</h4><span className="analyst-category-status">{category.veredicto}</span><div className="analyst-synthesis-meter" role="meter" aria-label={`Percentil de ${name==="Yield" ? "Dividendos":name}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={category.percentil}><span style={{width:`${category.percentil}%`}}/></div><p>Percentil {category.percentil.toFixed(1)} · {category.indicadores_validos} indicadores</p></div>)}</div>
    {["Favorável","Neutro","Desfavorável"].map(verdict=>{const details=company.synthesis.details.filter(detail=>detail.verdict===verdict);return details.length ? <details className="analyst-synthesis-details" key={verdict}><summary>{verdict} · {details.length} indicadores</summary><Table columns={[{key:"label",label:"Indicador"},{key:"value",label:"Valor"},{key:"unit",label:"Unidade"},{key:"median",label:"Mediana do setor"},{key:"mean",label:"Média aparada do setor"},{key:"percentile",label:"Percentil",unit:"number",digits:1},{key:"observations",label:"Observações",unit:"integer"}]} rows={details.map(detail=>({...detail,unit:detail.unit==="percent" ? "%":"×"}))} rowKey="label"/><p>Percentil é a proporção de empresas do setor superadas, sem contar a própria empresa; empates contam pela metade. Cada empresa entra uma vez, pela classe de ação mais líquida. A média do setor é aparada em 10% em cada ponta.</p></details>:null;})}
    <p>Fonte: Fundamentus · comparação setorial. Áreas e percentis não constituem recomendação de investimento.</p>
  </article>)}</>},
];

const tabKeys = ["ArrowLeft","ArrowRight","Home","End"];
const summaryIndicatorKeys = ["pl","roe","dy"];

export default function HomeResults({ ready, selection, watchlist, busy, onFavorite, onRefreshConsensus }: Props) {
  const [comparison,setComparison] = useState("pl");
  const [selectedTab,setSelectedTab] = useState(resultTabs[0].id);
  const single=selection.length===1;
  const companies=ready.companies;
  const summaryIndicators=ready.comparison_indicators.filter(item=>summaryIndicatorKeys.includes(item.key));
  const tabs=resultTabs.filter(tab=>tab.available(ready,single));
  const activeTab=tabs.find(tab=>tab.id===selectedTab) ?? tabs[0];
  function moveFocus(event:KeyboardEvent<HTMLButtonElement>, index:number) {
    if(!tabKeys.includes(event.key))return;
    event.preventDefault();
    const next=event.key==="Home" ? 0:event.key==="End" ? tabs.length-1:(index+(event.key==="ArrowRight" ? 1:-1)+tabs.length)%tabs.length;
    setSelectedTab(tabs[next].id);
    document.getElementById(`home-tab-${tabs[next].id}`)?.focus();
  }
  return <>
      {!!ready.missing_tickers.length && <Notice tone="warning">Dados Fundamentus indisponíveis para {ready.missing_tickers.join(", ")}. Os dados disponíveis foram preservados e os caches da fonte invalidados.</Notice>}
      <div className="home-results-bar result-lead section-toolbar">
        <div className="home-results-heading"><h2>Resultado da análise</h2><p className="caption">{companies.length} {companies.length===1 ? "ativo analisado":"ativos analisados"}: {ready.requested_tickers.join(", ")}</p></div>
        <div className="home-results-actions section-actions"><ExportButton path={`/api/home/${ready.id}/fundamentals.csv`}>Exportar CSV bruto</ExportButton></div>
      </div>
      <ul className="home-summary" aria-label="Resumo dos ativos analisados">{companies.map(company=>{const saved=watchlist.includes(company.ticker);const favorable=company.synthesis.favorable_categories;const valid=company.synthesis.categorias_validas;return <li className="home-summary-card" key={company.ticker}>
        <div className="home-summary-head"><strong className="home-summary-ticker ticker-badge">{company.ticker}</strong><button type="button" className="home-favorite-toggle" aria-pressed={saved} aria-label={`Favoritar ${company.ticker}`} title={saved ? `${company.ticker} está nos favoritos`:`Adicionar ${company.ticker} aos favoritos`} onClick={()=>onFavorite(company.ticker)}><svg width="20" height="20" viewBox="0 0 24 24" fill={saved ? "currentColor":"none"} stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" aria-hidden="true" focusable="false"><path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z"/></svg></button></div>
        <p className="home-summary-company">{company.identity.company ?? "Empresa não informada"}</p>
        <p className="home-summary-sector">{[company.identity.sector ?? "Setor indisponível",company.identity.subsector].filter(Boolean).join(" · ")}</p>
        <dl className="home-summary-facts">
          <div className="home-summary-quote"><dt>Cotação</dt><dd><ValueText value={company.market.price} unit="money" currency={company.market.currency}/></dd></div>
          {summaryIndicators.map(item=><div className="home-summary-indicator" key={item.key}><dt>{item.label}</dt><dd><ValueText value={company.indicators[item.key]} unit={item.unit} digits={2}/></dd></div>)}
          <div className="home-summary-row home-summary-synthesis"><dt>Síntese setorial</dt><dd>{valid ? <><span className="home-summary-meter" aria-hidden="true">{Array.from({length:valid},(_,index)=><span key={index} data-on={index<favorable}/>)}</span>{favorable}/{valid} áreas favoráveis</>:"Sem áreas avaliadas"}</dd></div>
        </dl>
        <PriceRange low={company.market.low_52w} high={company.market.high_52w} price={company.market.price} currency={company.market.currency} date={company.market.last_quote_date}/>
      </li>;})}</ul>
      <div className="home-tabs" role="tablist" aria-label="Seções da análise">{tabs.map((tab,index)=>{const selected=tab.id===activeTab.id;return <button type="button" role="tab" key={tab.id} id={`home-tab-${tab.id}`} aria-controls={`home-panel-${tab.id}`} aria-selected={selected} tabIndex={selected ? 0:-1} onClick={()=>setSelectedTab(tab.id)} onKeyDown={event=>moveFocus(event,index)}>{tab.label}</button>;})}</div>
      {tabs.map(tab=><div className="home-tab-panel" role="tabpanel" key={tab.id} id={`home-panel-${tab.id}`} aria-labelledby={`home-tab-${tab.id}`} tabIndex={0} hidden={tab.id!==activeTab.id}>
        {tab.id===activeTab.id && tab.render({ready,companies,single,busy,comparison,setComparison,onRefreshConsensus})}
      </div>)}
      <div className="next-step-card"><div className="next-step-eyebrow">Próximo passo</div><h2>Analise sua carteira</h2><p>Monte e otimize a carteira com {selection.slice(0,3).join(", ")}{selection.length>3 ? "…":""}.</p><Link className="button-link" href="/portfolio">Abrir Portfólio</Link></div>
  </>;
}
