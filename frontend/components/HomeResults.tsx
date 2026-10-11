"use client";
import Link from "next/link";
import { useState } from "react";
import Chart from "./Chart";
import { ExportButton, formatValue, Metrics, Notice, Section, Table } from "./ui";
import type { Column } from "./ui";
import { consensusColumns, marketColumns } from "../lib/home";
import type { HomeResult } from "../lib/home";

type Props = {
  ready: HomeResult;
  selection: string[];
  watchlist: string[];
  busy: boolean;
  onFavorite: (ticker: string) => void;
  onRefreshConsensus: () => void;
};

export default function HomeResults({ ready, selection, watchlist, busy, onFavorite, onRefreshConsensus }: Props) {
  const [comparison,setComparison] = useState("pl");
  const single=selection.length===1;
  const companies=ready.companies;
  const choice=ready.comparison_indicators.find(item=>item.key===comparison);
  const chartCompanies=companies.filter(company=>company.indicators_available);
  return <>
      {!!ready.missing_tickers.length && <Notice tone="warning">Dados Fundamentus indisponíveis para {ready.missing_tickers.join(", ")}. Os dados disponíveis foram preservados e os caches da fonte invalidados.</Notice>}
      <div className="home-results-bar">
        <div className="home-results-heading"><h2>Resultado da análise</h2><p className="caption">{companies.length} {companies.length===1 ? "ativo analisado":"ativos analisados"}: {ready.requested_tickers.join(", ")}</p></div>
        <div className="home-results-actions"><ExportButton path={`/api/home/${ready.id}/fundamentals.csv`}>Exportar CSV bruto</ExportButton></div>
      </div>
      <ul className="home-summary" aria-label="Resumo dos ativos analisados">{companies.map(company=>{const saved=watchlist.includes(company.ticker);return <li className="home-summary-card" key={company.ticker}>
        <div className="home-summary-head"><strong className="home-summary-ticker">{company.ticker}</strong><button aria-pressed={saved} aria-label={`Favoritar ${company.ticker}`} onClick={()=>onFavorite(company.ticker)}>{saved ? "★ Favoritado":"☆ Favoritar"}</button></div>
        <p className="home-summary-company">{company.identity.company ?? "Empresa não informada"}</p>
        <p className="home-summary-sector">{company.identity.sector ?? "Setor indisponível"}</p>
        <dl className="home-summary-facts">
          <div><dt>Cotação</dt><dd>{formatValue(company.market.price,"money",2,company.market.currency)}</dd></div>
          <div><dt>Síntese setorial</dt><dd>{company.synthesis.categorias_validas ? `${company.synthesis.favorable_categories}/${company.synthesis.categorias_validas} áreas favoráveis`:"Sem áreas avaliadas"}</dd></div>
        </dl>
      </li>;})}</ul>
        <div className="home-sections">
          <div id="home-mercado" className="home-anchor"><Section title="Informações de Mercado">
            {single ? companies.map(company=>company.available ? <div key={company.ticker}><Metrics items={marketColumns.filter(column=>column.unit).map(column=>({label:column.label,value:company.market[column.key as keyof typeof company.market] ?? 0,unit:column.unit,digits:column.digits}))}/><p>Última cotação registrada: {company.market.last_quote_date ?? "—"} para {company.ticker}</p></div>:<Notice key={company.ticker}>Sem dados de mercado para {company.ticker}.</Notice>):<Table columns={marketColumns} rows={companies.map(company=>({ticker:company.ticker,...company.market}))} rowKey="ticker"/>}
          </Section></div>
          <div id="home-indicadores" className="home-anchor"><Section title="Indicadores Financeiros">
            {single ? companies.map(company=>company.indicators_available ? <div key={company.ticker}>{Object.entries(company.metric_groups).map(([group,metrics])=><div className="home-group" key={group}><h3>{group}</h3><Metrics items={metrics.map(metric=>({...metric,value:metric.value ?? (metric.label==="EV/EBITDA" ? null:0)}))}/><div className="actions">{metrics.filter(metric=>metric.classification).map(metric=><span key={metric.label} className={`home-signal signal-${metric.status}`}>{metric.label}: {metric.classification}</span>)}</div></div>)}</div>:<Notice key={company.ticker}>Sem indicadores para {company.ticker}.</Notice>):<><p>Rentabilidade, margens e crescimento em % · múltiplos em × · dados ausentes: —.</p><Table columns={[{key:"ticker",label:"Ticker"},...ready.comparison_indicators.map(item=>({key:item.key,label:item.label,unit:item.unit,digits:2} as Column))]} rows={companies.map(company=>({ticker:company.ticker,...company.indicators}))} rowKey="ticker"/></>}
          </Section></div>
          <div id="home-saude" className="home-anchor"><Section title="Saúde Financeira"><p>Endividamento e liquidez da empresa. Dívida/PL acima de 3× e liquidez abaixo de 1× são sinais de alerta.</p>{companies.map(company=><div className="home-company-block" key={company.ticker}><h3>{company.ticker}{company.identity.company ? <> · {company.identity.company}</>:null}</h3>{company.health.metrics.length ? <><Metrics items={company.health.metrics}/>{company.health.diagnostics.map(diag=><Notice key={diag.text} tone={diag.status}>{diag.text}</Notice>)}</>:<Notice>Dados de endividamento não disponíveis via Fundamentus para este ticker.</Notice>}</div>)}</Section></div>
          {!single && <div id="home-grafico" className="home-anchor"><Section title="Comparação Gráfica de Múltiplos"><label>Indicador comparativo<select value={comparison} onChange={event=>setComparison(event.target.value)}>{ready.comparison_indicators.map(item=><option value={item.key} key={item.key}>{item.label}</option>)}</select></label><Chart figure={{data:[{type:"bar",x:chartCompanies.map(company=>company.ticker),y:chartCompanies.map(company=>company.indicators[comparison]),text:chartCompanies.map(company=>{const value=company.indicators[comparison];return value===null ? "N/D":!company.indicator_available[comparison] ? "N/A":`${value.toFixed(2)}${choice?.unit==="percent" ? "%":""}`;}),textposition:"outside",cliponaxis:false,marker:{color:chartCompanies.map((_,index)=>["#8b93ff","#4fd1c5","#f6ad55","#f472b6","#63b3ed","#a78bfa","#68d391","#fbd38d","#fc8181","#7dd3fc"][index%10])}}],layout:{height:350,title:{text:`Comparativo de ${choice?.label}`},yaxis:{title:{text:choice?.unit==="percent" ? `${choice.label} (%)`:choice?.label}}}}}/></Section></div>}
          <div id="home-consenso" className="home-anchor"><Section title="Preço-alvo e consenso"><button disabled={busy} onClick={onRefreshConsensus}>Atualizar consensos</button><Table columns={consensusColumns} rows={ready.consensus.rows} rowKey="ticker"/>{ready.consensus.failed_tickers.length ? <Notice tone="warning">Consenso indisponível para: {ready.consensus.failed_tickers.join(", ")}.</Notice>:!ready.consensus.has_targets && <Notice>O Yahoo Finance não informa preços-alvo para as ações selecionadas.</Notice>}<p>Fonte: Yahoo Finance. Potencial calculado sobre o alvo médio; opiniões e recomendações são agregadas pela fonte e não constituem recomendação de investimento.</p></Section></div>
          <div id="home-sintese" className="home-anchor"><Section title="Síntese do Analista">{companies.map(company=><article className="analyst-synthesis-card" key={company.ticker}>
            <h3>{company.ticker} · {company.identity.company ?? "—"}</h3><span className="analyst-synthesis-sector">{company.synthesis.sector ?? "Setor indisponível"}</span>
            <p>{company.synthesis.favorable_categories}/{company.synthesis.categorias_validas} áreas favoráveis · {company.synthesis.indicadores_validos} indicadores elegíveis</p>
            {company.synthesis.categorias_validas<2 && <Notice>Dados insuficientes para uma conclusão setorial: são necessárias pelo menos duas áreas com indicadores e no mínimo três observações.</Notice>}
            <div className="mcard-grid">{Object.entries(company.synthesis.categorias).map(([name,category])=><div className={`analyst-category ${category.veredicto==="Favorável" ? "is-positive":category.veredicto==="Desfavorável" ? "is-negative":"is-neutral"}`} key={name}><h4>{name==="Yield" ? "Dividendos":name}</h4><span className="analyst-category-status">{category.veredicto}</span><div className="analyst-synthesis-meter" role="meter" aria-label={`Percentil de ${name==="Yield" ? "Dividendos":name}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={category.percentil}><span style={{width:`${category.percentil}%`}}/></div><p>Percentil {category.percentil.toFixed(1)} · {category.indicadores_validos} indicadores</p></div>)}</div>
            {["Favorável","Neutro","Desfavorável"].map(verdict=>{const details=company.synthesis.details.filter(detail=>detail.verdict===verdict);return details.length ? <details className="analyst-synthesis-details" key={verdict}><summary>{verdict} · {details.length} indicadores</summary><Table columns={[{key:"label",label:"Indicador"},{key:"value",label:"Valor"},{key:"unit",label:"Unidade"},{key:"median",label:"Mediana do setor"},{key:"mean",label:"Média do setor"},{key:"percentile",label:"Percentil",unit:"number",digits:1},{key:"observations",label:"Observações",unit:"integer"}]} rows={details.map(detail=>({...detail,unit:detail.unit==="percent" ? "%":"×"}))} rowKey="label"/><p>Percentil é a proporção de observações estritamente superadas; inclui o próprio ticker. Empates não contam como superados.</p></details>:null;})}
            <p>Fonte: Fundamentus · comparação setorial. Áreas e percentis não constituem recomendação de investimento.</p>
          </article>)}</Section></div>
        </div>
      <div className="next-step-card"><div className="next-step-eyebrow">Próximo passo</div><h2>Analise sua carteira</h2><p>Monte e otimize a carteira com {selection.slice(0,3).join(", ")}{selection.length>3 ? "…":""}.</p><Link href="/portfolio">Abrir Portfólio</Link></div>
  </>;
}
