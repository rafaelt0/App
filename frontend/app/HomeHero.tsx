"use client";
import Link from "next/link";
import type { Stock } from "../components/AppProvider";
import { Icon } from "../components/ui";

const visibleSectors = 12;
const mapShades = ["var(--indigo-300)","var(--indigo-400)","var(--indigo-500)","var(--indigo-600)","var(--indigo-700)","var(--indigo-800)","var(--indigo-900)"];
const tools = [
  {href:"/portfolio",label:"Portfólio",icon:"portfolio",text:"Monte uma carteira e compare risco, retorno e pesos."},
  {href:"/simulacao",label:"Simulação",icon:"simulation",text:"Projete faixas de retorno e risco a partir da carteira."},
  {href:"/noticias",label:"Notícias",icon:"news",text:"Acompanhe eventos recentes e o tom das notícias."},
  {href:"/screener",label:"Screener",icon:"filter",text:"Filtre ações listadas por critérios fundamentalistas."},
];

/** Compact landing identity shown while nothing is selected, so the search stays above the fold. */
export function HomeHero() {
  return <header className="page-hero home-hero">
    <div className="home-hero-kicker"><span className="page-hero-icon"><Icon name="home"/></span><span>Fundamentos</span></div>
    <h1 className="page-hero-title">Os fundamentos da B3, lado a lado.</h1>
    <p className="page-hero-subtitle">Compare indicadores, saúde financeira, preço-alvo e síntese setorial das ações da B3.</p>
  </header>;
}

/** Sector composition of the local B3 list; each sector selects its most liquid stocks. */
export function SectorMap({stocks,busy,onSector}:{stocks:Stock[];busy:boolean;onSector:(sector:string)=>void}) {
  const counts=new Map<string,number>();
  for(const stock of stocks)if(stock.sector)counts.set(stock.sector,(counts.get(stock.sector) ?? 0)+1);
  const ranked=[...counts].sort((a,b)=>b[1]-a[1] || a[0].localeCompare(b[0],"pt-BR"));
  const top=ranked.slice(0,visibleSectors);
  const remaining=ranked.slice(visibleSectors);
  if(!ranked.length)return null;
  const number=new Intl.NumberFormat("pt-BR");
  const percent=new Intl.NumberFormat("pt-BR",{style:"percent",minimumFractionDigits:1,maximumFractionDigits:1});
  const sectorRow=([sector,count]:[string,number],index:number)=><li key={sector}><button type="button" disabled={busy} onClick={()=>onSector(sector)} aria-label={`Selecionar as ações mais líquidas de ${sector}, ${count} ${count===1 ? "ação":"ações"}, ${percent.format(count/stocks.length)} da lista local B3`}>
    <span className="home-sector-swatch" style={{background:mapShades[index%mapShades.length]}} aria-hidden="true"/>
    <span className="home-sector-name">{sector}</span>
    <span className="home-sector-values"><span className="home-sector-count">{number.format(count)} {count===1 ? "ação":"ações"}</span><span className="home-sector-percent">{percent.format(count/stocks.length)}</span></span>
  </button></li>;
  return <figure className="home-sector-map">
    <figcaption><strong>Explore por setor</strong><span>{number.format(stocks.length)} ações em {number.format(counts.size)} setores na lista local B3. Escolha um setor para selecionar os mais líquidos.</span></figcaption>
    <div className="home-sector-strip" aria-hidden="true">{ranked.map(([sector,count],index)=><span key={sector} title={`${sector}: ${count} ${count===1 ? "ação":"ações"} · ${percent.format(count/stocks.length)}`} style={{flexGrow:count,background:mapShades[index%mapShades.length]}}/>)}</div>
    <p className="home-sector-caption">{remaining.length ? `${top.length} maiores setores. Abra a lista abaixo para ver os demais.`:"Todos os setores, do maior para o menor."} Percentuais por quantidade de ações.</p>
    <ul className="home-sector-list">
      {top.map(sectorRow)}
    </ul>
    {remaining.length>0 && <details className="home-sector-details">
      <summary>Ver mais {remaining.length} {remaining.length===1 ? "setor":"setores"}</summary>
      <ul className="home-sector-list">{remaining.map((entry,index)=>sectorRow(entry,index+top.length))}</ul>
    </details>}
  </figure>;
}

export function HomeTools() {
  return <section className="home-tools-nav" aria-labelledby="home-tools-title">
    <h2 id="home-tools-title">Depois dos fundamentos</h2>
    <ul>{tools.map(tool=><li key={tool.href}><Link href={tool.href}>
      <span className="home-tool-icon"><Icon name={tool.icon}/></span>
      <span className="home-tool-text"><strong>{tool.label}</strong><span>{tool.text}</span></span>
      <svg className="home-tool-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
    </Link></li>)}</ul>
  </section>;
}
