"use client";
import Link from "next/link";
import type { Stock } from "../components/AppProvider";
import { Icon } from "../components/ui";

const mapSegments = 7;
const mapShades = ["var(--indigo-300)","var(--indigo-400)","var(--indigo-500)","var(--indigo-600)","var(--indigo-700)","var(--indigo-800)","var(--indigo-900)","var(--surface-raised)"];
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
  const top=ranked.slice(0,mapSegments);
  const others=ranked.slice(mapSegments).reduce((sum,[,count])=>sum+count,0);
  if(!top.length)return null;
  const number=new Intl.NumberFormat("pt-BR");
  return <figure className="home-sector-map">
    <figcaption><strong>Explore por setor</strong><span>{number.format(stocks.length)} ações em {number.format(counts.size)} setores na lista local B3. Escolha um setor para selecionar os mais líquidos.</span></figcaption>
    <div className="home-sector-strip" aria-hidden="true">{top.map(([sector,count],index)=><span key={sector} style={{flexGrow:count,background:mapShades[index]}}/>)}{others>0 && <span style={{flexGrow:others,background:mapShades[mapSegments]}}/>}</div>
    <ul className="home-sector-list">
      {top.map(([sector,count],index)=><li key={sector}><button type="button" disabled={busy} onClick={()=>onSector(sector)} aria-label={`Selecionar as ações mais líquidas de ${sector}, ${count} ações`}>
        <span className="home-sector-swatch" style={{background:mapShades[index]}} aria-hidden="true"/>
        <span className="home-sector-name">{sector}</span>
        <span className="home-sector-count">{count}</span>
      </button></li>)}
      {others>0 && <li className="home-sector-others"><span className="home-sector-swatch" style={{background:mapShades[mapSegments]}} aria-hidden="true"/><span className="home-sector-name">Outros {ranked.length-mapSegments} setores</span><span className="home-sector-count">{others}</span></li>}
    </ul>
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
