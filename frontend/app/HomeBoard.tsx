"use client";
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { MAX_SELECTION, type Stock } from "../components/AppProvider";
import { Change, stripQuote, tickerColor, useMarket } from "../components/Shell";
import StockLogo from "../components/StockLogo";
import { formatBrasilia, formatCompactValue, formatValue, Loading, Notice, Pane, relativeAge, ValueText } from "../components/ui";
import { quickTickers, type HomeResult } from "../lib/home";
import { loadMarketNews, type MarketNews } from "../lib/marketNews";
import { squarify } from "../lib/treemap";

const brasiliaTime = new Intl.DateTimeFormat("pt-BR", { hour: "2-digit", minute: "2-digit", timeZone: "America/Sao_Paulo" });
const shortDate = (value: string) => value.split("-").reverse().slice(0, 2).join("/");

/** Where the quote sits in the 52-week range, drawn in a 64px cell. */
function Range52({ low, high, price }: { low: number | null; high: number | null; price: number | null }) {
  if (low === null || high === null || price === null || !(high > low)) return <span className="muted">—</span>;
  const at = Math.min(100, Math.max(0, (price - low) / (high - low) * 100));
  const label = `Faixa de 52 semanas: mínima ${formatValue(low, "money")}, máxima ${formatValue(high, "money")}, cotação ${formatValue(price, "money")}`;
  return <span className="r52" role="img" aria-label={label} title={label}><span style={{ left: `${at}%` }} /></span>;
}

type MonitorProps = {
  selection: string[]; stocks: Stock[]; watchlist: string[]; ready: HomeResult | null; busy: boolean; analyze: ReactNode; notices: ReactNode;
  onQuick: (ticker: string) => void; onFavorite: (ticker: string) => void; onAdd: (ticker: string) => void; onRemove: (ticker: string) => void;
};

/** Selection quote table: delayed Yahoo quote from the tape, Fundamentus indicators once analysed. */
export function Monitor({ selection, stocks, watchlist, ready, busy, analyze, notices, onQuick, onFavorite, onAdd, onRemove }: MonitorProps) {
  const { strip } = useMarket();
  const byTicker = new Map(stocks.map(stock => [stock.ticker, stock]));
  const companies = new Map(ready?.companies.map(company => [company.ticker, company]) ?? []);
  const units = new Map(ready?.comparison_indicators.map(item => [item.key, item.unit]) ?? []);
  const liquid = stocks.filter(stock => stock.liquidity !== null).sort((a, b) => (b.liquidity ?? 0) - (a.liquidity ?? 0));
  const starters = liquid.length ? liquid.slice(0, 14) : quickTickers.map(ticker => byTicker.get(ticker)).filter((stock): stock is Stock => !!stock);
  const quote = (ticker: string) => {
    const item = stripQuote(strip, ticker);
    const company = companies.get(ticker);
    if (item) return { value: item.value as number | null, change: item.change, source: `Yahoo Finance com atraso, ${shortDate(item.as_of)}` };
    return { value: company?.market.price ?? null, change: null, source: company?.market.last_quote_date ? `Fundamentus, ${company.market.last_quote_date.split("-").reverse().join("/")}` : "Sem cotação no painel" };
  };
  return <Pane title="Monitor" area="mon" className="pane-monitor" meta={`${selection.length}/${MAX_SELECTION}`} actions={analyze} flush>
    {notices}
    {selection.length ? <div className="grid-scroll" role="region" aria-label="Ativos selecionados" tabIndex={0}><table className="grid monitor">
      <thead><tr><th scope="col">Ativo</th><th scope="col">Empresa</th><th scope="col" className="num">Últ. R$</th><th scope="col" className="num">Var.</th>
        {ready ? <><th scope="col" className="num">P/L</th><th scope="col" className="num">ROE</th><th scope="col" className="num">DY</th><th scope="col" className="num">52 sem.</th><th scope="col" className="num" title="Áreas favoráveis na síntese setorial">Sínt.</th><th scope="col">Setor</th></>
          : <th scope="col" className="num" title="Volume financeiro médio diário em 2 meses">Liq. 2m</th>}
        <th scope="col"><span className="visually-hidden">Ações</span></th></tr></thead>
      <tbody>{selection.map((ticker, index) => {
        const stock = byTicker.get(ticker);
        const company = companies.get(ticker);
        const { value, change, source } = quote(ticker);
        const favorite = watchlist.includes(ticker);
        return <tr key={ticker}>
          <th scope="row" title={stock?.company ?? "Fora da lista B3"}><span className="tk"><i className="swatch" style={{ background: tickerColor(index) }} aria-hidden="true" /><StockLogo ticker={ticker} company={stock?.company ?? ticker} size={16} />{ticker}</span></th>
          <td className="ell">{company?.identity.company ?? stock?.company ?? "Fora da lista B3"}</td>
          <td className="num" title={source}>{value === null ? "—" : <ValueText value={value} unit="money" currency={company?.market.currency ?? "BRL"} />}</td>
          <td className="num">{change === null ? <span className="muted">—</span> : <Change value={change} />}</td>
          {ready ? <>
            {["pl", "roe", "dy"].map(key => <td className="num" key={key}>{company?.indicators_available ? formatValue(company.indicators[key], units.get(key) ?? "number", key === "pl" ? 1 : 1) : "—"}</td>)}
            <td className="num"><Range52 low={company?.market.low_52w ?? null} high={company?.market.high_52w ?? null} price={company?.market.price ?? null} /></td>
            <td className="num">{company?.synthesis.categorias_validas ? <span className="synth" title={`${company.synthesis.favorable_categories} de ${company.synthesis.categorias_validas} áreas favoráveis no setor`}>{Array.from({ length: company.synthesis.categorias_validas }, (_, slot) => <i key={slot} data-on={slot < company.synthesis.favorable_categories} />)}</span> : <span className="muted">—</span>}</td>
            <td className="ell">{[company?.identity.sector, company?.identity.subsector].filter(Boolean).join(" · ") || stock?.sector || "Setor não informado"}</td>
          </> : <td className="num">{stock?.liquidity == null ? "—" : formatCompactValue(stock.liquidity, "money", 1)}</td>}
          <td className="row-tools">
            <button type="button" className="icon-btn" aria-pressed={favorite} aria-label={favorite ? `Remover ${ticker} dos favoritos` : `Adicionar ${ticker} aos favoritos`} onClick={() => onFavorite(ticker)}>{favorite ? "★" : "☆"}</button>
            <button type="button" className="icon-btn" aria-label={`Remover ${ticker}`} disabled={busy} onClick={() => onRemove(ticker)}>×</button>
          </td>
        </tr>;
      })}</tbody>
    </table></div> : <>
      <p className="pane-lead">Seleção vazia. Busque com <kbd>Ctrl K</kbd> ou clique em um dos mais líquidos para analisar:</p>
      <div className="grid-scroll" role="region" aria-label="Ativos mais líquidos" tabIndex={0}><table className="grid monitor is-quick"><caption className="visually-hidden">Ativos mais líquidos da lista B3</caption>
        <thead><tr><th scope="col">Ativo</th><th scope="col">Empresa</th><th scope="col" className="num" title="Volume financeiro médio diário em 2 meses">Liq. 2m</th><th scope="col" className="num">Últ.</th><th scope="col" className="num">Var.</th></tr></thead>
        <tbody>{starters.map(stock => {
          const item = stripQuote(strip, stock.ticker);
          return <tr key={stock.ticker} className="is-clickable" onClick={() => onQuick(stock.ticker)}>
            <th scope="row"><button type="button" className="row-btn" disabled={busy} aria-label={`Analisar ${stock.ticker} agora`}><span className="tk"><StockLogo ticker={stock.ticker} company={stock.company} size={16} />{stock.ticker}</span></button></th>
            <td className="ell">{stock.company || "—"}</td>
            <td className="num">{stock.liquidity === null ? "—" : formatCompactValue(stock.liquidity, "money", 1)}</td>
            <td className="num">{item ? formatValue(item.value, "money") : <span className="muted">—</span>}</td>
            <td className="num">{item ? <Change value={item.change} /> : <span className="muted">—</span>}</td>
          </tr>;
        })}</tbody>
      </table></div>
    </>}
    {watchlist.length > 0 && <>
      <h3 className="pane-sub">Favoritos</h3>
      <ul className="fav-list">{watchlist.map(ticker => {
        const known = byTicker.has(ticker);
        const inside = selection.includes(ticker);
        return <li key={ticker}><b>{ticker}</b><span className="ell">{byTicker.get(ticker)?.company ?? "Fora da lista B3"}</span>
          {known && !inside && <button type="button" disabled={busy || selection.length >= MAX_SELECTION} onClick={() => onAdd(ticker)} aria-label={`Adicionar ${ticker} à seleção`}>+ seleção</button>}
          {inside && <span className="muted">na seleção</span>}
          <button type="button" className="icon-btn" aria-label={`Remover ${ticker} dos favoritos`} onClick={() => onFavorite(ticker)}>★</button>
        </li>;
      })}</ul>
    </>}
    <p className="pane-foot">Últ./Var.: Yahoo com atraso quando disponível na faixa de cotações; senão, último preço Fundamentus.</p>
  </Pane>;
}

type Tile = { sector: string; value: number; liquidity: number | null; known: boolean };
const liquiditySteps = 5;

/** Treemap of the local B3 list: area is the number of listed stocks, shade the summed 2-month liquidity. */
export function SectorMap({ stocks, busy, active, available, onSector, onReset, onRefresh, note }: { stocks: Stock[]; busy: boolean; active: string[]; available: number; onSector: (sector: string) => void; onReset: () => void; onRefresh: () => void; note: ReactNode }) {
  const map = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 200, h: 100 });
  useEffect(() => {
    const element = map.current;
    if (!element) return;
    // Lay tiles out in the pane's real aspect ratio so they stay close to square.
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setBox(current => Math.abs(current.w - width) < 2 && Math.abs(current.h - height) < 2 ? current : { w: width, h: height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const groups = new Map<string, { count: number; liquidity: number; priced: number }>();
  for (const stock of stocks) {
    const key = stock.sector || "";
    const entry = groups.get(key) ?? { count: 0, liquidity: 0, priced: 0 };
    entry.count++;
    if (stock.liquidity !== null) { entry.liquidity += stock.liquidity; entry.priced++; }
    groups.set(key, entry);
  }
  const tiles: Tile[] = [...groups].map(([sector, entry]) => ({ sector: sector || "Setor não informado", value: entry.count, liquidity: entry.priced ? entry.liquidity : null, known: !!sector }));
  const logs = tiles.map(tile => tile.liquidity).filter((value): value is number => value !== null && value > 0).map(Math.log10);
  const low = Math.min(...logs), high = Math.max(...logs);
  const step = (liquidity: number | null) => liquidity === null || liquidity <= 0 || !logs.length ? -1 : high === low ? liquiditySteps - 1 : Math.min(liquiditySteps - 1, Math.floor((Math.log10(liquidity) - low) / (high - low) * liquiditySteps));
  const laid = squarify(tiles, { x: 0, y: 0, w: box.w, h: box.h });
  const filtered = active.length > 0 && !active.includes("Todos");
  return <Pane title="Setores" area="sec" className="pane-sectors" meta={`${stocks.length} ações · ${groups.size} setores${filtered ? ` · ${active.join(", ")}: ${available}` : ""}`}
    actions={<>{filtered && <button type="button" disabled={busy} onClick={onReset}>Todos</button>}<button type="button" disabled={busy} onClick={onRefresh} title="Limpa o cache Fundamentus e busca os dados de novo">Atualizar Fundamentus</button></>}>
    <div className="tmap" ref={map} role="group" aria-label="Mapa de setores da lista local B3">
      {laid.map(tile => {
        const shade = step(tile.liquidity);
        const style = { left: `${tile.x / box.w * 100}%`, top: `${tile.y / box.h * 100}%`, width: `${tile.w / box.w * 100}%`, height: `${tile.h / box.h * 100}%`, "--shade": shade < 0 ? 0 : (shade + 1) / liquiditySteps } as CSSProperties;
        const liquidity = tile.liquidity === null ? "liquidez indisponível" : `liquidez 2m somada ${formatCompactValue(tile.liquidity, "money", 1)}`;
        const label = `${tile.sector}: ${tile.value} ${tile.value === 1 ? "ação" : "ações"}, ${liquidity}`;
        return <button type="button" key={tile.sector} className={`tile${tile.known ? "" : " is-unknown"}${active.includes(tile.sector) ? " is-active" : ""}`} style={style}
          disabled={busy || !tile.known} title={tile.known ? `${label}. Selecionar os mais líquidos.` : `${label}. Sem setor informado; não pode ser escolhido por setor.`} aria-label={tile.known ? `Selecionar os mais líquidos de ${label}` : label}
          onClick={() => onSector(tile.sector)}>
          <span className="tile-name">{tile.sector}</span>
          <span className="tile-data"><b>{tile.value}</b>{tile.liquidity !== null && <span>{formatCompactValue(tile.liquidity, "money", 0)}</span>}</span>
        </button>;
      })}
    </div>
    <div className="tmap-legend" aria-hidden="true"><span>Área: nº de ações</span><span className="tmap-scale">Liquidez 2m {Array.from({ length: liquiditySteps }, (_, index) => <i key={index} style={{ "--shade": (index + 1) / liquiditySteps } as CSSProperties} />)} maior</span><span>Clique: seleciona os mais líquidos</span></div>
    {note}
  </Pane>;
}

/** BCB macro series and index quotes from the shell's market strip, each with its reference date. */
export function Macro() {
  const { strip, failed } = useMarket();
  const rows = strip ? [...strip.items.filter(item => item.group === "macro"), ...strip.items.filter(item => item.group === "market")] : [];
  return <Pane title="Macro e índices" area="mac" className="pane-macro" meta={strip ? `atualizado ${formatBrasilia(strip.updated_at, false).replace(" (horário de Brasília)", "")}` : undefined} flush>
    {!strip ? failed ? <Notice tone="warning">Indicadores indisponíveis no momento.</Notice> : <Loading>Carregando indicadores…</Loading> : <div className="grid-scroll"><table className="grid macro">
      <thead><tr><th scope="col">Série</th><th scope="col" className="num">Valor</th><th scope="col" className="num">Var.</th><th scope="col" className="num">Ref.</th></tr></thead>
      <tbody>{rows.map(item => <tr key={item.key} title={`${item.source}${item.note ? ` · ${item.note}` : ""}`}>
        <th scope="row">{item.label}{item.note && <small>{item.note}</small>}</th>
        <td className="num">{formatValue(item.value, item.unit, item.digits, item.currency ?? "BRL")}</td>
        <td className="num">{item.change === null ? <span className="muted">—</span> : <Change value={item.change} />}</td>
        <td className="num muted">{shortDate(item.as_of)}</td>
      </tr>)}</tbody>
    </table></div>}
    {strip && strip.missing.length > 0 && <p className="pane-foot">Indisponível: {strip.missing.join(", ")}.</p>}
    {strip && <p className="pane-foot">BCB SGS (Selic, CDI, IPCA, desemprego) · Yahoo Finance com atraso.</p>}
  </Pane>;
}

type Feed = { kind: "loading" } | { kind: "ready"; data: MarketNews } | { kind: "error"; message: string };

/** Market headlines ranked by keyword relevance; the list links out to the publisher. */
export function Headlines() {
  const [feed, setFeed] = useState<Feed>({ kind: "loading" });
  const request = useRef(0);
  async function load() {
    const current = ++request.current;
    setFeed({ kind: "loading" });
    try { const data = await loadMarketNews(); if (request.current === current) setFeed({ kind: "ready", data }); }
    catch (error) { if (request.current === current) setFeed({ kind: "error", message: error instanceof Error ? error.message : "Não foi possível carregar as manchetes." }); }
  }
  useEffect(() => { void load(); return () => { request.current++; }; }, []);
  return <Pane title="Manchetes do mercado" area="news" className="pane-news" meta={feed.kind === "ready" ? `${feed.data.window_hours} h · relevância` : undefined}
    actions={<button type="button" disabled={feed.kind === "loading"} onClick={() => void load()}>Atualizar</button>} flush>
    {feed.kind === "loading" && <Loading>Buscando manchetes…</Loading>}
    {feed.kind === "error" && <Notice tone="negative">{feed.message}</Notice>}
    {feed.kind === "ready" && (feed.data.articles.length ? <ol className="heads">{feed.data.articles.map(article => <li key={article.id}>
      <time dateTime={article.published} title={formatBrasilia(article.published)}>{brasiliaTime.format(new Date(article.published))}</time>
      {article.importance_label === "Alta" ? <span className="flag" title="Importância alta por palavras-chave">Alta</span> : <span className="flag is-dim" title={`Importância ${article.importance_label.toLocaleLowerCase("pt-BR")}`}>{article.importance_label.slice(0, 5)}</span>}
      <span className="heads-title">{article.link ? <a href={article.link} target="_blank" rel="noopener noreferrer">{article.title}<span className="visually-hidden"> (abre em nova aba)</span></a> : article.title}<small>{article.provider} · {relativeAge(article.published)}</small></span>
    </li>)}</ol> : <p className="pane-lead">Nenhuma manchete nas últimas {feed.data.window_hours} horas.</p>)}
    <p className="pane-foot">Google News e publicadores · relevância por palavras-chave.</p>
  </Pane>;
}
