"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { MAX_SELECTION, type Stock } from "../components/AppProvider";
import { Change, stripQuote, tickerColor, useMarket } from "../components/Shell";
import StockLogo from "../components/StockLogo";
import { formatBrasilia, formatCompactValue, formatValue, Loading, Notice, Pane, relativeAge, ScrollX, useMedia, ValueText } from "../components/ui";
import { quickTickers, universeChanges, type HomeResult, type UniverseChanges } from "../lib/home";
import { loadMarketNews, type MarketNews } from "../lib/marketNews";
import { nestedSquarify } from "../lib/treemap";

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
  const narrow = useMedia("(max-width: 900px)");
  const [showAll, setShowAll] = useState(false);
  // On phones the starter list stops at six rows so the explanation and the sector map stay within reach.
  const shownStarters = narrow && !showAll ? starters.slice(0, 6) : starters;
  const quote = (ticker: string) => {
    const item = stripQuote(strip, ticker);
    const company = companies.get(ticker);
    if (item) return { value: item.value as number | null, change: item.change, source: `Yahoo Finance com atraso, ${shortDate(item.as_of)}` };
    return { value: company?.market.price ?? null, change: null, source: company?.market.last_quote_date ? `Fundamentus, ${company.market.last_quote_date.split("-").reverse().join("/")}` : "Sem cotação no painel" };
  };
  return <Pane title="Monitor" area="mon" className="pane-monitor" meta={`${selection.length}/${MAX_SELECTION}`} actions={analyze} flush>
    {notices}
    {selection.length ? <ScrollX label="Ativos selecionados"><table className="grid monitor">
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
    </table></ScrollX> : <>
      <p className="pane-lead">Seleção vazia. Busque com <kbd>Ctrl K</kbd> ou clique em um dos mais líquidos para analisar:</p>
      <ScrollX label="Ativos mais líquidos"><table className="grid monitor is-quick"><caption className="visually-hidden">Ativos mais líquidos da lista B3</caption>
        <thead><tr><th scope="col">Ativo</th><th scope="col">Empresa</th><th scope="col" className="num" title="Volume financeiro médio diário em 2 meses">Liq. 2m</th><th scope="col" className="num">Últ.</th><th scope="col" className="num">Var.</th></tr></thead>
        <tbody>{shownStarters.map(stock => {
          const item = stripQuote(strip, stock.ticker);
          return <tr key={stock.ticker} className="is-clickable" onClick={() => onQuick(stock.ticker)}>
            <th scope="row"><button type="button" className="row-btn" disabled={busy} aria-label={`Analisar ${stock.ticker} agora`}><span className="tk"><StockLogo ticker={stock.ticker} company={stock.company} size={16} />{stock.ticker}</span></button></th>
            <td className="ell">{stock.company || "—"}</td>
            <td className="num">{stock.liquidity === null ? "—" : formatCompactValue(stock.liquidity, "money", 1)}</td>
            <td className="num">{item ? formatValue(item.value, "money") : <span className="muted">—</span>}</td>
            <td className="num">{item ? <Change value={item.change} /> : <span className="muted">—</span>}</td>
          </tr>;
        })}</tbody>
      </table></ScrollX>
      {narrow && starters.length > 6 && <button type="button" className="more-rows" aria-expanded={showAll} onClick={() => setShowAll(value => !value)}>{showAll ? "Ver menos" : `Ver mais ${starters.length - 6} ativos líquidos`}</button>}
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

type SizeBy = "liquidity" | "equal";
type ColorBy = "day" | "week";
type Block = Stock & { value: number; change: number | null };
type Changes = { kind: "loading" } | { kind: "ready"; data: UniverseChanges } | { kind: "error"; message: string };
const sizeOptions: Record<SizeBy, string> = { liquidity: "Liquidez 2m", equal: "Igual" };
const colorOptions: Record<ColorBy, { label: string; short: string; step: number }> = {
  day: { label: "Variação do dia", short: "Var. dia", step: 0.01 },
  week: { label: "Variação em 5 pregões", short: "Var. 5 pregões", step: 0.02 },
};
const headerHeight = 15;

/** Diverging bucket of a change: -4…-1 red, 0 flat, 1…4 green; each step is `step` wide and the last one is open-ended. */
function tone(change: number | null, step: number): number | null {
  if (change === null) return null;
  if (change === 0) return 0;
  return Math.sign(change) * Math.min(4, Math.floor(Math.abs(change) / step) + 1);
}

/** Stock heatmap of the local B3 list, nested by sector: area is the size metric, colour the sign and size of the change. */
export function SectorMap({ stocks, selection, busy, active, available, onSector, onAdd, onReset, onRefresh, note }: { stocks: Stock[]; selection: string[]; busy: boolean; active: string[]; available: number; onSector: (sector: string) => void; onAdd: (ticker: string) => void; onReset: () => void; onRefresh: () => void; note: ReactNode }) {
  const map = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 200, h: 100 });
  const [sizeBy, setSizeBy] = useState<SizeBy>("liquidity");
  const [colorBy, setColorBy] = useState<ColorBy>("day");
  const [only, setOnly] = useState("");
  const [focus, setFocus] = useState<string | null>(null);
  const [changes, setChanges] = useState<Changes>({ kind: "loading" });
  useEffect(() => {
    let live = true;
    universeChanges().then(data => { if (live) setChanges({ kind: "ready", data }); }, error => { if (live) setChanges({ kind: "error", message: error instanceof Error ? error.message : "Variações indisponíveis." }); });
    return () => { live = false; };
  }, []);
  useEffect(() => {
    const element = map.current;
    if (!element) return;
    // Lay blocks out in the pane's real aspect ratio so they stay close to square.
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setBox(current => Math.abs(current.w - width) < 2 && Math.abs(current.h - height) < 2 ? current : { w: width, h: height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const color = colorOptions[colorBy];
  const changeOf = (ticker: string) => changes.kind === "ready" ? changes.data.changes[ticker]?.[colorBy] ?? null : null;
  const sectors = new Map<string, Block[]>();
  let unsized = 0;
  for (const stock of stocks) {
    const value = sizeBy === "equal" ? 1 : stock.liquidity ?? 0;
    if (!(value > 0)) { unsized++; continue; }
    const key = stock.sector || "Setor não informado";
    sectors.set(key, [...sectors.get(key) ?? [], { ...stock, value, change: changeOf(stock.ticker) }]);
  }
  const names = [...sectors.keys()].sort((a, b) => a.localeCompare(b, "pt-BR"));
  const shown = only && sectors.has(only) ? [[only, sectors.get(only)!] as const] : [...sectors];
  const laid = nestedSquarify(shown.map(([key, items]) => ({ key, items })), { x: 0, y: 0, w: box.w, h: box.h }, headerHeight);
  const missing = changes.kind === "ready" ? stocks.filter(stock => changeOf(stock.ticker) === null).length : 0;
  const filtered = active.length > 0 && !active.includes("Todos");
  const pct = (x: number) => `${x / box.w * 100}%`, pctY = (y: number) => `${y / box.h * 100}%`;
  const describe = (block: Block, sector: string) => [`${block.ticker} · ${block.company || "empresa não informada"}`, sector,
    `Liquidez 2m ${block.liquidity === null ? "indisponível" : formatCompactValue(block.liquidity, "money", 1)}`,
    `${color.short} ${changes.kind === "loading" ? "carregando" : block.change === null ? "sem dado" : formatValue(block.change, "fraction", 2, "BRL", true)}`].join(" · ");
  const full = selection.length >= MAX_SELECTION;
  return <Pane title="Mapa de calor" area="sec" className="pane-sectors" meta={`${stocks.length} ações · ${sectors.size} setores${filtered ? ` · ${active.join(", ")}: ${available}` : ""}`}
    actions={<>{filtered && <button type="button" disabled={busy} onClick={onReset}>Todos</button>}<button type="button" disabled={busy} onClick={onRefresh} title="Limpa o cache Fundamentus e busca os dados de novo">Atualizar Fundamentus</button></>}>
    <div className="hm-controls">
      <label>Tamanho<select aria-label="Tamanho" value={sizeBy} onChange={event => setSizeBy(event.target.value as SizeBy)}>{Object.entries(sizeOptions).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
      <label>Cor<select aria-label="Cor" value={colorBy} onChange={event => setColorBy(event.target.value as ColorBy)}>{Object.entries(colorOptions).map(([key, option]) => <option key={key} value={key}>{option.label}</option>)}</select></label>
      <label>Setor<select aria-label="Setor" value={only} onChange={event => setOnly(event.target.value)}><option value="">Todos os setores</option>{names.map(name => <option key={name} value={name}>{name}</option>)}</select></label>
    </div>
    <div className="hmap" ref={map} role="group" aria-label={`Mapa de calor: área por ${sizeOptions[sizeBy].toLocaleLowerCase("pt-BR")}, cor por ${color.label.toLocaleLowerCase("pt-BR")}`}>
      {laid.map(sector => {
        const known = sector.key !== "Setor não informado";
        return <div key={sector.key} className="hm-sector" style={{ left: pct(sector.x), top: pctY(sector.y), width: pct(sector.w), height: pctY(sector.h) }}>
          {sector.header && (known
            ? <button type="button" className="hm-head" disabled={busy} title={`${sector.key}: selecionar os mais líquidos`} aria-label={`Selecionar os mais líquidos de ${sector.key}`} onClick={() => onSector(sector.key)}>{sector.key} <span aria-hidden="true">›</span></button>
            : <span className="hm-head">{sector.key}</span>)}
        </div>;
      })}
      {laid.flatMap(sector => sector.items.map(block => {
        const level = tone(block.change, color.step);
        const big = block.w >= 64 && block.h >= 52, medium = !big && block.w >= 34 && block.h >= 30;
        const font = big ? Math.round(Math.min(22, Math.max(11, Math.min(block.w / 5.5, block.h / 4.5)))) : 10;
        const logo = big ? Math.round(Math.min(32, Math.max(16, block.h / 4))) : 14;
        const inside = selection.includes(block.ticker);
        const label = describe(block, sector.key);
        const value = changes.kind === "loading" ? "…" : block.change === null ? "s/d" : formatValue(block.change, "fraction", big ? 2 : 1, "BRL", true);
        return <button type="button" key={block.ticker} className={`hm-block${inside ? " is-selected" : ""}`} data-ticker={block.ticker} data-sector={sector.key} data-tone={level ?? "none"} data-change={block.change ?? ""} data-size={block.value}
          style={{ left: pct(block.x), top: pctY(block.y), width: pct(block.w), height: pctY(block.h) }}
          disabled={busy || (!inside && full)} aria-label={inside ? `${label}. Já está na seleção.` : full ? `${label}. Seleção cheia.` : `Adicionar à seleção: ${label}`}
          onClick={() => { if (!inside) onAdd(block.ticker); }} onMouseEnter={() => setFocus(label)} onMouseLeave={() => setFocus(null)} onFocus={() => setFocus(label)} onBlur={() => setFocus(null)}>
          {(big || medium) && <span className="hm-body" aria-hidden="true">
            <StockLogo ticker={block.ticker} company={block.company} size={logo} />
            {big && <b style={{ fontSize: font }}>{block.ticker}</b>}
            <span style={{ fontSize: big ? Math.max(10, Math.round(font * .72)) : 10 }}>{value}</span>
          </span>}
        </button>;
      }))}
    </div>
    <div className="hm-legend">
      <span className="hm-readout" aria-live="polite">{focus ?? <>Área: {sizeOptions[sizeBy].toLocaleLowerCase("pt-BR")} · clique adiciona à seleção, o nome do setor pega os mais líquidos</>}</span>
      <span className="hm-scale" role="img" aria-label={`Escala de ${color.label.toLocaleLowerCase("pt-BR")}: vermelho para queda, cinza para zero, verde para alta, mais forte a cada ${formatValue(color.step, "fraction", 0)}; hachurado sem dado`}>
        <span className="hm-swatches">{[-4, -3, -2, -1, 0, 1, 2, 3, 4].map(level => <i key={level} data-tone={level} />)}</span>
        <span className="hm-ticks" aria-hidden="true">{[[1, -3], [2, -2], [3, -1], [4.5, 0], [6, 1], [7, 2], [8, 3]].map(([at, n]) => <span key={n} style={{ left: at * 24 }}>{n === 0 ? "0" : `${n < 0 ? "−" : "+"}${formatValue(Math.abs(n) * color.step, "fraction", 0)}`}</span>)}</span>
      </span>
      <span className="hm-nodata"><i data-tone="none" aria-hidden="true" />sem dado{missing ? ` (${missing})` : ""}</span>
    </div>
    <p className="pane-foot">{changes.kind === "loading" ? "Carregando variações do Yahoo Finance…" : changes.kind === "error" ? `Variações indisponíveis: ${changes.message}` : `${color.label}: ${changes.data.source}, pregão de ${shortDate(changes.data.session)}.`}
      {" "}Tamanho: Fundamentus, liquidez média diária em 2 meses.{unsized > 0 && sizeBy === "liquidity" ? ` ${unsized} ações sem liquidez ficam fora do mapa.` : ""}</p>
    {note}
  </Pane>;
}

/** BCB macro series and index quotes from the shell's market strip, each with its reference date. */
export function Macro() {
  const { strip, failed } = useMarket();
  const rows = strip ? [...strip.items.filter(item => item.group === "macro"), ...strip.items.filter(item => item.group === "market")] : [];
  return <Pane title="Macro e índices" area="mac" className="pane-macro" meta={strip ? `atualizado ${formatBrasilia(strip.updated_at, false).replace(" (horário de Brasília)", "")}` : undefined} flush>
    {!strip ? failed ? <Notice tone="warning">Indicadores indisponíveis no momento.</Notice> : <Loading>Carregando indicadores…</Loading> : <ScrollX label="Macro e índices"><table className="grid macro">
      <thead><tr><th scope="col">Série</th><th scope="col" className="num">Valor</th><th scope="col" className="num">Var.</th><th scope="col" className="num">Ref.</th></tr></thead>
      <tbody>{rows.map(item => <tr key={item.key} title={`${item.source}${item.note ? ` · ${item.note}` : ""}`}>
        <th scope="row">{item.label}{item.note && <small>{item.note}</small>}</th>
        <td className="num">{formatValue(item.value, item.unit, item.digits, item.currency ?? "BRL")}</td>
        <td className="num">{item.change === null ? <span className="muted">—</span> : <Change value={item.change} />}</td>
        <td className="num muted">{shortDate(item.as_of)}</td>
      </tr>)}</tbody>
    </table></ScrollX>}
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
