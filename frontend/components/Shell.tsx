"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { subscribeActivity } from "../lib/api";
import { comparisonColors } from "../lib/home";
import { rankStocks, sortStocks } from "../lib/tickerSearch";
import { MAX_SELECTION, SELECTION_FULL_MESSAGE, useApp, type Session, type Stock } from "./AppProvider";
import { formatBrasilia, formatCompactValue, formatValue, type Unit } from "./ui";

export type StripItem = { key: string; group: "macro" | "market" | "stock"; label: string; symbol: string | null; value: number; unit: Unit; currency: string | null; digits: number; change: number | null; change_unit: ChangeUnit | null; change_note: string | null; note: string | null; as_of: string; source: string };
/** Quote changes are daily fractions; macro changes are percentage-point differences. */
export type ChangeUnit = "fraction" | "pp";
export type Strip = { items: StripItem[]; missing: string[]; updated_at: string };
type Market = { strip: Strip | null; failed: boolean };

const MarketContext = createContext<Market>({ strip: null, failed: false });
/** Delayed quotes and BCB macro series fetched once by the shell for the tape, the macro pane and the monitor. */
export const useMarket = () => useContext(MarketContext);
export function stripQuote(strip: Strip | null, ticker: string): StripItem | undefined {
  return strip?.items.find(item => item.symbol === `${ticker}.SA`);
}

/** One color per position in the selection, shared by chips, monitor rows and comparison bars. */
export const tickerColor = (index: number) => comparisonColors[index % comparisonColors.length];

type View = { key: string; href: string; label: string; ready?: (session: Session) => boolean; requirement?: string };
export const views: View[] = [
  { key: "F1", href: "/", label: "Início" },
  { key: "F2", href: "/portfolio", label: "Carteira", ready: session => session.portfolio_ready, requirement: "carteira ainda não carregada" },
  { key: "F3", href: "/simulacao", label: "Simulação", ready: session => session.simulation_ready, requirement: "requer carteira carregada" },
  { key: "F4", href: "/noticias", label: "Notícias", ready: session => session.portfolio_ready, requirement: "requer carteira carregada" },
  { key: "F5", href: "/screener", label: "Screener" },
];

const refreshMs = 5 * 60 * 1000;
const brasiliaClock = new Intl.DateTimeFormat("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "America/Sao_Paulo" });
const brasiliaParts = new Intl.DateTimeFormat("en-US", { weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZone: "America/Sao_Paulo" });
/** Regular B3 equity session by the clock only; holidays are not known here, so the label says "estimado". */
function sessionOpen(now: Date): boolean {
  const parts = Object.fromEntries(brasiliaParts.formatToParts(now).map(part => [part.type, part.value]));
  const minutes = Number(parts.hour) * 60 + Number(parts.minute);
  return !["Sat", "Sun"].includes(parts.weekday) && minutes >= 600 && minutes < 1020;
}

function Change({ value, unit = "fraction", tone: toned = true }: { value: number | null; unit?: ChangeUnit | null; tone?: boolean }) {
  if (value === null) return null;
  const tone = !toned ? "neutral" : value > 0 ? "positive" : value < 0 ? "negative" : "neutral";
  const text = unit === "pp" ? `${formatValue(value, "number", 2, "BRL", true)} p.p.` : formatValue(value, "fraction", 2, "BRL", true);
  return <span className={`chg is-${tone}`}><span aria-hidden="true">{value > 0 ? "▲" : value < 0 ? "▼" : "■"}</span>{text}</span>;
}
export { Change };

function Tape() {
  const { strip, failed } = useMarket();
  const [paused, setPaused] = useState(false);
  if (!strip?.items.length) return <div className="tape is-empty" role="status">{failed || strip ? "Cotações e indicadores indisponíveis no momento." : "Carregando cotações…"}</div>;
  const items = strip.items.filter(item => item.group !== "macro");
  const list = (hidden?: boolean) => <ul className="tape-list" aria-hidden={hidden || undefined}>{items.map(item => <li key={item.key} title={`${item.label}: ${item.source}, referência ${item.as_of.split("-").reverse().join("/")}`}>
    <b>{item.label}</b><span>{formatValue(item.value, item.unit, item.digits, item.currency ?? "BRL")}</span><Change value={item.change} unit={item.change_unit} />
  </li>)}</ul>;
  return <section className="tape" aria-label="Cotações com atraso" data-paused={paused}>
    <div className="tape-viewport"><div className="tape-track" style={{ animationDuration: `${items.length * 5}s` }}>{list()}{list(true)}</div></div>
    <span className="tape-source">Yahoo · atraso · {formatBrasilia(strip.updated_at, false).replace(" (horário de Brasília)", "")}</span>
    <button type="button" className="tape-toggle" aria-pressed={paused} aria-label={paused ? "Retomar rolagem das cotações" : "Pausar rolagem das cotações"} onClick={() => setPaused(!paused)}>{paused ? "▶" : "❚❚"}</button>
  </section>;
}

function Clock() {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  if (!now) return <span className="cmd-clock" />;
  const open = sessionOpen(now);
  return <span className="cmd-clock" title="Horário de Brasília. Pregão regular estimado pelo relógio (10h às 17h, dias úteis); feriados não são considerados.">
    <span className={`cmd-market is-${open ? "open" : "closed"}`}><span aria-hidden="true">{open ? "●" : "○"}</span>B3 {open ? "aberta" : "fechada"}</span>
    <time>{brasiliaClock.format(now)}</time><small>BRT</small>
  </span>;
}

type Flash = { text: string; tone: "positive" | "negative" | "info"; undo?: string[] };
type Option = { kind: "view"; view: View } | { kind: "stock"; stock: Stock };

function CommandSearch({ onFlash, compact }: { onFlash: (flash: Flash) => void; compact: boolean }) {
  const { session, stocks, setSelection, toggleFavorite } = useApp();
  const router = useRouter();
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const busy = useRef(false);
  const selection = session?.selection ?? [];
  const watchlist = session?.watchlist ?? [];
  const options = useMemo<Option[]>(() => {
    const needle = query.trim().toLocaleLowerCase("pt-BR");
    const viewHits = needle ? views.filter(view => view.key.toLowerCase() === needle || view.label.toLocaleLowerCase("pt-BR").startsWith(needle)).map(view => ({ kind: "view" as const, view })) : [];
    const liquid = sortStocks(stocks, "liquidity");
    const pool = needle ? liquid : [...liquid.filter(stock => watchlist.includes(stock.ticker)), ...liquid.filter(stock => !watchlist.includes(stock.ticker))];
    return [...viewHits, ...rankStocks(pool, query, 9).items.map(stock => ({ kind: "stock" as const, stock }))];
  }, [query, stocks, watchlist]);
  useEffect(() => {
    const focus = () => { input.current?.focus(); input.current?.select(); setOpen(true); };
    window.addEventListener("b3lab:focus-search", focus);
    return () => window.removeEventListener("b3lab:focus-search", focus);
  }, []);
  const current = Math.min(active, Math.max(0, options.length - 1));
  async function choose(option: Option | undefined) {
    if (!option || busy.current) return;
    if (option.kind === "view") { setQuery(""); setOpen(false); router.push(option.view.href); return; }
    const ticker = option.stock.ticker;
    const before = [...selection];
    const removing = before.includes(ticker);
    if (!removing && before.length >= MAX_SELECTION) { onFlash({ text: SELECTION_FULL_MESSAGE, tone: "negative" }); return; }
    busy.current = true;
    setQuery(""); setActive(0); setOpen(false);
    try {
      await setSelection(current => removing ? current.filter(item => item !== ticker) : current.includes(ticker) ? [...current] : [...current, ticker]);
      onFlash({ text: removing ? `${ticker} removida da seleção.` : `${ticker} adicionada à seleção (${Math.min(before.length + 1, MAX_SELECTION)}/${MAX_SELECTION}).`, tone: "positive", undo: before });
    } catch (reason) {
      onFlash({ text: reason instanceof Error ? reason.message : String(reason), tone: "negative" });
    } finally { busy.current = false; }
  }
  const expanded = open && options.length > 0;
  return <div className="cmd-search" onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false); }}>
    <span className="cmd-prompt" aria-hidden="true">&gt;</span>
    <input ref={input} type="search" autoComplete="off" spellCheck={false} value={query}
      aria-label="Buscar ativo: ticker, empresa ou setor" aria-controls={`${id}-list`} aria-activedescendant={expanded ? `${id}-${current}` : undefined}
      placeholder={compact ? "Ticker ou empresa" : "Ticker, empresa, setor ou F1–F5"}
      onFocus={() => setOpen(true)} onClick={() => setOpen(true)}
      onChange={event => { setQuery(event.target.value); setActive(0); setOpen(true); }}
      onKeyDown={event => {
        if (event.key === "ArrowDown") { event.preventDefault(); setOpen(true); setActive(Math.min(current + 1, options.length - 1)); }
        else if (event.key === "ArrowUp") { event.preventDefault(); setActive(Math.max(current - 1, 0)); }
        else if (event.key === "Enter" && expanded) { event.preventDefault(); void choose(options[current]); }
        else if (event.key === "Escape") { event.preventDefault(); if (open) setOpen(false); else input.current?.blur(); }
      }} />
    {!compact && <kbd className="cmd-kbd" aria-hidden="true">Ctrl K</kbd>}
    {expanded && <ul className="cmd-results" id={`${id}-list`} role="listbox" aria-label="Resultados da busca">
      {options.map((option, index) => {
        const selected = option.kind === "stock" && selection.includes(option.stock.ticker);
        const favorite = option.kind === "stock" && watchlist.includes(option.stock.ticker);
        return <li key={option.kind === "view" ? option.view.key : option.stock.ticker} id={`${id}-${index}`} role="option" aria-selected={index === current} className={`cmd-option${selected ? " is-selected" : ""}`}
          onMouseMove={() => { if (index !== current) setActive(index); }} onMouseDown={event => event.preventDefault()} onClick={() => void choose(option)}>
          {option.kind === "view" ? <>
            <span className="cmd-option-mark" aria-hidden="true">↗</span><b>{option.view.key}</b><span className="cmd-option-name">Ir para {option.view.label}</span><span className="cmd-option-hint">abrir</span>
          </> : <>
            <span className="cmd-option-mark" aria-hidden="true">{selected ? "✓" : "+"}</span>
            <b>{option.stock.ticker}</b>
            <span className="cmd-option-name">{option.stock.company || "Empresa não informada"}<small>{option.stock.sector || "Setor não informado"}</small></span>
            <span className="cmd-option-liq" title="Volume financeiro médio diário em 2 meses">{option.stock.liquidity === null ? "—" : formatCompactValue(option.stock.liquidity, "money", 1)}</span>
            <span className="cmd-option-hint">{selected ? "remover" : "adicionar"}</span>
            <button type="button" className="cmd-star" tabIndex={-1} aria-pressed={favorite} aria-label={favorite ? `Remover ${option.stock.ticker} dos favoritos` : `Adicionar ${option.stock.ticker} aos favoritos`}
              onClick={event => { event.stopPropagation(); void toggleFavorite(option.stock.ticker).catch(reason => onFlash({ text: String(reason instanceof Error ? reason.message : reason), tone: "negative" })); }}>{favorite ? "★" : "☆"}</button>
          </>}
        </li>;
      })}
      <li className="cmd-results-foot" role="presentation">↑↓ navegar · Enter adiciona ou remove · Esc fecha · a seleção vale para todas as telas</li>
    </ul>}
  </div>;
}

function SelectionChips({ onFlash }: { onFlash: (flash: Flash) => void }) {
  const { session, stocks, setSelection } = useApp();
  const [open, setOpen] = useState(false);
  const [hidden, setHidden] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const selection = session?.selection ?? [];
  const selectionKey = selection.join();
  useLayoutEffect(() => {
    const element = list.current;
    if (!element) return;
    // Chips wrap onto clipped lines when the bar is full; count them so the "+N" stays honest.
    const measure = () => {
      const chips = Array.from(element.children) as HTMLElement[];
      setHidden(chips.filter(chip => chip.offsetTop > chips[0].offsetTop).length);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [selectionKey]);
  const known = new Map(stocks.map(stock => [stock.ticker, stock.company]));
  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("pointerdown", onPointer); document.removeEventListener("keydown", onKey); };
  }, [open]);
  async function update(next: string[], text: string) {
    const before = [...selection];
    try { await setSelection(next); onFlash({ text, tone: "positive", undo: before }); }
    catch (reason) { onFlash({ text: reason instanceof Error ? reason.message : String(reason), tone: "negative" }); }
  }
  const remove = (ticker: string) => void update(selection.filter(item => item !== ticker), `${ticker} removida da seleção.`);
  return <div className="cmd-selection" ref={root}>
    <button type="button" className="cmd-count" aria-expanded={open} aria-haspopup="true" onClick={() => setOpen(value => !value)} title="Editar a seleção de ativos">
      <span>Seleção</span><b>{selection.length}/{MAX_SELECTION}</b><span className="visually-hidden">{selection.length === 1 ? "ativo selecionado" : "ativos selecionados"}</span>
    </button>
    <ul className="cmd-chips" ref={list} aria-label="Ativos selecionados">
      {selection.map((ticker, index) => <li key={ticker} className="chip" title={known.get(ticker) ?? ticker}>
        <i style={{ background: tickerColor(index) }} aria-hidden="true" />{ticker}
        <button type="button" aria-label={`Remover ${ticker}`} onClick={() => remove(ticker)}>×</button>
      </li>)}
      {!selection.length && <li className="cmd-chips-empty">vazia · use a busca</li>}
    </ul>
    {hidden > 0 && <button type="button" className="cmd-more" aria-expanded={open} onClick={() => setOpen(value => !value)} aria-label={`Mais ${hidden} ${hidden === 1 ? "ativo" : "ativos"} na seleção`}>+{hidden}</button>}
    {open && <div className="tray" role="group" aria-label="Editar seleção">
      <p className="tray-head">Seleção · {selection.length}/{MAX_SELECTION}</p>
      {selection.length ? <ol>{selection.map((ticker, index) => <li key={ticker}>
        <i style={{ background: tickerColor(index) }} aria-hidden="true" /><b>{ticker}</b><span>{known.get(ticker) ?? "Fora da lista B3"}</span>
        <button type="button" aria-label={`Remover ${ticker}`} onClick={() => remove(ticker)}>×</button>
      </li>)}</ol> : <p className="tray-empty">Nenhum ativo. Digite um ticker na busca (Ctrl+K) e pressione Enter.</p>}
      <div className="tray-foot">
        <button type="button" onClick={() => { setOpen(false); window.dispatchEvent(new Event("b3lab:focus-search")); }}>+ Adicionar</button>
        {selection.length > 0 && <button type="button" onClick={() => void update([], "Seleção limpa.")}>Limpar</button>}
      </div>
    </div>}
  </div>;
}

export default function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { session, setSelection } = useApp();
  const [market, setMarket] = useState<Market>({ strip: null, failed: false });
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);
  const [compact, setCompact] = useState(false);
  const [flash, setFlash] = useState<Flash | null>(null);
  useEffect(() => subscribeActivity(setBusy), []);
  useEffect(() => {
    const query = window.matchMedia("(max-width: 900px)");
    const sync = () => setCompact(query.matches);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);
  useEffect(() => {
    let active = true;
    const load = async () => {
      if (document.visibilityState === "hidden") return;
      try {
        // Plain fetch: background refreshes should not drive the shell's activity indicator.
        const response = await fetch("/api/market/strip", { credentials: "same-origin", cache: "no-store" });
        if (!response.ok) throw new Error(String(response.status));
        const strip: Strip = await response.json();
        if (active) setMarket({ strip, failed: false });
      } catch {
        if (active) setMarket(current => ({ ...current, failed: true }));
      }
    };
    const onVisible = () => { if (document.visibilityState === "visible") void load(); };
    void load();
    const timer = setInterval(load, refreshMs);
    document.addEventListener("visibilitychange", onVisible);
    return () => { active = false; clearInterval(timer); document.removeEventListener("visibilitychange", onVisible); };
  }, []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const typing = event.target instanceof Element && !!event.target.closest("input, textarea, select, [contenteditable='true']");
      const viewIndex = /^F[1-5]$/.test(event.key) ? Number(event.key.slice(1)) - 1 : event.altKey && !event.ctrlKey && !event.metaKey && /^Digit[1-5]$/.test(event.code) ? Number(event.code.slice(5)) - 1 : -1;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); window.dispatchEvent(new Event("b3lab:focus-search")); }
      else if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey && !event.altKey) { event.preventDefault(); window.dispatchEvent(new Event("b3lab:focus-search")); }
      else if (viewIndex >= 0 && !event.shiftKey) { event.preventDefault(); router.push(views[viewIndex].href); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [router]);
  useEffect(() => { setMenu(false); }, [pathname]);
  useEffect(() => {
    if (!flash) return;
    const timer = window.setTimeout(() => setFlash(null), flash.undo ? 9000 : 5000);
    return () => window.clearTimeout(timer);
  }, [flash]);
  const showFlash = useCallback((next: Flash) => setFlash(next), []);
  async function undo(before: string[]) {
    setFlash(null);
    try { await setSelection(before); setFlash({ text: "Seleção anterior restaurada.", tone: "info" }); }
    catch (reason) { setFlash({ text: reason instanceof Error ? reason.message : String(reason), tone: "negative" }); }
  }
  return <MarketContext value={market}>
    <div className="mesa" data-menu={menu ? "open" : undefined}>
      <a className="skip-link" href="#conteudo">Pular para o conteúdo</a>
      <header className="cmd">
        <Link href="/" className="cmd-brand" aria-label="B3Lab, Início"><span>B3</span>LAB<i aria-hidden="true" /></Link>
        <CommandSearch onFlash={showFlash} compact={compact} />
        <button type="button" className="cmd-menu" aria-expanded={menu} aria-controls="cmd-drawer" onClick={() => setMenu(value => !value)}>{menu ? "Fechar" : "Menu"}</button>
        <div className="cmd-drawer" id="cmd-drawer">
          <nav className="cmd-views" aria-label="Telas">{views.map(view => {
            const ready = !session || !view.ready || view.ready(session);
            return <Link key={view.href} href={view.href} aria-current={pathname === view.href ? "page" : undefined} aria-keyshortcuts={view.key} title={`${view.label} (${view.key} ou Alt+${view.key.slice(1)})${ready ? "" : ` · ${view.requirement}`}`}>
              <kbd>{view.key}</kbd>{view.label}{view.ready && <span className={`cmd-ready${ready ? " is-ready" : ""}`} aria-hidden="true" />}
              {!ready && <small className="cmd-req">{view.requirement}</small>}
            </Link>;
          })}</nav>
          <SelectionChips onFlash={showFlash} />
          <Clock />
        </div>
      </header>
      <Tape />
      <main id="conteudo" className="mesa-main" tabIndex={-1} aria-busy={busy}>{children}</main>
      <footer className="status">
        <span className={`status-activity${busy ? " is-busy" : ""}`}><span aria-hidden="true">{busy ? "◐" : "●"}</span>{busy ? "Processando" : "Pronto"}</span>
        <span className="status-flash" role="status" aria-live="polite">{flash && <span className={`is-${flash.tone}`}>{flash.text}{flash.undo && <button type="button" onClick={() => void undo(flash.undo!)}>Desfazer</button>}</span>}</span>
        <span className="status-note">Sessão anônima · resultados temporários</span>
        <span className="status-note">Dados para pesquisa, não recomendação de investimento</span>
      </footer>
    </div>
  </MarketContext>;
}
