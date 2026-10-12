"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { api, subscribeActivity } from "../lib/api";
import CommandPalette, { SELECTION_FULL_MESSAGE } from "./CommandPalette";
import MarketStrip from "./MarketStrip";
import { Icon, Loading, Notice } from "./ui";

export type Stock = { ticker: string; company: string; sector: string; liquidity: number | null };
export type Session = {
  selection: string[];
  revision: number;
  watchlist: string[];
  portfolio_ready: boolean;
  simulation_ready: boolean;
  portfolio_id?: string | null;
  restored: boolean;
  restoration?: { original_count: number; invalid_count: number; truncated_count: number } | null;
};
export type SelectionUpdate = string[] | ((current: readonly string[]) => string[]);
type AppContext = {
  session: Session | null;
  stocks: Stock[];
  loading: boolean;
  error: string | null;
  setSelection: (tickers: SelectionUpdate) => Promise<void>;
  refreshSession: () => Promise<void>;
  toggleFavorite: (ticker: string) => Promise<void>;
};
const Context = createContext<AppContext | null>(null);
let bootstrap: Promise<{ session: Session; stocks: Stock[] }> | null = null;

function initialize() {
  if (!bootstrap) {
    bootstrap = Promise.all([api<Session>("/api/session"), api<{ stocks: Omit<Stock, "liquidity">[] }>("/api/universe")]).then(([session, universe]) => {
      return { session, stocks: universe.stocks.map(stock => ({ ...stock, liquidity: null })) };
    }).catch(error => { bootstrap = null; throw error; });
  }
  return bootstrap;
}

export function useApp() {
  const context = useContext(Context);
  if (!context) throw new Error("B3Lab AppProvider is missing");
  return context;
}

const navigation = [
  { href: "/", label: "Fundamentos", icon: "home" },
  { href: "/portfolio", label: "Portfólio", icon: "portfolio" },
  { href: "/simulacao", label: "Simulação", icon: "simulation" },
  { href: "/noticias", label: "Notícias", icon: "news" },
  { href: "/screener", label: "Screener", icon: "filter" },
];

function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { session, stocks, setSelection } = useApp();
  const currentPage = navigation.find(item => item.href === pathname);
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [palette, setPalette] = useState(false);
  const [busy, setBusy] = useState(false);
  const [shellError, setShellError] = useState<string | null>(null);
  const [shortcut, setShortcut] = useState("Ctrl K");
  useEffect(() => subscribeActivity(setBusy), []);
  useEffect(() => {
    if (/Mac|iPhone|iPad/.test(navigator.platform)) setShortcut("⌘K");
    const onKey = (event: KeyboardEvent) => {
      const typing = event.target instanceof Element && !!event.target.closest("input, textarea, select, [contenteditable='true']");
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); setPalette(value => !value); }
      else if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey && !event.altKey) { event.preventDefault(); setPalette(true); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);
  const selection = session?.selection ?? [];
  const watchlist = session?.watchlist ?? [];
  const known = new Set(stocks.map(stock => stock.ticker));
  const openTicker = useCallback((ticker: string) => {
    setOpen(false);
    if (selection.includes(ticker)) { router.push("/"); return; }
    if (selection.length >= 20) { setShellError(SELECTION_FULL_MESSAGE); return; }
    setShellError(null);
    void setSelection(current => current.includes(ticker) ? [...current] : [...current, ticker])
      .then(() => router.push("/"))
      .catch(exc => setShellError(exc instanceof Error ? exc.message : String(exc)));
  }, [router, selection, setSelection]);
  const openPage = useCallback((href: string) => { setOpen(false); router.push(href); }, [router]);
  const menuRef = useRef<HTMLButtonElement>(null);
  const navRef = useRef<HTMLElement>(null);
  const closeMenu = () => {
    setOpen(false);
    if (navRef.current?.contains(document.activeElement) && window.matchMedia("(max-width: 1023px)").matches) menuRef.current?.focus();
  };
  useLayoutEffect(() => {
    setCollapsed(document.documentElement.dataset.sidebar === "collapsed");
  }, []);
  const toggleCollapsed = () => {
    const next = !collapsed;
    setCollapsed(next);
    if (next) document.documentElement.dataset.sidebar = "collapsed";
    else delete document.documentElement.dataset.sidebar;
    localStorage.setItem("b3lab-sidebar", next ? "collapsed" : "expanded");
  };
  useEffect(() => {
    setOpen(false);
    setShellError(null);
    if (navRef.current?.contains(document.activeElement) && window.matchMedia("(max-width: 1023px)").matches) menuRef.current?.focus();
  }, [pathname]);
  useEffect(() => {
    if (!open) return;
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      if (navRef.current?.contains(event.target as Node)) menuRef.current?.focus();
    };
    document.addEventListener("keydown", onEscape);
    return () => document.removeEventListener("keydown", onEscape);
  }, [open]);
  return <div className="app-shell">
    <a className="skip-link" href="#conteudo">Pular para o conteúdo</a>
    <header className="mobile-header"><Link href="/" aria-label="B3Lab, página inicial" onClick={closeMenu}><img src="/logo.svg" alt="B3Lab" width="142" height="44" /></Link><div className="mobile-actions"><button type="button" aria-label="Buscar ativo ou página" onClick={() => setPalette(true)}><Icon name="search" /></button><button ref={menuRef} aria-expanded={open} aria-controls="navegacao" onClick={() => setOpen(!open)}><Icon name="menu" />Menu</button></div></header>
    <aside ref={navRef} className={`sidebar${open ? " is-open" : ""}`} id="navegacao">
      <Link href="/" className="brand" aria-label="B3Lab, página inicial" onClick={closeMenu}><img src="/logo.svg" alt="B3Lab" width="180" height="56" /></Link>
      <button type="button" className="sidebar-search" aria-label="Buscar ativo ou página" aria-keyshortcuts="Control+K Meta+K /" title={collapsed ? "Buscar" : undefined} onClick={() => setPalette(true)}><Icon name="search" /><span>Buscar</span><kbd>{shortcut}</kbd></button>
      <nav aria-label="Navegação principal">{navigation.map(item => <Link key={item.href} href={item.href} aria-label={item.label} title={collapsed ? item.label : undefined} aria-current={pathname === item.href ? "page" : undefined} onClick={() => { if (open) closeMenu(); }}><Icon name={item.icon} /><span>{item.label}</span></Link>)}</nav>
      {session && <section className="sidebar-watchlist" aria-labelledby="sidebar-watchlist-title">
        <h2 className="sidebar-section-label" id="sidebar-watchlist-title">Favoritos</h2>
        {watchlist.length ? <ul>{watchlist.map(ticker => <li key={ticker}><button type="button" aria-disabled={!known.has(ticker)} title={known.has(ticker) ? `Adicionar ${ticker} à seleção de Fundamentos` : `${ticker} está fora da lista B3`} onClick={() => { if (known.has(ticker)) openTicker(ticker); }}><span>{ticker}</span>{!known.has(ticker) && <small>Fora da lista B3</small>}<Icon name="arrow-right" /></button></li>)}</ul>
          : <p className="sidebar-watchlist-empty">Use a estrela na busca de ações para guardar ativos aqui.</p>}
      </section>}
      <div className="sidebar-footer">
        <p className="sidebar-note"><strong>Análise quantitativa da B3</strong><span>Dados para pesquisa, não recomendações de investimento.</span></p>
        <button type="button" className="sidebar-toggle" aria-expanded={!collapsed} aria-controls="navegacao" aria-label={collapsed ? "Expandir barra lateral" : "Recolher barra lateral"} title={collapsed ? "Expandir barra lateral" : "Recolher barra lateral"} onClick={toggleCollapsed}><Icon name={collapsed ? "panel-open" : "panel-close"} />{!collapsed && <span>Recolher</span>}</button>
      </div>
    </aside>
    <main id="conteudo" className="main-frame" tabIndex={-1} aria-busy={busy}>
      <div className="workspace-bar"><div className="workspace-location"><span>B3Lab</span><span aria-hidden="true">/</span><span>{currentPage?.label ?? "Pesquisa de ações"}</span></div>
        <div className="workspace-tools">{session && <span className="workspace-selection" data-empty={!selection.length}>{selection.length ? `${selection.length} ${selection.length === 1 ? "ativo selecionado" : "ativos selecionados"}` : "Nenhum ativo selecionado"}</span>}<span className="workspace-market">Mercado brasileiro <span>B3</span></span></div>
        <span className="activity-line" data-active={busy} aria-hidden="true" />
      </div>
      <MarketStrip />
      <div className="main-content">{shellError && <Notice tone="negative">{shellError}</Notice>}{children}<footer className="mobile-disclaimer">Análise quantitativa da B3.<br />Dados para pesquisa, não recomendações de investimento.</footer></div>
    </main>
    <CommandPalette open={palette} onClose={() => setPalette(false)} pages={navigation} stocks={stocks} watchlist={watchlist} selection={selection} onPage={openPage} onTicker={openTicker} />
  </div>;
}

export default function AppProvider({ children }: { children: ReactNode }) {
  const [current, setCurrent] = useState<Session | null>(null);
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<Session | null>(null);
  const queue = useRef(Promise.resolve());
  const enqueue = useCallback((operation: () => Promise<void>) => {
    const next = queue.current.then(operation);
    queue.current = next.catch(() => {});
    return next;
  }, []);
  const accept = useCallback((value: Session) => { sessionRef.current = value; setCurrent(value); }, []);
  useEffect(() => {
    let active = true;
    initialize().then(value => {
      if (active) { accept(value.session); setStocks(value.stocks); setLoading(false); }
    }).catch(reason => { if (active) { setError(reason instanceof Error ? reason.message : String(reason)); setLoading(false); } });
    return () => { active = false; };
  }, [accept]);
  useEffect(() => {
    if (loading || error) return;
    let active = true;
    api<{ liquidity: Record<string, number | null> }>("/api/universe/liquidity").then(value => {
      if (active) setStocks(current => current.map(stock => ({ ...stock, liquidity: value.liquidity[stock.ticker] ?? null })));
    }).catch(() => {});
    return () => { active = false; };
  }, [loading, error]);
  const refreshSession = useCallback(() => enqueue(async () => {
    accept(await api<Session>("/api/session"));
  }), [accept, enqueue]);
  const setSelection = useCallback((update: SelectionUpdate) => enqueue(async () => {
    const state = sessionRef.current;
    if (!state) throw new Error("A sessão ainda não foi carregada.");
    const tickers = typeof update === "function" ? update(state.selection) : update;
    accept(await api<Session>("/api/session/selection", { tickers, revision: state.revision }, "PUT"));
  }), [accept, enqueue]);
  const toggleFavorite = useCallback((ticker: string) => enqueue(async () => {
    const state = sessionRef.current;
    if (!state) throw new Error("A sessão ainda não foi carregada.");
    const result = await api<{ watchlist: string[] }>(`/api/watchlist/${encodeURIComponent(ticker)}`, undefined, state.watchlist.includes(ticker) ? "DELETE" : "PUT");
    accept({ ...state, watchlist: result.watchlist });
  }), [accept, enqueue]);
  async function retry() {
    setLoading(true); setError(null); bootstrap = null;
    try { const value = await initialize(); accept(value.session); setStocks(value.stocks); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setLoading(false); }
  }
  return <Context value={{ session: current, stocks, loading, error, setSelection, refreshSession, toggleFavorite }}>
    <Shell>{loading ? <Loading>Carregando a sessão e a lista de ativos.</Loading> : error ? <Notice tone="negative">{error}<button onClick={retry}>Tentar novamente</button></Notice> : children}</Shell>
  </Context>;
}
