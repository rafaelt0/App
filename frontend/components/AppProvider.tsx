"use client";

import { usePathname } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "../lib/api";
import QuickGuide from "./QuickGuide";
import Shell from "./Shell";
import { Notice } from "./ui";

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
export const MAX_SELECTION = 20;
export const SELECTION_FULL_MESSAGE = "Seleção cheia: remova um ativo para adicionar outro.";
const Context = createContext<AppContext | null>(null);
let bootstrap: Promise<{ session: Session; stocks: Stock[] }> | null = null;
let liquidity: Promise<Record<string, number | null>> | null = null;

function loadLiquidity() {
  if (!liquidity) {
    // Previewed alongside the session so the heatmap's sizes do not wait an extra round trip.
    liquidity = api<{ liquidity: Record<string, number | null> }>("/api/universe/liquidity").then(value => value.liquidity);
    // A failed preview must not be memoized, or the heatmap would stay size-less for the whole visit.
    liquidity.catch(() => { liquidity = null; });
  }
  return liquidity;
}

function initialize() {
  loadLiquidity();
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

export default function AppProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
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
    loadLiquidity().then(values => {
      if (active) setStocks(current => current.map(stock => ({ ...stock, liquidity: values[stock.ticker] ?? null })));
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
  let content = children;
  if (pathname !== "/guia") {
    if (loading) content = <QuickGuide loading />;
    else if (error) content = <Notice tone="negative">{error}<button onClick={retry}>Tentar novamente</button></Notice>;
  }
  return <Context value={{ session: current, stocks, loading, error, setSelection, refreshSession, toggleFavorite }}>
    <Shell>{content}</Shell>
  </Context>;
}
