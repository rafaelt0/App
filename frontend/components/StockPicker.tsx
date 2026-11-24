"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { searchUniverse } from "../lib/home";
import { ASSET_CLASS_FILTERS, assetBadge, rankStocks, sortStocks, synthesizeCustomCandidate } from "../lib/tickerSearch";
import { useApp, type AssetClass, type Stock } from "./AppProvider";
import StockLogo from "./StockLogo";

type Props = {
  selection: readonly string[];
  action: "Analisar" | "Adicionar";
  disabled?: boolean;
  onChoose: (ticker: string) => void;
};

export default function StockPicker({ selection, action, disabled, onChoose }: Props) {
  const { stocks, session, upsertStocks } = useApp();
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<"" | AssetClass>("");
  const [remote, setRemote] = useState<Stock[]>([]);
  const [open, setOpen] = useState(false);
  const [index, setIndex] = useState(-1);
  const ordered = useMemo(() => sortStocks(stocks, "liquidity"), [stocks]);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setRemote([]);
      return;
    }
    let active = true;
    const timer = window.setTimeout(() => {
      searchUniverse(trimmed, category).then(response => {
        if (!active) return;
        upsertStocks(response.items);
        setRemote(response.items.map(item => ({ ...item, liquidity: null })));
      }).catch(() => {});
    }, 220);
    return () => { active = false; window.clearTimeout(timer); };
  }, [query, category, upsertStocks]);

  const matches = useMemo(() => {
    const classFiltered = category ? ordered.filter(stock => (stock.asset_class ?? "acao_br") === category) : ordered;
    const available = classFiltered.filter(stock => !selection.includes(stock.ticker));
    const favorites = session?.watchlist ?? [];
    const pool = query.trim() ? available : [
      ...available.filter(stock => favorites.includes(stock.ticker)),
      ...available.filter(stock => !favorites.includes(stock.ticker)),
    ];
    const localMatches = rankStocks(pool, query, 10).items;
    const seen = new Set(localMatches.map(item => item.ticker));
    const merged = [...localMatches];
    for (const item of remote) {
      if (!seen.has(item.ticker) && !selection.includes(item.ticker) && (!category || item.asset_class === category)) {
        seen.add(item.ticker);
        merged.push(item);
      }
    }
    const custom = query.trim() ? synthesizeCustomCandidate(query) : null;
    if (custom && !seen.has(custom.ticker) && !selection.includes(custom.ticker) && (!category || custom.asset_class === category)) {
      merged.push({ ...custom, liquidity: null });
    }
    return merged.slice(0, 12);
  }, [ordered, query, category, remote, selection, session?.watchlist]);

  const active = Math.min(index, matches.length - 1);
  const expanded = open && matches.length > 0;

  useEffect(() => {
    function focus() { input.current?.focus(); input.current?.select(); }
    window.addEventListener("b3lab:focus-picker", focus);
    return () => window.removeEventListener("b3lab:focus-picker", focus);
  }, []);

  function move(next: number) {
    const position = Math.max(0, Math.min(next, matches.length - 1));
    setOpen(true);
    setIndex(position);
    list.current?.children[position]?.scrollIntoView({ block: "nearest" });
  }

  function choose(stock: Stock) {
    if (disabled) return;
    upsertStocks([stock]);
    setIndex(-1);
    setQuery("");
    onChoose(stock.ticker);
  }

  function openPicker() {
    if (!disabled) setOpen(true);
  }

  return <div className="stock-picker" onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
  }}>
    <label htmlFor={id}>{action === "Analisar" ? "Qual ativo você quer analisar?" : "Adicionar ativos"}</label>
    <input ref={input} id={id} type="search" role="combobox" autoComplete="off" spellCheck={false} readOnly={disabled} aria-disabled={disabled}
      aria-autocomplete="list" aria-expanded={expanded} aria-controls={expanded ? `${id}-matches` : undefined}
      aria-activedescendant={expanded && active >= 0 ? `${id}-option-${active}` : undefined}
      placeholder="Código, empresa ou classe, ex. PETR4, AAPL, HGLG11, O, TLT, CDI" value={query}
      onFocus={openPicker} onClick={openPicker}
      onChange={event => { setQuery(event.target.value); setOpen(true); setIndex(-1); }}
      onKeyDown={event => {
        if (event.key === "Escape" && open) { event.preventDefault(); setOpen(false); setIndex(-1); return; }
        if (disabled) return;
        if (event.key === "ArrowDown" && matches.length) { event.preventDefault(); move(open ? active + 1 : 0); }
        else if (event.key === "ArrowUp" && matches.length) { event.preventDefault(); move(active < 0 ? matches.length - 1 : active - 1); }
        else if (event.key === "Enter" && expanded && (active >= 0 || query.trim())) {
          event.preventDefault();
          choose(matches[Math.max(0, active)]);
        }
      }} />
    {open && <div className="stock-picker-dropdown">
      <div className="picker-cats" role="group" aria-label="Filtrar por classe de ativo">
        {ASSET_CLASS_FILTERS.map(item => <button type="button" key={item.key || "all"} aria-pressed={category === item.key}
          onMouseDown={event => event.preventDefault()} onClick={() => { setCategory(item.key); setIndex(-1); input.current?.focus(); }}>{item.label}</button>)}
      </div>
      {expanded ? <ul ref={list} className="stock-picker-options" id={`${id}-matches`} role="listbox" aria-label="Ativos encontrados">
        {matches.map((stock, position) => <li key={stock.ticker} id={`${id}-option-${position}`} role="option" aria-selected={active === position} aria-disabled={disabled || undefined}
          onMouseDown={event => event.preventDefault()} onClick={() => choose(stock)}>
          <StockLogo ticker={stock.ticker} company={stock.company} size={24} />
          <span className="stock-picker-company"><b>{stock.ticker} <small className="asset-badge">{assetBadge(stock)}</small></b><span>{stock.company || stock.sector || "Ativo"}</span></span>
          <span className="stock-picker-action">{action}</span>
        </li>)}
      </ul> : <p className="stock-picker-options stock-picker-empty" role="status">Nenhum ativo disponível para essa busca.</p>}
    </div>}
  </div>;
}
