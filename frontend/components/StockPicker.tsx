"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { rankStocks, sortStocks } from "../lib/tickerSearch";
import { useApp } from "./AppProvider";
import StockLogo from "./StockLogo";

type Props = {
  selection: readonly string[];
  action: "Analisar" | "Adicionar";
  disabled?: boolean;
  onChoose: (ticker: string) => void;
};

export default function StockPicker({ selection, action, disabled, onChoose }: Props) {
  const { stocks, session } = useApp();
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [index, setIndex] = useState(-1);
  const ordered = useMemo(() => sortStocks(stocks, "liquidity"), [stocks]);
  const matches = useMemo(() => {
    const available = ordered.filter(stock => !selection.includes(stock.ticker));
    const favorites = session?.watchlist ?? [];
    const pool = query.trim() ? available : [
      ...available.filter(stock => favorites.includes(stock.ticker)),
      ...available.filter(stock => !favorites.includes(stock.ticker)),
    ];
    return rankStocks(pool, query, 8).items;
  }, [ordered, query, selection, session?.watchlist]);
  const active = Math.min(index, matches.length - 1);
  const visible = open && !disabled;
  const expanded = visible && matches.length > 0;

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

  function choose(ticker: string) {
    setOpen(false);
    setIndex(-1);
    setQuery("");
    onChoose(ticker);
  }

  return <div className="stock-picker" onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
  }}>
    <label htmlFor={id}>{action === "Analisar" ? "Qual ação você quer analisar?" : "Adicionar ações"}</label>
    <input ref={input} id={id} type="search" role="combobox" autoComplete="off" spellCheck={false} readOnly={disabled} aria-disabled={disabled}
      aria-autocomplete="list" aria-expanded={expanded} aria-controls={expanded ? `${id}-matches` : undefined}
      aria-activedescendant={expanded && active >= 0 ? `${id}-option-${active}` : undefined}
      placeholder="Código ou empresa, ex. PETR4" value={query}
      onFocus={() => setOpen(true)} onClick={() => setOpen(true)}
      onChange={event => { setQuery(event.target.value); setOpen(true); setIndex(-1); }}
      onKeyDown={event => {
        if (disabled) return;
        if (event.key === "ArrowDown" && matches.length) { event.preventDefault(); move(visible ? active + 1 : 0); }
        else if (event.key === "ArrowUp" && matches.length) { event.preventDefault(); move(active < 0 ? matches.length - 1 : active - 1); }
        else if (event.key === "Enter" && expanded && (active >= 0 || query.trim())) {
          event.preventDefault();
          choose(matches[Math.max(0, active)].ticker);
        } else if (event.key === "Escape" && visible) { event.preventDefault(); setOpen(false); setIndex(-1); }
      }} />
    {visible && (expanded ? <ul ref={list} className="stock-picker-options" id={`${id}-matches`} role="listbox" aria-label="Ações encontradas">
      {matches.map((stock, position) => <li key={stock.ticker} id={`${id}-option-${position}`} role="option" aria-selected={active === position}
        onMouseDown={event => event.preventDefault()} onClick={() => choose(stock.ticker)}>
        <StockLogo ticker={stock.ticker} company={stock.company} size={24} />
        <span className="stock-picker-company"><b>{stock.ticker}</b><span>{stock.company || stock.sector || "Empresa não informada"}</span></span>
        <span className="stock-picker-action">{action}</span>
      </li>)}
    </ul> : <p className="stock-picker-options stock-picker-empty" role="status">Nenhuma ação disponível para essa busca.</p>)}
  </div>;
}
