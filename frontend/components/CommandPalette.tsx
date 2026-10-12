"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { Stock } from "./AppProvider";
import { Icon, Notice } from "./ui";

export const SELECTION_FULL_MESSAGE = "Seleção cheia: remova um ativo para adicionar outro.";
export type PaletteAction = { id: string; group: string; label: string; detail?: string; icon?: string; ticker?: string; hint?: string; blocked?: string; run: () => void };

type Props = {
  open: boolean;
  onClose: () => void;
  pages: { href: string; label: string; icon: string }[];
  stocks: Stock[];
  watchlist: string[];
  selection: string[];
  onPage: (href: string) => void;
  onTicker: (ticker: string) => void;
};

const normalize = (value: string) => value.normalize("NFD").replace(/[̀-ͯ]/g, "").toLocaleLowerCase("pt-BR");

export default function CommandPalette({ open, onClose, pages, stocks, watchlist, selection, onPage, onTicker }: Props) {
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    const node = dialog.current;
    if (!node) return;
    if (open && !node.open) { setQuery(""); setActive(0); setMessage(null); node.showModal(); input.current?.focus(); }
    if (!open && node.open) node.close();
  }, [open]);

  const actions = useMemo<PaletteAction[]>(() => {
    const needle = normalize(query.trim());
    const full = selection.length >= 20;
    const tickerAction = (stock: Stock, group: string): PaletteAction => ({
      id: `${group}-${stock.ticker}`, group, ticker: stock.ticker, label: stock.company || stock.ticker,
      detail: stock.sector || "Setor não informado",
      hint: selection.includes(stock.ticker) ? "Na seleção" : full ? "Seleção cheia" : "Adicionar",
      blocked: full && !selection.includes(stock.ticker) ? SELECTION_FULL_MESSAGE : undefined,
      run: () => onTicker(stock.ticker),
    });
    const pageActions = pages.filter(page => !needle || normalize(page.label).includes(needle)).map(page => ({
      id: `page-${page.href}`, group: "Páginas", label: page.label, icon: page.icon, hint: "Abrir", run: () => onPage(page.href),
    }));
    const byTicker = new Map(stocks.map(stock => [stock.ticker, stock]));
    if (!needle) {
      const favorites = watchlist.map(ticker => byTicker.get(ticker)).filter((stock): stock is Stock => !!stock).slice(0, 6).map(stock => tickerAction(stock, "Favoritos"));
      return [...pageActions, ...favorites];
    }
    const ranked = stocks.map(stock => {
      const ticker = normalize(stock.ticker);
      const company = normalize(stock.company);
      const rank = ticker === needle ? 0 : ticker.startsWith(needle) ? 1 : company.startsWith(needle) ? 2
        : company.split(/[^a-z0-9]+/).some(word => word.startsWith(needle)) ? 3 : company.includes(needle) ? 4 : normalize(stock.sector).includes(needle) ? 5 : -1;
      return { stock, rank };
    }).filter(item => item.rank >= 0).sort((a, b) => a.rank - b.rank).slice(0, 8);
    return [...ranked.map(item => tickerAction(item.stock, "Ativos")), ...pageActions];
  }, [query, pages, stocks, watchlist, selection, onPage, onTicker]);

  const current = Math.min(active, Math.max(actions.length - 1, 0));
  function choose(action: PaletteAction | undefined) {
    if (!action) return;
    if (action.blocked) { setMessage(action.blocked); return; }
    onClose();
    action.run();
  }
  useEffect(() => {
    document.getElementById(`${id}-option-${current}`)?.scrollIntoView({ block: "nearest" });
  }, [current, id]);

  let lastGroup = "";
  return <dialog ref={dialog} className="palette" aria-label="Busca rápida" onClose={onClose} onClick={event => { if (event.target === dialog.current) onClose(); }}>
    <div className="palette-search">
      <Icon name="search" />
      <input ref={input} type="text" aria-label="Buscar ativo ou página" role="combobox" aria-expanded={actions.length > 0} aria-controls={`${id}-list`} aria-activedescendant={actions.length ? `${id}-option-${current}` : undefined} aria-autocomplete="list" autoComplete="off" spellCheck={false}
        placeholder="Buscar ticker, empresa ou página…" value={query}
        onChange={event => { setQuery(event.target.value); setActive(0); }}
        onKeyDown={event => {
          if (event.key === "ArrowDown") { event.preventDefault(); setActive(Math.min(current + 1, actions.length - 1)); }
          if (event.key === "ArrowUp") { event.preventDefault(); setActive(Math.max(current - 1, 0)); }
          if (event.key === "Enter") { event.preventDefault(); choose(actions[current]); }
        }} />
      <kbd>Esc</kbd>
    </div>
    <ul className="palette-list" id={`${id}-list`} role="listbox" aria-label="Resultados">
      {actions.map((action, index) => {
        const heading = action.group !== lastGroup ? action.group : null;
        lastGroup = action.group;
        return <li key={action.id} role="presentation">
          {heading && <p className="palette-group">{heading}</p>}
          <div id={`${id}-option-${index}`} role="option" aria-selected={index === current} aria-label={`${action.ticker ? `${action.ticker}, ` : ""}${action.label}${action.hint ? `, ${action.hint}` : ""}`} className="palette-option"
            onMouseMove={() => { if (index !== current) setActive(index); }} onMouseDown={event => event.preventDefault()} onClick={() => choose(action)}>
            {action.ticker ? <span className="palette-ticker">{action.ticker}</span> : <span className="palette-icon"><Icon name={action.icon} /></span>}
            <span className="palette-label">{action.label}{action.detail && <small>{action.detail}</small>}</span>
            {action.hint && <span className="palette-hint">{action.hint}</span>}
          </div>
        </li>;
      })}
      {!actions.length && <li className="palette-empty" role="presentation">Nenhum ativo ou página para “{query}”.</li>}
    </ul>
    <div className="visually-hidden" aria-live="polite">{actions.length === 1 ? "1 resultado" : `${actions.length} resultados`}</div>
    {message && <div role="status"><Notice tone="warning">{message}</Notice></div>}
    <p className="palette-footer" aria-hidden="true"><span><kbd>↑</kbd><kbd>↓</kbd> navegar</span><span><kbd>Enter</kbd> abrir</span><span><kbd>/</kbd> abre a busca</span><span>Ativos entram na seleção de Fundamentos</span></p>
  </dialog>;
}
