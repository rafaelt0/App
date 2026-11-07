"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useApp } from "./AppProvider";
import StockLogo from "./StockLogo";
import { Icon } from "./ui";

export default function SelectionMenu() {
  const { session, stocks, setSelection } = useApp();
  const id = useId();
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selection = session?.selection ?? [];
  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); } };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("pointerdown", onPointer); document.removeEventListener("keydown", onKey); };
  }, [open]);
  if (!session) return null;
  const companies = new Map(stocks.map(stock => [stock.ticker, stock.company]));
  async function update(next: (current: readonly string[]) => string[]) {
    setBusy(true); setError(null);
    try { await setSelection(next); } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setBusy(false); }
  }
  const label = selection.length ? `${selection.length} ${selection.length === 1 ? "ativo selecionado" : "ativos selecionados"}` : "Nenhum ativo selecionado";
  return <div className="selection-menu" ref={root}>
    <button ref={trigger} type="button" className="workspace-selection" data-empty={!selection.length} aria-expanded={open} aria-controls={id} aria-haspopup="true" onClick={() => setOpen(value => !value)}>{label}<Icon name="chevron-down" /></button>
    {open && <div className="selection-popover" id={id} role="group" aria-label="Seleção de ativos">
      {selection.length ? <ul>{selection.map(ticker => <li key={ticker}>
        <StockLogo ticker={ticker} company={companies.get(ticker) ?? ""} size={20} />
        <strong>{ticker}</strong>
        <button type="button" disabled={busy} aria-label={`Remover ${ticker}`} onClick={() => void update(current => current.filter(item => item !== ticker))}>×</button>
      </li>)}</ul> : <p className="caption">Use a busca (Ctrl+K) para adicionar ativos.</p>}
      {error && <p className="scr-field-error" role="alert">{error}</p>}
      {selection.length > 0 && <div className="selection-popover-footer"><span className="caption">{selection.length}/20</span><button type="button" disabled={busy} onClick={() => void update(() => [])}>Limpar</button></div>}
    </div>}
  </div>;
}
