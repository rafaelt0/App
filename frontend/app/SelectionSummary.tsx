"use client";
import { useEffect, useRef } from "react";

const VISIBLE = 4;

export default function SelectionSummary({ tickers, onEdit }: { tickers: string[]; onEdit: () => void }) {
  const button = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const active = document.activeElement;
    if (!active || active === document.body) button.current?.focus();
  }, []);
  const rest = tickers.slice(VISIBLE);
  return <div className="home-selection-summary">
    <p className="home-selection-list">
      <span className="home-selection-label">Ações analisadas</span>
      <span className="home-selection-tickers">{tickers.slice(0, VISIBLE).join(", ")}{!!rest.length && <> <span className="home-selection-more" title={rest.join(", ")}>+{rest.length}</span></>}</span>
      <span className="home-selection-count">{tickers.length} {tickers.length === 1 ? "ativo" : "ativos"}</span>
    </p>
    <button ref={button} type="button" onClick={onEdit}>Editar seleção</button>
  </div>;
}
