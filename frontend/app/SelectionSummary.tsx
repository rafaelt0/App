"use client";
import { useEffect, useRef } from "react";

export default function SelectionSummary({ tickers, onEdit }: { tickers: string[]; onEdit: () => void }) {
  const button = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const active = document.activeElement;
    if (!active || active === document.body) button.current?.focus();
  }, []);
  return <div className="home-selection-summary">
    <p className="home-selection-list">
      <span className="home-selection-label">Ações analisadas</span>
      <span className="home-selection-tickers">{tickers.join(", ")}</span>
      <span className="home-selection-count">{tickers.length} {tickers.length === 1 ? "ativo" : "ativos"}</span>
    </p>
    <button ref={button} type="button" onClick={onEdit}>Editar seleção</button>
  </div>;
}
