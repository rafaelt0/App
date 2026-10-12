"use client";

import { useEffect, useState } from "react";
import { formatBrasilia, formatValue, type Unit } from "./ui";

type StripItem = { key: string; group: "macro" | "market" | "stock"; label: string; symbol: string | null; value: number; unit: Unit; currency: string | null; digits: number; change: number | null; note: string | null; as_of: string; source: string };
type Strip = { items: StripItem[]; missing: string[]; updated_at: string };

const refreshMs = 5 * 60 * 1000;
const dateOnly = new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "2-digit", year: "numeric", timeZone: "UTC" });
/** Reference dates are calendar dates from the source, so they are formatted without a time zone shift. */
const referenceDate = (value: string) => dateOnly.format(new Date(`${value}T00:00:00Z`));

function Change({ value }: { value: number | null }) {
  if (value === null) return null;
  const tone = value > 0 ? "positive" : value < 0 ? "negative" : "neutral";
  const glyph = value > 0 ? "▲" : value < 0 ? "▼" : "•";
  return <span className={`strip-change is-${tone}`}><span aria-hidden="true">{glyph}</span>{formatValue(value, "fraction", 2, "BRL", true)}</span>;
}

function Items({ data, hidden }: { data: Strip; hidden?: boolean }) {
  return <ul className="strip-list" aria-hidden={hidden || undefined}>
    {data.items.map(item => <li key={item.key} className={`strip-item is-${item.group}${item.symbol?.endsWith(".SA") ? " is-ticker" : ""}`} title={`${item.label}: ${item.source}, referência ${referenceDate(item.as_of)}`}>
      <span className="strip-label">{item.label}{item.note && <small>{item.note}</small>}</span>
      <span className="strip-value">{formatValue(item.value, item.unit, item.digits, item.currency ?? "BRL")}</span>
      <Change value={item.change} />
    </li>)}
    <li className="strip-item strip-source">Cotações do Yahoo Finance com atraso; indicadores do BCB. Atualizado em {formatBrasilia(data.updated_at, false)}.{data.missing.length > 0 && ` Indisponível: ${data.missing.join(", ")}.`}</li>
  </ul>;
}

/** Scrolling strip of delayed B3 quotes and BCB macro indicators, each with its source and reference date. */
export default function MarketStrip() {
  const [data, setData] = useState<Strip | null>(null);
  const [failed, setFailed] = useState(false);
  const [paused, setPaused] = useState(false);
  useEffect(() => {
    let active = true;
    const load = async () => {
      if (document.visibilityState === "hidden") return;
      try {
        // Plain fetch: background refreshes should not drive the shell's activity line.
        const response = await fetch("/api/market/strip", { credentials: "same-origin", cache: "no-store" });
        if (!response.ok) throw new Error(String(response.status));
        const next: Strip = await response.json();
        if (active) { setData(next); setFailed(false); }
      } catch {
        if (active) setFailed(true);
      }
    };
    load();
    const timer = setInterval(load, refreshMs);
    return () => { active = false; clearInterval(timer); };
  }, []);
  if (!data?.items.length) {
    return <section className="market-strip is-empty" aria-label="Indicadores de mercado">
      <p>{failed || data ? "Indicadores de mercado indisponíveis no momento." : "Carregando indicadores de mercado."}</p>
    </section>;
  }
  return <section className="market-strip" aria-label="Indicadores de mercado" data-paused={paused}>
    <div className="strip-viewport">
      <div className="strip-track" style={{ animationDuration: `${data.items.length * 4}s` }}>
        <Items data={data} />
        <Items data={data} hidden />
      </div>
    </div>
    <button type="button" className="strip-toggle" aria-pressed={paused} aria-label={paused ? "Retomar rolagem dos indicadores" : "Pausar rolagem dos indicadores"} title={paused ? "Retomar" : "Pausar"} onClick={() => setPaused(!paused)}>
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" fill="currentColor">{paused ? <path d="M3 1.5v9l7-4.5z" /> : <path d="M2.5 1.5h2.5v9H2.5zm4.5 0h2.5v9H7z" />}</svg>
    </button>
  </section>;
}
