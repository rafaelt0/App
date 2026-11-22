"use client";

import { useEffect, useRef, useState } from "react";
import { comparisonColors } from "../lib/home";
import { idleCursor, type CursorState } from "./TutorialCursor";

export type DemoMode = "demo" | "paused" | "interactive";

export type SceneProps = {
  mode: DemoMode;
  replaySeq: number;
  beatJump: { beat: number; seq: number } | null;
  onBeatChange: (beat: number) => void;
  onCursorChange: (cursor: CursorState) => void;
  onLayoutTick: () => void;
  onUserInteract: () => void;
};

type DemoStock = {
  ticker: string;
  company: string;
  sector: string;
  dyBand: "alto" | "medio" | "baixo";
  roeBand: "alto" | "medio";
  plBand: "baixo" | "medio" | "alto";
  /** Relative bar lengths (0..100) for schematic illustration only. */
  bars: { roe: number; pl: number; dy: number };
  /** Schematic target range positions (0..100) for illustration only. */
  range: { low: number; high: number; mean: number; price: number; positive: boolean };
};

const DEMO_STOCKS: DemoStock[] = [
  {
    ticker: "PETR4",
    company: "Petrobras",
    sector: "Petróleo e Gás",
    dyBand: "alto",
    roeBand: "alto",
    plBand: "baixo",
    bars: { roe: 78, pl: 34, dy: 92 },
    range: { low: 26, high: 84, mean: 64, price: 48, positive: true },
  },
  {
    ticker: "VALE3",
    company: "Vale",
    sector: "Mineração",
    dyBand: "alto",
    roeBand: "alto",
    plBand: "baixo",
    bars: { roe: 68, pl: 42, dy: 76 },
    range: { low: 30, high: 88, mean: 68, price: 54, positive: true },
  },
  {
    ticker: "BBAS3",
    company: "Banco do Brasil",
    sector: "Bancos",
    dyBand: "alto",
    roeBand: "alto",
    plBand: "baixo",
    bars: { roe: 72, pl: 30, dy: 82 },
    range: { low: 28, high: 82, mean: 62, price: 50, positive: true },
  },
  {
    ticker: "ITUB4",
    company: "Itaú Unibanco",
    sector: "Bancos",
    dyBand: "medio",
    roeBand: "alto",
    plBand: "medio",
    bars: { roe: 75, pl: 52, dy: 56 },
    range: { low: 34, high: 86, mean: 66, price: 58, positive: true },
  },
  {
    ticker: "WEGE3",
    company: "WEG",
    sector: "Bens Industriais",
    dyBand: "baixo",
    roeBand: "alto",
    plBand: "alto",
    bars: { roe: 90, pl: 86, dy: 24 },
    range: { low: 22, high: 78, mean: 52, price: 60, positive: false },
  },
  {
    ticker: "ABEV3",
    company: "Ambev",
    sector: "Bebidas",
    dyBand: "medio",
    roeBand: "medio",
    plBand: "medio",
    bars: { roe: 54, pl: 58, dy: 58 },
    range: { low: 24, high: 74, mean: 50, price: 46, positive: true },
  },
];

const demoColor = (index: number) => comparisonColors[index % comparisonColors.length];

/** Helper for building abortable choreographed timelines cleanly inside useEffect. */
function createRunner(
  onCursorChange: (cursor: CursorState) => void,
  onLayoutTick: () => void,
) {
  let cancelled = false;
  let clickSeq = 0;
  const timers: number[] = [];
  let currentCursor: CursorState = { ...idleCursor };

  const emitCursor = (next: CursorState) => {
    if (cancelled) return;
    currentCursor = next;
    onCursorChange(next);
  };

  const wait = (ms: number) =>
    new Promise<boolean>(resolve => {
      if (cancelled) {
        resolve(false);
        return;
      }
      const id = window.setTimeout(() => resolve(!cancelled), ms);
      timers.push(id);
    });

  const moveTo = async (
    target: string,
    badge?: string | null,
    options?: { offsetX?: number; offsetY?: number; settleMs?: number },
  ) => {
    if (cancelled) return false;
    onLayoutTick();
    emitCursor({
      ...currentCursor,
      target,
      offsetX: options?.offsetX,
      offsetY: options?.offsetY,
      visible: true,
      pressing: false,
      badge: badge ?? null,
    });
    return wait(options?.settleMs ?? 660);
  };

  const click = async (onAction: () => void, afterMs = 380) => {
    if (cancelled) return false;
    clickSeq += 1;
    emitCursor({ ...currentCursor, pressing: true, clickSeq });
    if (!(await wait(140))) return false;
    onAction();
    onLayoutTick();
    emitCursor({ ...currentCursor, pressing: false });
    return wait(afterMs);
  };

  const cancel = () => {
    cancelled = true;
    timers.forEach(id => window.clearTimeout(id));
    timers.length = 0;
  };

  return { wait, moveTo, click, cancel, isCancelled: () => cancelled };
}

function demoProps(
  id: string,
  hoverTarget: string | null,
  pressTarget: string | null,
) {
  return {
    "data-demo-target": id,
    "data-demo-hover": hoverTarget === id ? "true" : undefined,
    "data-demo-press": pressTarget === id ? "true" : undefined,
  };
}

/* ==========================================================================
   SCENE 1: Top Command Bar Search, Ticker Selection & Favorites
   ========================================================================== */

export function SearchScene({
  mode,
  replaySeq,
  beatJump,
  onBeatChange,
  onCursorChange,
  onLayoutTick,
  onUserInteract,
}: SceneProps) {
  const [query, setQuery] = useState("");
  const [dropdownOpen, setDropdownOpen] = useState(true);
  const [selection, setSelection] = useState<string[]>([]);
  const [favorites, setFavorites] = useState<string[]>([]);
  const [flash, setFlash] = useState("Clique na busca ou use Ctrl K para encontrar ativos.");
  const [hoverTarget, setHoverTarget] = useState<string | null>(null);
  const [pressTarget, setPressTarget] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const forwardCursor = (next: CursorState) => {
    setHoverTarget(next.visible ? next.target : null);
    setPressTarget(next.visible && next.pressing ? next.target : null);
    onCursorChange(next);
  };

  useEffect(() => {
    if (!beatJump) return;
    if (beatJump.beat === 0) {
      setQuery("");
      setDropdownOpen(true);
      setSelection([]);
      setFavorites([]);
      setFlash("Busca aberta: digite um ticker ou empresa.");
    } else if (beatJump.beat === 1) {
      setQuery("PETR");
      setDropdownOpen(true);
      setSelection([]);
      setFavorites([]);
      setFlash("Resultados filtrados por \"PETR\".");
    } else if (beatJump.beat === 2) {
      setQuery("");
      setDropdownOpen(true);
      setSelection(["PETR4", "VALE3"]);
      setFavorites([]);
      setFlash("PETR4 e VALE3 adicionadas à seleção (2/20).");
    } else {
      setQuery("");
      setDropdownOpen(true);
      setSelection(["PETR4", "VALE3"]);
      setFavorites(["PETR4"]);
      setFlash("PETR4 marcada como favorita.");
    }
    onLayoutTick();
  }, [beatJump]);

  useEffect(() => {
    if (mode !== "demo") {
      setHoverTarget(null);
      setPressTarget(null);
      if (mode === "interactive") onCursorChange(idleCursor);
      return;
    }

    const runner = createRunner(forwardCursor, onLayoutTick);

    const runLoop = async () => {
      while (!runner.isCancelled()) {
        // Reset cleanly for each cycle
        onBeatChange(0);
        setQuery("");
        setDropdownOpen(false);
        setSelection([]);
        setFavorites([]);
        setFlash("A seleção no topo vale para todas as telas.");
        if (!(await runner.wait(320))) break;

        // Beat 0: Move to search input & click
        if (!(await runner.moveTo("search-input", "Ctrl K", { offsetX: 0.42, offsetY: 0.54, settleMs: 650 }))) break;
        if (!(await runner.click(() => {
          setDropdownOpen(true);
          setFlash("Digite um código como PETR4 ou nome da empresa.");
        }, 380))) break;

        // Beat 1: Type "PETR" character by character
        onBeatChange(1);
        if (!(await runner.moveTo("search-input", "Digitar \"PETR\"", { offsetX: 0.48, offsetY: 0.54, settleMs: 220 }))) break;
        for (const partial of ["P", "PE", "PET", "PETR"]) {
          if (!(await runner.wait(140))) break;
          setQuery(partial);
        }
        if (runner.isCancelled() || !(await runner.wait(360))) break;

        // Beat 2: Click PETR4, then VALE3
        onBeatChange(2);
        if (!(await runner.moveTo("option-PETR4", "Adicionar PETR4", { offsetX: 0.44, offsetY: 0.52, settleMs: 650 }))) break;
        if (!(await runner.click(() => {
          setSelection(["PETR4"]);
          setQuery("");
          setFlash("PETR4 adicionada à seleção (1/20).");
        }, 560))) break;

        if (!(await runner.moveTo("option-VALE3", "Adicionar VALE3", { offsetX: 0.44, offsetY: 0.52, settleMs: 620 }))) break;
        if (!(await runner.click(() => {
          setSelection(["PETR4", "VALE3"]);
          setFlash("VALE3 adicionada à seleção (2/20).");
        }, 560))) break;

        // Beat 3: Favorite PETR4 with the star button
        onBeatChange(3);
        if (!(await runner.moveTo("star-PETR4", "Favoritar ★", { offsetX: 0.5, offsetY: 0.52, settleMs: 620 }))) break;
        if (!(await runner.click(() => {
          setFavorites(["PETR4"]);
          setFlash("PETR4 salva nos favoritos.");
        }, 1600))) break;
      }
    };

    void runLoop();
    return () => runner.cancel();
  }, [mode, replaySeq]);

  const needle = query.trim().toLowerCase();
  const filtered = DEMO_STOCKS.filter(
    stock =>
      !needle ||
      stock.ticker.toLowerCase().includes(needle) ||
      stock.company.toLowerCase().includes(needle) ||
      stock.sector.toLowerCase().includes(needle),
  ).slice(0, 4);

  const toggleTicker = (ticker: string) => {
    onUserInteract();
    setSelection(current => {
      const exists = current.includes(ticker);
      const next = exists ? current.filter(item => item !== ticker) : [...current, ticker];
      setFlash(
        exists
          ? `${ticker} removida da seleção (${next.length}/20).`
          : `${ticker} adicionada à seleção (${next.length}/20).`,
      );
      return next;
    });
  };

  const toggleFav = (ticker: string) => {
    onUserInteract();
    setFavorites(current => {
      const exists = current.includes(ticker);
      setFlash(exists ? `${ticker} removida dos favoritos.` : `${ticker} salva nos favoritos.`);
      return exists ? current.filter(item => item !== ticker) : [...current, ticker];
    });
  };

  return (
    <div className="tut-mini-app">
      <div className="tut-mini-cmd">
        <span className="tut-mini-brand"><span>B3</span>LAB</span>
        <div className={`tut-mini-search${dropdownOpen ? " is-open" : ""}`} {...demoProps("search-input", hoverTarget, pressTarget)}>
          <span className="tut-mini-prompt" aria-hidden="true">&gt;</span>
          <input
            ref={inputRef}
            type="text"
            value={query}
            aria-label="Campo de busca demonstrativo"
            placeholder="Ticker, empresa ou setor"
            onFocus={() => setDropdownOpen(true)}
            onClick={() => {
              onUserInteract();
              setDropdownOpen(true);
            }}
            onChange={event => {
              onUserInteract();
              setQuery(event.target.value);
              setDropdownOpen(true);
            }}
          />
          <kbd>Ctrl K</kbd>
        </div>
        <div className="tut-mini-sel" aria-label="Ativos selecionados na demonstração">
          <span className="tut-mini-count">Seleção <b>{selection.length}/20</b></span>
          <div className="tut-mini-chips">
            {selection.length ? (
              selection.map((ticker, index) => (
                <span key={ticker} className="tut-chip">
                  <i style={{ background: demoColor(index) }} aria-hidden="true" />
                  {ticker}
                  <button
                    type="button"
                    aria-label={`Remover ${ticker}`}
                    onClick={() => toggleTicker(ticker)}
                  >
                    ×
                  </button>
                </span>
              ))
            ) : (
              <span className="tut-mini-empty">vazia</span>
            )}
          </div>
        </div>
      </div>

      <div className="tut-mini-body tut-search-body">
        {dropdownOpen ? (
          <ul className="tut-mini-dropdown" aria-label="Resultados da busca demonstrativa">
            {filtered.map(stock => {
              const selected = selection.includes(stock.ticker);
              const fav = favorites.includes(stock.ticker);
              return (
                <li
                  key={stock.ticker}
                  className={`tut-mini-option${selected ? " is-selected" : ""}`}
                  {...demoProps(`option-${stock.ticker}`, hoverTarget, pressTarget)}
                >
                  <button
                    type="button"
                    className="tut-option-main"
                    onClick={() => toggleTicker(stock.ticker)}
                  >
                    <span className="tut-option-mark" aria-hidden="true">{selected ? "✓" : "+"}</span>
                    <b>{stock.ticker}</b>
                    <span className="tut-option-co">{stock.company}<small>{stock.sector}</small></span>
                    <span className="tut-option-hint">{selected ? "remover" : "adicionar"}</span>
                  </button>
                  <button
                    type="button"
                    className={`tut-option-star${fav ? " is-on" : ""}`}
                    aria-pressed={fav}
                    aria-label={fav ? `Remover ${stock.ticker} dos favoritos` : `Favoritar ${stock.ticker}`}
                    {...demoProps(`star-${stock.ticker}`, hoverTarget, pressTarget)}
                    onClick={() => toggleFav(stock.ticker)}
                  >
                    {fav ? "★" : "☆"}
                  </button>
                </li>
              );
            })}
            {!filtered.length && (
              <li className="tut-mini-empty-row">Nenhum ativo encontrado para &ldquo;{query}&rdquo;.</li>
            )}
          </ul>
        ) : (
          <div className="tut-mini-placeholder">
            <p>Use a busca no topo ou pressione <kbd>Ctrl K</kbd> para montar sua seleção.</p>
          </div>
        )}
        <div className="tut-mini-flash" aria-live="polite">
          <span className="tut-dot" aria-hidden="true" />
          <span>{flash}</span>
        </div>
      </div>
    </div>
  );
}

/* ==========================================================================
   SCENE 2: Analyze & Compare Companies (Monitor + Tabs + Comparison Bars)
   ========================================================================== */

type AnalyzeTab = "visao" | "comparacao" | "alvo";
type CompareMetric = "roe" | "pl" | "dy";

const metricLabels: Record<CompareMetric, string> = {
  roe: "ROE",
  pl: "P/L",
  dy: "Dividend Yield",
};

export function AnalyzeScene({
  mode,
  replaySeq,
  beatJump,
  onBeatChange,
  onCursorChange,
  onLayoutTick,
  onUserInteract,
}: SceneProps) {
  const [analyzed, setAnalyzed] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [tab, setTab] = useState<AnalyzeTab>("visao");
  const [metric, setMetric] = useState<CompareMetric>("roe");
  const [hoverTarget, setHoverTarget] = useState<string | null>(null);
  const [pressTarget, setPressTarget] = useState<string | null>(null);

  const forwardCursor = (next: CursorState) => {
    setHoverTarget(next.visible ? next.target : null);
    setPressTarget(next.visible && next.pressing ? next.target : null);
    onCursorChange(next);
  };

  useEffect(() => {
    if (!beatJump) return;
    setScanning(false);
    if (beatJump.beat === 0) {
      setAnalyzed(false);
      setTab("visao");
    } else if (beatJump.beat === 1) {
      setAnalyzed(true);
      setTab("visao");
    } else if (beatJump.beat === 2) {
      setAnalyzed(true);
      setTab("comparacao");
      setMetric("dy");
    } else {
      setAnalyzed(true);
      setTab("alvo");
    }
    onLayoutTick();
  }, [beatJump]);

  useEffect(() => {
    if (mode !== "demo") {
      setHoverTarget(null);
      setPressTarget(null);
      if (mode === "interactive") {
        setScanning(false);
        setAnalyzed(true);
        onCursorChange(idleCursor);
      }
      return;
    }

    const runner = createRunner(forwardCursor, onLayoutTick);

    const runLoop = async () => {
      while (!runner.isCancelled()) {
        // Beat 0: Monitor ready, click Analisar 2
        onBeatChange(0);
        setAnalyzed(false);
        setScanning(false);
        setTab("visao");
        setMetric("roe");
        if (!(await runner.wait(360))) break;

        if (!(await runner.moveTo("btn-analyze", "Clique em Analisar", { offsetX: 0.52, offsetY: 0.54, settleMs: 650 }))) break;
        if (!(await runner.click(() => {
          setScanning(true);
        }, 420))) break;

        // Beat 1: Results loaded on Visão geral
        onBeatChange(1);
        setScanning(false);
        setAnalyzed(true);
        if (!(await runner.wait(900))) break;

        // Beat 2: Switch to Comparação tab and change indicator
        onBeatChange(2);
        if (!(await runner.moveTo("tab-comparacao", "Comparar ações", { offsetX: 0.5, offsetY: 0.54, settleMs: 650 }))) break;
        if (!(await runner.click(() => {
          setTab("comparacao");
        }, 560))) break;

        if (!(await runner.moveTo("metric-dy", "Trocar indicador", { offsetX: 0.5, offsetY: 0.54, settleMs: 560 }))) break;
        if (!(await runner.click(() => {
          setMetric("dy");
        }, 720))) break;

        // Beat 3: Switch to Preço-alvo tab
        onBeatChange(3);
        if (!(await runner.moveTo("tab-alvo", "Ver preços-alvo", { offsetX: 0.5, offsetY: 0.54, settleMs: 620 }))) break;
        if (!(await runner.click(() => {
          setTab("alvo");
        }, 1650))) break;
      }
    };

    void runLoop();
    return () => runner.cancel();
  }, [mode, replaySeq]);

  const activeStocks = DEMO_STOCKS.slice(0, 3);
  const sortedBars = [...activeStocks].sort((a, b) => b.bars[metric] - a.bars[metric]);
  const medianPos = sortedBars[1].bars[metric];

  const handleAnalyzeClick = () => {
    onUserInteract();
    setScanning(false);
    setAnalyzed(true);
    setTab("visao");
  };

  return (
    <div className="tut-mini-split">
      <div className="tut-mini-pane">
        <div className="tut-mini-bar">
          <span>Monitor</span>
          <button
            type="button"
            className="tut-mini-btn primary"
            {...demoProps("btn-analyze", hoverTarget, pressTarget)}
            onClick={handleAnalyzeClick}
          >
            {scanning ? "Analisando…" : "Analisar 3"}
          </button>
        </div>
        <ul className="tut-monitor-list">
          {activeStocks.map((stock, index) => (
            <li key={stock.ticker}>
              <i style={{ background: demoColor(index) }} aria-hidden="true" />
              <b>{stock.ticker}</b>
              <span>{stock.company}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="tut-mini-pane">
        <div className="tut-mini-bar">
          <span>Análise</span>
          <span className="tut-mini-meta">{analyzed ? "3 ativos · ilustrativo" : "aguardando análise"}</span>
        </div>

        {scanning ? (
          <div className="tut-mini-loading">
            <span className="load-bar" aria-hidden="true" />
            <span>Buscando indicadores e consensos…</span>
          </div>
        ) : !analyzed ? (
          <div className="tut-mini-idle">
            <p>Seleção pronta: <strong>PETR4, VALE3, BBAS3</strong>.</p>
            <p className="muted">Clique em <strong>Analisar 3</strong> para preencher indicadores, comparação e preços-alvo.</p>
          </div>
        ) : (
          <div className="tut-analysis-content">
            <div className="tut-mini-tabs" role="tablist" aria-label="Abas de análise demonstrativa">
              <button
                type="button"
                role="tab"
                aria-selected={tab === "visao"}
                {...demoProps("tab-visao", hoverTarget, pressTarget)}
                onClick={() => { onUserInteract(); setTab("visao"); }}
              >
                Visão geral
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "comparacao"}
                {...demoProps("tab-comparacao", hoverTarget, pressTarget)}
                onClick={() => { onUserInteract(); setTab("comparacao"); }}
              >
                Comparação
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "alvo"}
                {...demoProps("tab-alvo", hoverTarget, pressTarget)}
                onClick={() => { onUserInteract(); setTab("alvo"); }}
              >
                Preço-alvo
              </button>
            </div>

            {tab === "visao" && (
              <div className="tut-visao-grid">
                {activeStocks.map((stock, index) => (
                  <div key={stock.ticker} className="tut-visao-row">
                    <span className="tut-tk"><i style={{ background: demoColor(index) }} aria-hidden="true" />{stock.ticker}</span>
                    <span className="tut-pill">ROE {stock.roeBand}</span>
                    <span className="tut-pill">DY {stock.dyBand}</span>
                    <span className="tut-pill">P/L {stock.plBand}</span>
                  </div>
                ))}
                <p className="tut-mini-note">Na análise real, a tabela exibe múltiplos, margens, dívida e síntese setorial.</p>
              </div>
            )}

            {tab === "comparacao" && (
              <div className="tut-compare">
                <div className="tut-metric-pills" role="group" aria-label="Indicador de comparação">
                  {(["roe", "pl", "dy"] as CompareMetric[]).map(key => (
                    <button
                      key={key}
                      type="button"
                      aria-pressed={metric === key}
                      {...demoProps(`metric-${key}`, hoverTarget, pressTarget)}
                      onClick={() => { onUserInteract(); setMetric(key); }}
                    >
                      {metricLabels[key]}
                    </button>
                  ))}
                </div>
                <ul className="tut-bars" aria-label={`Comparação ilustrativa de ${metricLabels[metric]}`}>
                  {sortedBars.map(stock => {
                    const colorIdx = activeStocks.findIndex(item => item.ticker === stock.ticker);
                    return (
                      <li key={stock.ticker}>
                        <span className="tut-bar-tk">
                          <i style={{ background: demoColor(colorIdx) }} aria-hidden="true" />
                          {stock.ticker}
                        </span>
                        <span className="tut-bar-track">
                          <span
                            className="tut-bar-fill"
                            style={{ width: `${stock.bars[metric]}%`, background: demoColor(colorIdx) }}
                          />
                          <span className="tut-bar-median" style={{ left: `${medianPos}%` }} />
                        </span>
                      </li>
                    );
                  })}
                </ul>
                <p className="tut-mini-note">Barras ordenadas do maior para o menor · tracejado âmbar indica a mediana.</p>
              </div>
            )}

            {tab === "alvo" && (
              <div className="tut-ladder">
                {activeStocks.map((stock, index) => (
                  <div key={stock.ticker} className="tut-ladder-row">
                    <span className="tut-tk"><i style={{ background: demoColor(index) }} aria-hidden="true" />{stock.ticker}</span>
                    <span className="tut-range-track" aria-hidden="true">
                      <span
                        className="tut-range-band"
                        style={{ left: `${stock.range.low}%`, width: `${stock.range.high - stock.range.low}%` }}
                      />
                      <span className="tut-range-mean" style={{ left: `${stock.range.mean}%` }} />
                      <span className="tut-range-price" style={{ left: `${stock.range.price}%` }} />
                    </span>
                    <span className={`tut-ladder-tag${stock.range.positive ? " is-pos" : " is-neg"}`}>
                      {stock.range.positive ? "▲ potencial" : "▼ abaixo"}
                    </span>
                  </div>
                ))}
                <p className="tut-mini-note">Faixa mínima a máxima, alvo médio em âmbar e cotação de referência em branco.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ==========================================================================
   SCENE 3: Screener Filters, Presets & Handoff to Selection
   ========================================================================== */

type ScreenerPreset = "todos" | "dividendos" | "qualidade";

export function ScreenerScene({
  mode,
  replaySeq,
  beatJump,
  onBeatChange,
  onCursorChange,
  onLayoutTick,
  onUserInteract,
}: SceneProps) {
  const [preset, setPreset] = useState<ScreenerPreset>("todos");
  const [filterDy, setFilterDy] = useState(false);
  const [filterRoe, setFilterRoe] = useState(false);
  const [filterPl, setFilterPl] = useState(false);
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [added, setAdded] = useState<string[]>(["PETR4", "VALE3"]);
  const [hoverTarget, setHoverTarget] = useState<string | null>(null);
  const [pressTarget, setPressTarget] = useState<string | null>(null);

  const forwardCursor = (next: CursorState) => {
    setHoverTarget(next.visible ? next.target : null);
    setPressTarget(next.visible && next.pressing ? next.target : null);
    onCursorChange(next);
  };

  useEffect(() => {
    if (!beatJump) return;
    if (beatJump.beat === 0) {
      setPreset("dividendos");
      setFilterDy(true);
      setFilterRoe(false);
      setFilterPl(false);
      setSelectedRow(null);
      setAdded(["PETR4", "VALE3"]);
    } else if (beatJump.beat === 1) {
      setPreset("dividendos");
      setFilterDy(true);
      setFilterRoe(true);
      setFilterPl(false);
      setSelectedRow(null);
      setAdded(["PETR4", "VALE3"]);
    } else if (beatJump.beat === 2) {
      setPreset("dividendos");
      setFilterDy(true);
      setFilterRoe(true);
      setFilterPl(false);
      setSelectedRow("BBAS3");
      setAdded(["PETR4", "VALE3"]);
    } else {
      setPreset("dividendos");
      setFilterDy(true);
      setFilterRoe(true);
      setFilterPl(false);
      setSelectedRow("BBAS3");
      setAdded(["PETR4", "VALE3", "BBAS3"]);
    }
    onLayoutTick();
  }, [beatJump]);

  useEffect(() => {
    if (mode !== "demo") {
      setHoverTarget(null);
      setPressTarget(null);
      if (mode === "interactive") onCursorChange(idleCursor);
      return;
    }

    const runner = createRunner(forwardCursor, onLayoutTick);

    const runLoop = async () => {
      while (!runner.isCancelled()) {
        // Beat 0: Apply Dividendos preset
        onBeatChange(0);
        setPreset("todos");
        setFilterDy(false);
        setFilterRoe(false);
        setFilterPl(false);
        setSelectedRow(null);
        setAdded(["PETR4", "VALE3"]);
        if (!(await runner.wait(360))) break;

        if (!(await runner.moveTo("preset-dividendos", "Aplicar preset", { offsetX: 0.52, offsetY: 0.54, settleMs: 640 }))) break;
        if (!(await runner.click(() => {
          setPreset("dividendos");
          setFilterDy(true);
        }, 560))) break;

        // Beat 1: Toggle ROE criterion
        onBeatChange(1);
        if (!(await runner.moveTo("crit-roe", "Refinar por ROE", { offsetX: 0.5, offsetY: 0.54, settleMs: 620 }))) break;
        if (!(await runner.click(() => {
          setFilterRoe(true);
        }, 560))) break;

        // Beat 2: Click BBAS3 row
        onBeatChange(2);
        if (!(await runner.moveTo("scr-row-BBAS3", "Escolher ativo", { offsetX: 0.42, offsetY: 0.52, settleMs: 640 }))) break;
        if (!(await runner.click(() => {
          setSelectedRow("BBAS3");
        }, 520))) break;

        // Beat 3: Send to shared selection
        onBeatChange(3);
        if (!(await runner.moveTo("btn-scr-add", "Enviar para seleção", { offsetX: 0.5, offsetY: 0.54, settleMs: 620 }))) break;
        if (!(await runner.click(() => {
          setAdded(["PETR4", "VALE3", "BBAS3"]);
        }, 1650))) break;
      }
    };

    void runLoop();
    return () => runner.cancel();
  }, [mode, replaySeq]);

  const rows = DEMO_STOCKS.filter(stock => {
    if (filterDy && stock.dyBand !== "alto") return false;
    if (filterRoe && stock.roeBand !== "alto") return false;
    if (filterPl && stock.plBand !== "baixo") return false;
    return true;
  });

  const applyDemoPreset = (next: ScreenerPreset) => {
    onUserInteract();
    setPreset(next);
    if (next === "todos") {
      setFilterDy(false);
      setFilterRoe(false);
      setFilterPl(false);
    } else if (next === "dividendos") {
      setFilterDy(true);
      setFilterRoe(false);
      setFilterPl(false);
    } else {
      setFilterDy(false);
      setFilterRoe(true);
      setFilterPl(true);
    }
    setSelectedRow(null);
  };

  const addSelectedToSelection = () => {
    if (!selectedRow) return;
    onUserInteract();
    setAdded(current => (current.includes(selectedRow) ? current : [...current, selectedRow]));
  };

  return (
    <div className="tut-mini-split">
      <div className="tut-mini-pane">
        <div className="tut-mini-bar">
          <span>Filtros · F5</span>
          <span className="tut-mini-meta">{rows.length} de {DEMO_STOCKS.length}</span>
        </div>
        <div className="tut-scr-controls">
          <div className="tut-scr-presets" role="group" aria-label="Presets demonstrativos">
            {([
              ["todos", "Todos"],
              ["dividendos", "Dividendos"],
              ["qualidade", "Qualidade"],
            ] as const).map(([key, label]) => (
              <button
                key={key}
                type="button"
                aria-pressed={preset === key}
                {...demoProps(`preset-${key}`, hoverTarget, pressTarget)}
                onClick={() => applyDemoPreset(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="tut-scr-checks">
            <button
              type="button"
              className="tut-check-btn"
              aria-pressed={filterDy}
              {...demoProps("crit-dy", hoverTarget, pressTarget)}
              onClick={() => { onUserInteract(); setFilterDy(value => !value); }}
            >
              <span aria-hidden="true">{filterDy ? "■" : "□"}</span> Dividend Yield alto
            </button>
            <button
              type="button"
              className="tut-check-btn"
              aria-pressed={filterRoe}
              {...demoProps("crit-roe", hoverTarget, pressTarget)}
              onClick={() => { onUserInteract(); setFilterRoe(value => !value); }}
            >
              <span aria-hidden="true">{filterRoe ? "■" : "□"}</span> ROE alto
            </button>
            <button
              type="button"
              className="tut-check-btn"
              aria-pressed={filterPl}
              {...demoProps("crit-pl", hoverTarget, pressTarget)}
              onClick={() => { onUserInteract(); setFilterPl(value => !value); }}
            >
              <span aria-hidden="true">{filterPl ? "■" : "□"}</span> P/L descontado
            </button>
          </div>
        </div>
      </div>

      <div className="tut-mini-pane">
        <div className="tut-mini-bar">
          <span>Resultados</span>
          <span className="tut-mini-meta">Seleção: {added.length}/20</span>
          <button
            type="button"
            className="tut-mini-btn primary"
            disabled={!selectedRow || added.includes(selectedRow)}
            {...demoProps("btn-scr-add", hoverTarget, pressTarget)}
            onClick={addSelectedToSelection}
          >
            {selectedRow && added.includes(selectedRow) ? "✓ Na seleção" : "+ Seleção"}
          </button>
        </div>
        <ul className="tut-scr-table" aria-label="Ações filtradas demonstrativas">
          {rows.map(stock => {
            const isSelected = selectedRow === stock.ticker;
            const inSelection = added.includes(stock.ticker);
            return (
              <li key={stock.ticker} className={isSelected ? "is-active" : undefined}>
                <button
                  type="button"
                  aria-pressed={isSelected}
                  {...demoProps(`scr-row-${stock.ticker}`, hoverTarget, pressTarget)}
                  onClick={() => { onUserInteract(); setSelectedRow(stock.ticker); }}
                >
                  <b>{stock.ticker}</b>
                  <span className="tut-scr-co">{stock.company}</span>
                  <span className="tut-scr-badge">{inSelection ? "na seleção" : stock.sector}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

/* ==========================================================================
   SCENE 4: Portfolio Strategy, Unlocking Simulation & News
   ========================================================================== */

type PortfolioStrategy = "min_vol" | "markowitz" | "hrp";
type PortfolioView = "carteira" | "simulacao" | "noticias";
type Horizon = "1a" | "2a" | "5a";

const STRATEGY_WEIGHTS: Record<PortfolioStrategy, number[]> = {
  min_vol: [28, 30, 42],
  markowitz: [44, 24, 32],
  hrp: [34, 33, 33],
};

const HORIZON_FAN: Record<Horizon, { outer: string; inner: string; median: string }> = {
  "1a": {
    outer: "M0 72 C90 66 180 56 300 42 L300 104 C180 94 90 82 0 72 Z",
    inner: "M0 72 C90 69 180 63 300 55 L300 88 C180 83 90 77 0 72 Z",
    median: "M0 72 C90 70 180 67 300 64",
  },
  "2a": {
    outer: "M0 72 C90 62 180 46 300 24 L300 112 C180 98 90 84 0 72 Z",
    inner: "M0 72 C90 66 180 56 300 42 L300 92 C180 84 90 78 0 72 Z",
    median: "M0 72 C90 68 180 62 300 54",
  },
  "5a": {
    outer: "M0 72 C90 56 180 34 300 10 L300 118 C180 102 90 86 0 72 Z",
    inner: "M0 72 C90 62 180 48 300 30 L300 94 C180 84 90 78 0 72 Z",
    median: "M0 72 C90 65 180 54 300 40",
  },
};

export function PortfolioScene({
  mode,
  replaySeq,
  beatJump,
  onBeatChange,
  onCursorChange,
  onLayoutTick,
  onUserInteract,
}: SceneProps) {
  const [view, setView] = useState<PortfolioView>("carteira");
  const [strategy, setStrategy] = useState<PortfolioStrategy>("min_vol");
  const [ready, setReady] = useState(false);
  const [horizon, setHorizon] = useState<Horizon>("2a");
  const [hoverTarget, setHoverTarget] = useState<string | null>(null);
  const [pressTarget, setPressTarget] = useState<string | null>(null);

  const forwardCursor = (next: CursorState) => {
    setHoverTarget(next.visible ? next.target : null);
    setPressTarget(next.visible && next.pressing ? next.target : null);
    onCursorChange(next);
  };

  useEffect(() => {
    if (!beatJump) return;
    if (beatJump.beat === 0) {
      setView("carteira");
      setStrategy("markowitz");
      setReady(false);
    } else if (beatJump.beat === 1) {
      setView("carteira");
      setStrategy("markowitz");
      setReady(true);
    } else if (beatJump.beat === 2) {
      setView("simulacao");
      setStrategy("markowitz");
      setReady(true);
      setHorizon("2a");
    } else {
      setView("simulacao");
      setStrategy("markowitz");
      setReady(true);
      setHorizon("5a");
    }
    onLayoutTick();
  }, [beatJump]);

  useEffect(() => {
    if (mode !== "demo") {
      setHoverTarget(null);
      setPressTarget(null);
      if (mode === "interactive") {
        setReady(true);
        onCursorChange(idleCursor);
      }
      return;
    }

    const runner = createRunner(forwardCursor, onLayoutTick);

    const runLoop = async () => {
      while (!runner.isCancelled()) {
        // Beat 0: Choose strategy Markowitz
        onBeatChange(0);
        setView("carteira");
        setStrategy("min_vol");
        setReady(false);
        setHorizon("2a");
        if (!(await runner.wait(360))) break;

        if (!(await runner.moveTo("strat-markowitz", "Escolher estratégia", { offsetX: 0.52, offsetY: 0.54, settleMs: 640 }))) break;
        if (!(await runner.click(() => {
          setStrategy("markowitz");
        }, 560))) break;

        // Beat 1: Click Calcular carteira -> unlocks F3 & F4
        onBeatChange(1);
        if (!(await runner.moveTo("btn-load-pf", "Calcular carteira", { offsetX: 0.52, offsetY: 0.54, settleMs: 640 }))) break;
        if (!(await runner.click(() => {
          setReady(true);
        }, 760))) break;

        // Beat 2: Click F3 Simulação in nav bar
        onBeatChange(2);
        if (!(await runner.moveTo("nav-simulacao", "Abrir Simulação (F3)", { offsetX: 0.5, offsetY: 0.54, settleMs: 640 }))) break;
        if (!(await runner.click(() => {
          setView("simulacao");
        }, 620))) break;

        // Beat 3: Change simulation horizon to 5a
        onBeatChange(3);
        if (!(await runner.moveTo("horizon-5a", "Ajustar horizonte", { offsetX: 0.5, offsetY: 0.54, settleMs: 600 }))) break;
        if (!(await runner.click(() => {
          setHorizon("5a");
        }, 1650))) break;
      }
    };

    void runLoop();
    return () => runner.cancel();
  }, [mode, replaySeq]);

  const tickers = ["PETR4", "VALE3", "BBAS3"];
  const weights = STRATEGY_WEIGHTS[strategy];
  const fan = HORIZON_FAN[horizon];

  return (
    <div className="tut-mini-app">
      <div className="tut-mini-nav" role="tablist" aria-label="Telas de carteira demonstrativas">
        <button
          type="button"
          role="tab"
          aria-selected={view === "carteira"}
          {...demoProps("nav-carteira", hoverTarget, pressTarget)}
          onClick={() => { onUserInteract(); setView("carteira"); }}
        >
          <kbd>F2</kbd> Carteira
          <span className={`cmd-ready${ready ? " is-ready" : ""}`} aria-hidden="true" />
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={view === "simulacao"}
          disabled={!ready}
          {...demoProps("nav-simulacao", hoverTarget, pressTarget)}
          onClick={() => { onUserInteract(); if (ready) setView("simulacao"); }}
        >
          <kbd>F3</kbd> Simulação
          <span className={`cmd-ready${ready ? " is-ready" : ""}`} aria-hidden="true" />
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={view === "noticias"}
          disabled={!ready}
          {...demoProps("nav-noticias", hoverTarget, pressTarget)}
          onClick={() => { onUserInteract(); if (ready) setView("noticias"); }}
        >
          <kbd>F4</kbd> Notícias
          <span className={`cmd-ready${ready ? " is-ready" : ""}`} aria-hidden="true" />
        </button>
        <span className="tut-mini-nav-hint">
          {ready ? "■ liberadas" : "□ carregue a carteira"}
        </span>
      </div>

      {view === "carteira" && (
        <div className="tut-mini-split">
          <div className="tut-mini-pane">
            <div className="tut-mini-bar">
              <span>Parâmetros</span>
            </div>
            <div className="tut-pf-params">
              <div className="tut-scr-presets" role="group" aria-label="Estratégia de alocação">
                {([
                  ["min_vol", "Mín. vol."],
                  ["markowitz", "Markowitz"],
                  ["hrp", "HRP"],
                ] as const).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    aria-pressed={strategy === key}
                    {...demoProps(`strat-${key}`, hoverTarget, pressTarget)}
                    onClick={() => { onUserInteract(); setStrategy(key); }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <button
                type="button"
                className="tut-mini-btn primary tut-pf-run"
                {...demoProps("btn-load-pf", hoverTarget, pressTarget)}
                onClick={() => { onUserInteract(); setReady(true); }}
              >
                {ready ? "✓ Carteira calculada" : "Calcular carteira"}
              </button>
            </div>
          </div>

          <div className="tut-mini-pane">
            <div className="tut-mini-bar">
              <span>Pesos do portfólio</span>
              <span className="tut-mini-meta">{ready ? "pronto" : "prévia ilustrativa"}</span>
            </div>
            <ul className="tut-weights">
              {tickers.map((ticker, index) => (
                <li key={ticker}>
                  <span className="tut-tk"><i style={{ background: demoColor(index) }} aria-hidden="true" />{ticker}</span>
                  <span className="tut-weight-track">
                    <span style={{ width: `${weights[index]}%`, background: demoColor(index) }} />
                  </span>
                  <span className="tut-weight-pct">{weights[index]}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {view === "simulacao" && (
        <div className="tut-sim-pane">
          <div className="tut-mini-bar">
            <span>Faixas de percentis · Monte Carlo</span>
            <div className="tut-metric-pills" role="group" aria-label="Horizonte demonstrativo">
              {(["1a", "2a", "5a"] as Horizon[]).map(key => (
                <button
                  key={key}
                  type="button"
                  aria-pressed={horizon === key}
                  {...demoProps(`horizon-${key}`, hoverTarget, pressTarget)}
                  onClick={() => { onUserInteract(); setHorizon(key); }}
                >
                  {key === "1a" ? "1 ano" : key === "2a" ? "2 anos" : "5 anos"}
                </button>
              ))}
            </div>
          </div>
          <div className="tut-fan-wrap">
            <svg viewBox="0 0 300 126" preserveAspectRatio="none" role="img" aria-label="Ilustração das faixas de percentis da simulação">
              <line x1="0" x2="300" y1="36" y2="36" stroke="#1d2229" />
              <line x1="0" x2="300" y1="72" y2="72" stroke="#8a929e" strokeDasharray="4 4" />
              <line x1="0" x2="300" y1="104" y2="104" stroke="#1d2229" />
              <path d={fan.outer} fill="rgb(255 178 36 / .1)" />
              <path d={fan.inner} fill="rgb(255 178 36 / .22)" />
              <path d={fan.median} stroke="#ffb224" strokeWidth="1.75" fill="none" />
            </svg>
            <p className="tut-mini-note">Faixa clara P5–P95 · escura P25–P75 · linha âmbar mediana · tracejado capital inicial.</p>
          </div>
        </div>
      )}

      {view === "noticias" && (
        <div className="tut-news-pane">
          <div className="tut-mini-bar">
            <span>Notícias da carteira · F4</span>
            <span className="tut-mini-meta">filtradas pelos 3 ativos</span>
          </div>
          <ul className="tut-news-list">
            <li>
              <span className="tut-tk"><i style={{ background: demoColor(0) }} aria-hidden="true" />PETR4</span>
              <span>Cobertura recente agrupada por ativo da carteira</span>
              <span className="tut-pill is-pos">+ positiva</span>
            </li>
            <li>
              <span className="tut-tk"><i style={{ background: demoColor(1) }} aria-hidden="true" />VALE3</span>
              <span>Sentimento classificado com fontes identificadas</span>
              <span className="tut-pill">■ neutra</span>
            </li>
            <li>
              <span className="tut-tk"><i style={{ background: demoColor(2) }} aria-hidden="true" />BBAS3</span>
              <span>Link direto para a matéria original e filtro por ticker</span>
              <span className="tut-pill is-pos">+ positiva</span>
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
