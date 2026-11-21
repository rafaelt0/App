"use client";

import { useCallback, useEffect, useState } from "react";
import "./intro.css";

const tips = [
  {
    key: "F1–F5",
    title: "Cinco telas, um atalho cada",
    description: "F1 a F5 abre Início, Carteira, Simulação, Notícias e Screener.",
    loading: "Carregando o universo de ações…",
  },
  {
    key: "Ctrl K",
    title: "Busca de qualquer tela",
    description: "Ctrl K busca qualquer ativo; Enter adiciona à seleção.",
    loading: "Carregando cotações com atraso…",
  },
  {
    key: "+",
    title: "Monte a sua seleção",
    description: "Cartões, mapa setorial ou busca: a seleção segue para a Carteira.",
    loading: "Lendo indicadores e consenso…",
  },
  {
    key: "F2",
    title: "Da seleção à carteira",
    description: "Pesos, retorno, drawdown e sugestão de compra de cotas.",
    loading: "Restaurando carteira e favoritos…",
  },
];

const tipMs = 1250;
const fadeMs = 120;
const exitMs = 340;
const minMs = 600;
const maxMs = 2400;
const reducedMotion = "(prefers-reduced-motion: reduce)";

export default function Intro({ loading, onFinish }: { loading: boolean; onFinish: () => void }) {
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [index, setIndex] = useState(0);
  const [fading, setFading] = useState(false);
  const [leaving, setLeaving] = useState(false);

  const dismiss = useCallback(() => {
    if (window.matchMedia(reducedMotion).matches) onFinish();
    else setLeaving(true);
  }, [onFinish]);

  useEffect(() => {
    function start(event?: Event) {
      if (document.hidden) return;
      // The server-rendered intro can already be visible before hydration.
      const firstPaint = event ? undefined : performance.getEntriesByName("first-contentful-paint", "paint")[0];
      setStartedAt(firstPaint?.startTime ?? performance.now());
      document.removeEventListener("visibilitychange", start);
    }
    document.addEventListener("visibilitychange", start);
    start();
    return () => document.removeEventListener("visibilitychange", start);
  }, []);

  useEffect(() => {
    if (startedAt === null || leaving) return;
    const elapsed = performance.now() - startedAt;
    const timers = tips.slice(1).flatMap((_, offset) => {
      const next = offset + 1;
      return [
        window.setTimeout(() => setFading(true), next * tipMs - fadeMs - elapsed),
        window.setTimeout(() => { setIndex(next); setFading(false); }, next * tipMs - elapsed),
      ];
    });
    return () => timers.forEach(timer => window.clearTimeout(timer));
  }, [startedAt, leaving]);

  useEffect(() => {
    if (startedAt === null || leaving) return;
    const start = startedAt;
    const motion = window.matchMedia(reducedMotion);
    let timer: number;
    function schedule() {
      window.clearTimeout(timer);
      // Reserve the slide time so a slow bootstrap still clears the viewport by maxMs.
      const deadline = loading ? maxMs - (motion.matches ? 0 : exitMs) : minMs;
      timer = window.setTimeout(dismiss, Math.max(0, start + deadline - performance.now()));
    }
    schedule();
    motion.addEventListener("change", schedule);
    return () => {
      window.clearTimeout(timer);
      motion.removeEventListener("change", schedule);
    };
  }, [startedAt, loading, leaving, dismiss]);

  useEffect(() => {
    if (!leaving) return;
    const motion = window.matchMedia(reducedMotion);
    const finishIfReduced = () => { if (motion.matches) onFinish(); };
    motion.addEventListener("change", finishIfReduced);
    finishIfReduced();
    const timer = window.setTimeout(onFinish, exitMs);
    return () => {
      window.clearTimeout(timer);
      motion.removeEventListener("change", finishIfReduced);
    };
  }, [leaving, onFinish]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key !== "Escape" && event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      dismiss();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [dismiss]);

  const tip = tips[index];
  const step = index + 1;
  return (
    <div className="intro" data-leaving={leaving} role="status" aria-live="polite" aria-label="Abertura do B3Lab" onClick={dismiss}>
      <div className="intro-inner">
        <div className="intro-mark">
          <svg viewBox="6 0 50 54" width="42" height="45" aria-hidden="true">
            <rect x="22" y="1" width="18" height="5" rx="2.5" fill="var(--amber)" />
            <path d="M22 6 L22 25 L10 47 Q7 53 14 53 L50 53 Q57 53 54 47 L42 25 L42 6 Z" fill="var(--raise)" stroke="var(--amber)" strokeWidth="2" strokeLinejoin="round" />
            <path d="M16 51 Q14 53 14 53 L50 53 Q57 53 54 47 L42 35 L22 35 Z" fill="var(--line)" />
            <g fill="var(--pos)">
              <path d="M24 31v20 M40 33v18" stroke="var(--pos)" opacity=".8" />
              <rect x="21" y="37" width="6" height="10" rx="1" />
              <rect x="37" y="39" width="6" height="8" rx="1" />
            </g>
            <path d="M32 29v22" stroke="var(--neg)" opacity=".8" />
            <rect x="29" y="31" width="6" height="14" rx="1" fill="var(--neg)" />
          </svg>
          <div className="intro-brand">
            <span className="intro-wordmark"><b>B3</b>LAB<i aria-hidden="true" /></span>
            <span>Análise quantitativa · B3</span>
          </div>
        </div>
        <div className="intro-tour" data-fading={fading}>
          <div className="intro-counter"><span>Dica</span><span>0{step} / 0{tips.length}</span></div>
          <kbd className="intro-key" data-glyph={tip.key.length <= 2}>{tip.key}</kbd>
          <div className="intro-title">{tip.title}</div>
          <p>{tip.description}</p>
        </div>
        <div className="intro-progress" role="progressbar" aria-label="Progresso das dicas" aria-valuemin={0} aria-valuemax={tips.length} aria-valuenow={step}>
          <span style={{ width: `${step / tips.length * 100}%` }} />
          <i /><i /><i />
        </div>
        <div className="intro-meta">
          <div className="intro-loading"><i aria-hidden="true" /><span>{tip.loading}</span></div>
          <button className="intro-skip" type="button"><kbd>esc</kbd><span>ou clique para pular</span></button>
        </div>
      </div>
    </div>
  );
}
