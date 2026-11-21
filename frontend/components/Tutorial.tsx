"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ComponentType, type ReactNode } from "react";
import TutorialCursor, { idleCursor, type CursorState } from "./TutorialCursor";
import {
  AnalyzeScene,
  PortfolioScene,
  ScreenerScene,
  SearchScene,
  type DemoMode,
  type SceneProps,
} from "./TutorialScenes";
import "./tutorial.css";

type StepDef = {
  id: string;
  shortLabel: string;
  title: string;
  body: ReactNode;
  hint: string;
  beats: [string, string, string, string];
  tryAction: { label: string; href: string; focusSearch?: boolean };
  Scene: ComponentType<SceneProps>;
};

const steps: StepDef[] = [
  {
    id: "search",
    shortLabel: "Buscar",
    title: "Busque e monte sua seleção",
    body: <>Na barra do topo, digite o nome de uma empresa ou um código como <strong>PETR4</strong> e clique para adicionar. A seleção fica sincronizada entre todas as telas — clique na prévia abaixo para testar.</>,
    hint: "Atalho: Ctrl K ou /",
    beats: ["Abrir busca", "Digitar código", "Selecionar ativos", "Favoritar ★"],
    tryAction: { label: "Abrir busca agora", href: "/", focusSearch: true },
    Scene: SearchScene,
  },
  {
    id: "analyze",
    shortLabel: "Analisar",
    title: "Analise e compare empresas",
    body: <>No <strong>Início</strong>, clique em <strong>Analisar</strong> para carregar indicadores, saúde financeira e preços-alvo. Com mais de uma ação, abra a aba <strong>Comparação</strong> para ver o ranking contra a mediana.</>,
    hint: "Tela: F1 Início",
    beats: ["Clicar em Analisar", "Ver indicadores", "Comparar ações", "Consultar preço-alvo"],
    tryAction: { label: "Ir para o Início", href: "/" },
    Scene: AnalyzeScene,
  },
  {
    id: "screener",
    shortLabel: "Screener",
    title: "Filtre o mercado no Screener",
    body: <>Use presets prontos ou combine filtros de <strong>Dividend Yield</strong>, <strong>ROE</strong>, <strong>P/L</strong> e liquidez. Clique em uma linha da tabela e envie o ativo direto para a sua seleção.</>,
    hint: "Tela: F5 Screener",
    beats: ["Aplicar preset", "Refinar filtros", "Selecionar linha", "Enviar para seleção"],
    tryAction: { label: "Abrir Screener", href: "/screener" },
    Scene: ScreenerScene,
  },
  {
    id: "portfolio",
    shortLabel: "Carteira",
    title: "Monte a carteira e simule cenários",
    body: <>Com dois ou mais ativos, escolha a estratégia na <strong>Carteira</strong> e clique em <strong>Carregar portfólio</strong>. Isso libera <strong>Simulação</strong> (Monte Carlo) e <strong>Notícias</strong> para acompanhar os papéis.</>,
    hint: "Telas: F2 · F3 · F4",
    beats: ["Escolher estratégia", "Carregar portfólio", "Abrir Simulação", "Ajustar horizonte"],
    tryAction: { label: "Abrir Carteira", href: "/portfolio" },
    Scene: PortfolioScene,
  },
];

export default function Tutorial({ onDismiss }: { onDismiss: () => void }) {
  const router = useRouter();
  const dialog = useRef<HTMLDialogElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [index, setIndex] = useState(0);
  const [mode, setMode] = useState<DemoMode>(() => {
    if (typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return "interactive";
    }
    return "demo";
  });
  const [replaySeq, setReplaySeq] = useState(0);
  const [activeBeat, setActiveBeat] = useState(0);
  const [beatJump, setBeatJump] = useState<{ beat: number; seq: number } | null>(null);
  const [cursor, setCursor] = useState<CursorState>(idleCursor);
  const [layoutTick, setLayoutTick] = useState(0);

  const step = steps[index];
  const last = index === steps.length - 1;
  const Scene = step.Scene;

  useLayoutEffect(() => {
    const element = dialog.current!;
    element.showModal();
    return () => element.close();
  }, []);

  useLayoutEffect(() => {
    heading.current?.focus();
  }, [index]);

  const dismiss = useCallback(() => {
    dialog.current?.close();
    if (document.activeElement === document.body) {
      document.getElementById("conteudo")?.focus({ preventScroll: true });
    }
    onDismiss();
  }, [onDismiss]);

  const goToStep = useCallback((nextIndex: number) => {
    const clamped = Math.max(0, Math.min(steps.length - 1, nextIndex));
    setIndex(clamped);
    setActiveBeat(0);
    setBeatJump(null);
    setCursor(idleCursor);
    const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setMode(reduced ? "interactive" : "demo");
    setReplaySeq(value => value + 1);
  }, []);

  const handleUserInteract = useCallback(() => {
    setMode(current => (current === "interactive" ? current : "interactive"));
    setCursor(idleCursor);
  }, []);

  const handleLayoutTick = useCallback(() => {
    setLayoutTick(value => value + 1);
  }, []);

  const handleReplay = useCallback(() => {
    setBeatJump(null);
    setActiveBeat(0);
    setCursor(idleCursor);
    setMode("demo");
    setReplaySeq(value => value + 1);
  }, []);

  const handleTogglePlay = useCallback(() => {
    if (mode === "demo") {
      setMode("paused");
    } else {
      setBeatJump(null);
      setMode("demo");
      setReplaySeq(value => value + 1);
    }
  }, [mode]);

  const handleBeatClick = useCallback((beat: number) => {
    setMode("interactive");
    setCursor(idleCursor);
    setActiveBeat(beat);
    setBeatJump(current => ({ beat, seq: (current?.seq ?? 0) + 1 }));
  }, []);

  const handleTryInApp = useCallback(() => {
    const { href, focusSearch } = step.tryAction;
    dismiss();
    router.push(href);
    if (focusSearch) {
      window.setTimeout(() => {
        window.dispatchEvent(new Event("b3lab:focus-search"));
      }, 120);
    }
  }, [dismiss, router, step.tryAction]);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && target.closest("input, textarea, select")) return;
      if (event.key === "ArrowRight" && index < steps.length - 1) {
        event.preventDefault();
        goToStep(index + 1);
      } else if (event.key === "ArrowLeft" && index > 0) {
        event.preventDefault();
        goToStep(index - 1);
      }
    };
    element.addEventListener("keydown", onKeyDown);
    return () => element.removeEventListener("keydown", onKeyDown);
  }, [goToStep, index]);

  const modeLabel =
    mode === "demo"
      ? "Demonstração guiada · clique para testar"
      : mode === "paused"
        ? "Demonstração pausada"
        : "Modo interativo · sua vez";

  return (
    <dialog
      ref={dialog}
      className="tutorial"
      aria-labelledby="tutorial-title"
      aria-describedby="tutorial-description"
      onCancel={event => {
        event.preventDefault();
        dismiss();
      }}
    >
      <header className="tutorial-top">
        <nav className="tutorial-steps" aria-label="Passos do tutorial">
          {steps.map((item, idx) => (
            <button
              key={item.id}
              type="button"
              className={`tutorial-step-tab${idx === index ? " is-active" : ""}${idx < index ? " is-done" : ""}`}
              aria-current={idx === index ? "step" : undefined}
              onClick={() => goToStep(idx)}
            >
              <span className="tutorial-step-num">0{idx + 1}</span>
              <span>{item.shortLabel}</span>
            </button>
          ))}
        </nav>
        <button type="button" className="tutorial-close" aria-label="Fechar tutorial" onClick={dismiss}>
          ×
        </button>
      </header>

      <div className="tutorial-step" key={step.id}>
        <div className="tutorial-head">
          <div>
            <p className="tutorial-progress" id="tutorial-progress">
              Passo {index + 1} de {steps.length} · <span>{step.hint}</span>
            </p>
            <h2
              id="tutorial-title"
              ref={heading}
              tabIndex={-1}
              aria-describedby="tutorial-progress tutorial-description"
            >
              {step.title}
            </h2>
          </div>
        </div>
        <p id="tutorial-description">{step.body}</p>

        <section className="tut-stage-shell" aria-label={`Demonstração interativa: ${step.title}`}>
          <div className="tut-stage-bar">
            <span className={`tut-mode is-${mode}`}>
              <i aria-hidden="true" />
              {modeLabel}
            </span>
            <span className="tut-example-tag">Exemplo ilustrativo</span>
            <div className="tut-stage-tools">
              <button type="button" onClick={handleTogglePlay}>
                {mode === "demo" ? "Pausar" : "Reproduzir"}
              </button>
              <button type="button" onClick={handleReplay}>
                Repetir
              </button>
            </div>
          </div>

          <div className="tut-stage" ref={stageRef}>
            <Scene
              mode={mode}
              replaySeq={replaySeq}
              beatJump={beatJump}
              onBeatChange={setActiveBeat}
              onCursorChange={setCursor}
              onLayoutTick={handleLayoutTick}
              onUserInteract={handleUserInteract}
            />
            <TutorialCursor
              stageRef={stageRef}
              cursor={cursor}
              layoutKey={`${step.id}-${activeBeat}-${layoutTick}`}
            />
          </div>

          <ol className="tut-beats" aria-label="Etapas desta demonstração">
            {step.beats.map((label, beatIdx) => (
              <li key={label}>
                <button
                  type="button"
                  className={beatIdx === activeBeat ? "is-active" : beatIdx < activeBeat && mode === "demo" ? "is-passed" : undefined}
                  aria-pressed={beatIdx === activeBeat}
                  onClick={() => handleBeatClick(beatIdx)}
                >
                  <span>{beatIdx + 1}.</span> {label}
                </button>
              </li>
            ))}
          </ol>
        </section>
      </div>

      <footer className="tutorial-foot">
        <div className="tutorial-nav-actions">
          {index > 0 && (
            <button type="button" onClick={() => goToStep(index - 1)}>
              ← Voltar
            </button>
          )}
          <button
            type="button"
            className="primary"
            onClick={() => (last ? dismiss() : goToStep(index + 1))}
          >
            {last ? "Concluir" : "Próximo →"}
          </button>
          <button type="button" className="tutorial-try" onClick={handleTryInApp}>
            {step.tryAction.label} ↗
          </button>
        </div>
        <button type="button" className="tutorial-skip" onClick={dismiss}>
          Pular tutorial
        </button>
      </footer>
    </dialog>
  );
}
