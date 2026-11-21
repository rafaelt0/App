"use client";

import { useLayoutEffect, useRef, useState } from "react";

const steps = [
  {
    title: "Encontre ações",
    body: <>Na busca do topo, digite o nome de uma empresa ou um código como <strong>PETR4</strong> e escolha um resultado. Use o <strong>Screener</strong> para encontrar ações com filtros de liquidez e rentabilidade.</>,
  },
  {
    title: "Compare empresas",
    body: <>No <strong>Início</strong>, clique em <strong>Analisar</strong> para ver indicadores, saúde financeira e preços-alvo disponíveis. Selecione mais de uma ação para comparar os resultados.</>,
  },
  {
    title: "Monte e explore sua carteira",
    body: <>Selecione pelo menos dois ativos na <strong>Carteira</strong> e clique em <strong>Carregar portfólio</strong>. Use a <strong>Simulação</strong> para explorar cenários e <strong>Notícias</strong> para acompanhar os ativos da carteira.</>,
  },
];

export default function Tutorial({ onDismiss }: { onDismiss: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const [index, setIndex] = useState(0);
  const step = steps[index];
  const last = index === steps.length - 1;

  useLayoutEffect(() => {
    const element = dialog.current!;
    element.showModal();
    return () => element.close();
  }, []);

  useLayoutEffect(() => {
    heading.current?.focus();
  }, [index]);

  function dismiss() {
    dialog.current?.close();
    if (document.activeElement === document.body) {
      document.getElementById("conteudo")?.focus({ preventScroll: true });
    }
    onDismiss();
  }

  return <dialog ref={dialog} className="tutorial" aria-labelledby="tutorial-title" aria-describedby="tutorial-description"
    onCancel={event => { event.preventDefault(); dismiss(); }}>
    <div className="tutorial-step" key={index}>
      <p className="tutorial-progress" id="tutorial-progress">Passo {index + 1} de {steps.length}</p>
      <h2 id="tutorial-title" ref={heading} tabIndex={-1} aria-describedby="tutorial-progress tutorial-description">{step.title}</h2>
      <p id="tutorial-description">{step.body}</p>
    </div>
    <div className="onboarding-actions">
      {index > 0 && <button type="button" onClick={() => setIndex(index - 1)}>Voltar</button>}
      <button type="button" className="primary" onClick={() => last ? dismiss() : setIndex(index + 1)}>
        {last ? "Concluir" : "Próximo"}
      </button>
      <button type="button" className="tutorial-skip" onClick={dismiss}>Pular tutorial</button>
    </div>
  </dialog>;
}
