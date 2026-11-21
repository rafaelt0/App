"use client";

import Link from "next/link";
import { useApp } from "./AppProvider";
import { Loading } from "./ui";

const overview = [
  { num: "01", title: "Buscar ativos", detail: "Use Ctrl K ou a barra do topo para montar uma seleção sincronizada." },
  { num: "02", title: "Analisar empresas", detail: "Compare indicadores, saúde financeira e preços-alvo no Início." },
  { num: "03", title: "Filtrar no Screener", detail: "Aplique presets ou critérios de DY, ROE, P/L e liquidez." },
  { num: "04", title: "Carteira e cenários", detail: "Otimize pesos, simule trajetórias Monte Carlo e acompanhe notícias." },
];

export default function Welcome() {
  const { loading, openTutorial } = useApp();

  return <section className="welcome" aria-labelledby="welcome-title">
    <div className="welcome-content">
      <h1 id="welcome-title">Boas-vindas ao B3Lab</h1>
      <p>Pesquise ações da B3, compare empresas e explore cenários para sua carteira.</p>
      {loading ? <Loading>Carregando a sessão e a lista de ativos…</Loading>
        : <>
          <div className="onboarding-actions">
            <button type="button" className="primary" onClick={openTutorial}>Iniciar tutorial interativo</button>
            <Link className="button-link" href="/">Ir para o Início <span aria-hidden="true">→</span></Link>
          </div>
          <ol className="welcome-steps" aria-label="Etapas do fluxo no B3Lab">
            {overview.map(item => (
              <li key={item.num}>
                <span>{item.num}</span>
                <strong>{item.title}</strong>
                <p>{item.detail}</p>
              </li>
            ))}
          </ol>
        </>}
    </div>
  </section>;
}
