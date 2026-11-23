"use client";

import Link from "next/link";
import { useApp } from "./AppProvider";
import { Loading } from "./ui";

const overview = [
  { href: "/screener", title: "Screener", detail: "Encontre ações por dividendos, rentabilidade, preço ou liquidez." },
  { href: "/portfolio", title: "Carteira", detail: "Escolha ações, calcule a distribuição dos pesos e avalie o desempenho histórico." },
  { href: "/simulacao", title: "Simulação", detail: "Explore cenários futuros para uma carteira calculada." },
  { href: "/noticias", title: "Notícias", detail: "Leia as matérias recentes sobre os ativos da carteira e veja o tom da cobertura." },
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
            <Link className="button-link primary" href="/">Analisar ações <span aria-hidden="true">→</span></Link>
            <button type="button" onClick={openTutorial}>Ver tutorial interativo</button>
          </div>
          <nav aria-label="Ferramentas do B3Lab">
            <ul className="welcome-steps">
              {overview.map(item => <li key={item.href}>
                <Link href={item.href}><strong>{item.title} <span aria-hidden="true">→</span></strong><p>{item.detail}</p></Link>
              </li>)}
            </ul>
          </nav>
        </>}
    </div>
  </section>;
}
