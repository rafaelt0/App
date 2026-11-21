import Link from "next/link";
import { Loading } from "./ui";

export default function QuickGuide({ loading = false }: { loading?: boolean }) {
  return <section className="quick-guide" aria-labelledby="quick-guide-title">
    <div className="quick-guide-content">
      <header>
        <h1 id="quick-guide-title">Guia rápido do B3Lab</h1>
        <p>Da primeira busca à simulação da carteira.</p>
      </header>
      <ol className="quick-guide-steps">
        <li>
          <h2>Escolha os ativos</h2>
          <p>Busque pelo código ou nome da empresa. Use o Screener para encontrar ações por indicadores como P/L e ROE.</p>
        </li>
        <li>
          <h2>Analise os fundamentos</h2>
          <p>No Início, clique em <strong>Analisar</strong> para ver indicadores, comparar empresas do setor e consultar preços-alvo disponíveis.</p>
        </li>
        <li>
          <h2>Monte e simule a carteira</h2>
          <p>Na Carteira, selecione pelo menos dois ativos e clique em <strong>Carregar portfólio</strong>. Abra a Simulação para explorar cenários.</p>
        </li>
      </ol>
      {loading ? <Loading>Carregando a sessão e a lista de ativos.</Loading>
        : <Link className="button-link primary" href="/">Ir para o Início <span aria-hidden="true">→</span></Link>}
    </div>
  </section>;
}
