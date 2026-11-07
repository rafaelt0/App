"use client";
import type { ReactNode } from "react";
import { Metrics, Pane } from "./ui";

/* Schematic shapes for empty boards. Every sketch sits inside `.example`, which labels it as an illustration, and carries no numbers. */
const grid = "#1d2229", ink = "#3d4652", ink2 = "#8a929e", amber = "#ffb224";

function Example({ label, children, caption }: { label: string; children: ReactNode; caption: ReactNode }) {
  return <figure className="example"><svg viewBox="0 0 600 200" preserveAspectRatio="none" role="img" aria-label={`Ilustração de exemplo, sem dados: ${label}`}>
    {[50, 100, 150].map(y => <line key={y} x1="0" x2="600" y1={y} y2={y} stroke={grid} />)}
    {children}
  </svg><figcaption>{caption}</figcaption></figure>;
}

const fan = <>
  <path d="M0 110 C150 95 300 60 600 18 L600 192 C300 165 150 140 0 110 Z" fill="rgb(255 178 36 / .08)" />
  <path d="M0 110 C150 102 300 84 600 64 L600 150 C300 140 150 126 0 110 Z" fill="rgb(255 178 36 / .18)" />
  <path d="M0 110 C150 106 300 102 600 98" stroke={amber} strokeWidth="1.5" fill="none" vectorEffect="non-scaling-stroke" />
  <line x1="0" x2="600" y1="110" y2="110" stroke={ink2} strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
</>;
const histogram = <>{[4, 9, 18, 34, 58, 82, 96, 88, 70, 50, 34, 22, 13, 8, 5, 3].map((h, i) => <rect key={i} x={20 + i * 35} y={190 - h * 1.7} width="31" height={h * 1.7} fill={ink} />)}
  {[150, 230, 330].map(x => <line key={x} x1={x} x2={x} y1="20" y2="190" stroke={amber} strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />)}</>;
const paths = <>{["M0 100 C80 90 160 120 240 80 S420 60 600 40", "M0 100 C90 120 200 90 300 130 S480 110 600 150", "M0 100 C100 80 180 110 280 100 S460 70 600 90", "M0 100 C70 110 170 140 260 120 S430 160 600 170", "M0 100 C120 70 220 60 320 70 S500 30 600 20"].map(d => <path key={d} d={d} stroke={ink2} strokeWidth="1" fill="none" opacity=".7" vectorEffect="non-scaling-stroke" />)}</>;
const lines = <>
  <path d="M0 140 C60 130 100 150 160 132 S260 120 320 100 S440 90 480 70 S560 50 600 46" stroke={amber} strokeWidth="1.5" fill="none" vectorEffect="non-scaling-stroke" />
  <path d="M0 140 C70 145 120 138 180 140 S280 120 340 118 S460 100 520 96 S580 84 600 82" stroke={ink2} strokeWidth="1" strokeDasharray="5 4" fill="none" vectorEffect="non-scaling-stroke" />
</>;
const weights = <>{[0.92, 0.7, 0.52, 0.34, 0.16].map((w, i) => <g key={i}><rect x="0" y={14 + i * 36} width="80" height="20" fill={ink} /><rect x="100" y={18 + i * 36} width={w * 480} height="12" fill={i ? ink2 : amber} opacity={i ? .55 : 1} /></g>)}</>;
const heat = <>{Array.from({ length: 36 }, (_, i) => { const level = [3, 1, 4, 2, 0, 3, 1, 2, 4, 0, 2, 1][(i * 7) % 12]; return <rect key={i} x={(i % 12) * 50 + 1} y={Math.floor(i / 12) * 66 + 2} width="48" height="62" fill={ink2} opacity={.08 + level * .1} />; })}</>;

const portfolioMetrics = ["Retorno total", "Volatilidade anual", "Índice Sharpe", "Índice Sortino", "Queda máxima (max drawdown)"];

/** The Carteira board before the first run: every pane it will fill, with what it shows and a labelled sketch. */
export function PortfolioPreview({ status }: { status: ReactNode }) {
  return <>
    <Pane title="Resumo de desempenho" className="span-12" meta="aguardando cálculo">
      {status}
      <Metrics size="lg" items={portfolioMetrics.map(label => ({ label, value: null }))} />
    </Pane>
    <Pane title="Pesos do portfólio" className="span-4" meta="prévia">
      <Example label="pesos por ativo" caption="Peso de cada ativo, do maior para o menor, com a barra de composição e exportação em CSV.">{weights}</Example>
    </Pane>
    <Pane title="Evolução do valor" className="span-8" meta="prévia">
      <Example label="valor da carteira contra o IBOVESPA" caption="Valor histórico da carteira (linha cheia) contra o IBOVESPA (tracejada) nos pregões em comum, com pesos rebalanceados diariamente.">{lines}</Example>
    </Pane>
    <Pane title="Retornos mensais" className="span-6" meta="prévia">
      <Example label="mapa de retornos mensais" caption="Retorno de cada mês e do ano em %; a intensidade do fundo acompanha o tamanho do movimento.">{heat}</Example>
    </Pane>
    <Pane title="Sugestão de compra e risco" className="span-6" meta="prévia">
      <dl className="preview-list">
        <div><dt>Sugestão de compra</dt><dd>Cotas inteiras por ativo com o capital informado, pelo último fechamento bruto; total alocado, caixa e eficiência.</dd></div>
        <div><dt>Drawdown</dt><dd>Queda desde o pico da carteira e a pior queda de cada ativo, com data.</dd></div>
        <div><dt>Stress test</dt><dd>Resultado dos pesos atuais em crises históricas, contra o IBOVESPA.</dd></div>
        <div><dt>Risco</dt><dd>Regime dos últimos 21 pregões, concentração (HHI) e diagnóstico heurístico.</dd></div>
        <div><dt>Móveis</dt><dd>Beta e Sharpe em janelas móveis de 60 pregões.</dd></div>
      </dl>
    </Pane>
  </>;
}

const simulationMetrics = ["Valor esperado final", "CAGR do valor final médio (a.a.)", "Probabilidade de ganho", "Retorno final P5"];

/** The Simulação board before a run (or while the carteira is missing): the panes a run fills, sketched and labelled as examples. */
export function SimulationPreview({ status, meta }: { status: ReactNode; meta: string }) {
  return <>
    <Pane title="Estatísticas da simulação" className="span-12" meta={meta}>
      {status}
      <Metrics size="lg" items={simulationMetrics.map(label => ({ label, value: null }))} />
    </Pane>
    <Pane title="Faixas de percentis" className="span-8" meta="prévia">
      <Example label="faixas de percentis" caption="Faixa escura P25 a P75, clara P5 a P95, linha central a mediana e tracejado o capital inicial.">{fan}</Example>
    </Pane>
    <Pane title="Distribuição do valor final" className="span-4" meta="prévia">
      <Example label="histograma dos valores finais" caption="Histograma dos valores finais com Q1, mediana e Q3 marcados, mais mínimo, máximo, média e desvio padrão.">{histogram}</Example>
    </Pane>
    <Pane title="Trajetórias individuais" className="span-8" meta="prévia">
      <Example label="trajetórias simuladas" caption="Algumas trajetórias sorteadas, para ver a dispersão ao longo do horizonte; a quantidade exibida é ajustável.">{paths}</Example>
    </Pane>
    <Pane title="Síntese e modelos" className="span-4" meta="prévia">
      <dl className="preview-list">
        <div><dt>Síntese</dt><dd>Probabilidade de ganho, CAGR implícito, retorno final P5 e amplitude entre P5 e P95.</dd></div>
        <div><dt>Calibração</dt><dd>Média log anual com erro-padrão e sensibilidade à janela histórica.</dd></div>
        <div><dt>Modelos</dt><dd>Modelo normal multivariado contra reamostragem histórica de dias inteiros.</dd></div>
      </dl>
    </Pane>
  </>;
}
