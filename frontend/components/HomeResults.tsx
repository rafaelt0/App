"use client";
import { useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { Diagnostics, ExportButton, formatValue, Metrics, Notice, Pane, ScrollX, Table } from "./ui";
import { TargetRange } from "./RangeBar";
import { tickerColor } from "./Shell";
import type { Column } from "./ui";
import { consensusColumns, marketColumns } from "../lib/home";
import type { Company, HomeResult } from "../lib/home";

type Props = { ready: HomeResult; selection: string[]; busy: boolean; onRefreshConsensus: () => void };
type TabContext = { ready: HomeResult; companies: Company[]; single: boolean; busy: boolean; comparison: string; setComparison: (key: string) => void; onRefreshConsensus: () => void; color: (ticker: string) => string | undefined };
type ResultTab = { id: string; label: string; available: (ready: HomeResult, single: boolean) => boolean; render: (context: TabContext) => ReactNode };

const always = () => true;
const median = (values: number[]) => { const sorted = [...values].sort((a, b) => a - b); const middle = Math.floor(sorted.length / 2); return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2; };

type Bar = { ticker: string; value: number | null };
/**
 * Sorted horizontal bars with a dashed median. When the largest value is far above the rest, the scale follows the
 * second largest and that bar is drawn broken with its value spelled out, so it does not flatten the others.
 */
function ComparisonBars({ companies, indicator, color }: { companies: Company[]; indicator: { key: string; label: string; unit: "multiple" | "percent" }; color: (ticker: string) => string | undefined }) {
  const bars: Bar[] = companies.map(company => {
    const raw = company.indicators_available ? company.indicators[indicator.key] : null;
    return { ticker: company.ticker, value: typeof raw === "number" && Number.isFinite(raw) ? raw : null };
  });
  const known = bars.filter((bar): bar is { ticker: string; value: number } => bar.value !== null).sort((a, b) => b.value - a.value);
  const missing = bars.filter(bar => bar.value === null);
  const values = known.map(bar => bar.value);
  const text = (value: number) => formatValue(value, indicator.unit, 2);
  const outlier = known.length >= 3 && known[1].value > 0 && known[0].value > 2.5 * known[1].value;
  const high = Math.max(0, outlier ? known[1].value * 1.2 : Math.max(...values, 0));
  const low = Math.min(0, ...values);
  const span = high - low || 1;
  const at = (value: number) => (Math.min(Math.max(value, low), high) - low) / span * 100;
  const mid = values.length >= 2 ? median(values) : null;
  return <div className="cmp">
    <ul className="cmp-rows" aria-label={`${indicator.label} por ativo, do maior para o menor`}>
      {known.map((bar, index) => {
        const clipped = outlier && index === 0;
        const left = at(Math.min(0, bar.value)), right = at(Math.max(0, bar.value));
        return <li key={bar.ticker} className={clipped ? "is-clipped" : undefined}>
          <span className="cmp-tk"><i className="swatch" style={{ background: color(bar.ticker) ?? "#9aa2b4" }} aria-hidden="true" />{bar.ticker}</span>
          <span className="cmp-track" aria-hidden="true">
            {low < 0 && <span className="cmp-zero" style={{ left: `${at(0)}%` }} />}
            <span className="cmp-bar" style={{ left: `${left}%`, width: `${Math.max(right - left, .4)}%`, background: color(bar.ticker) ?? "#9aa2b4" }} />
            {clipped && <span className="cmp-break" />}
            {mid !== null && <span className="cmp-mid" style={{ left: `${at(mid)}%` }} />}
          </span>
          <span className="cmp-val">{text(bar.value)}{clipped && <small>fora da escala</small>}</span>
        </li>;
      })}
      {missing.map(bar => <li key={bar.ticker} className="is-missing"><span className="cmp-tk"><i className="swatch" style={{ background: color(bar.ticker) ?? "#9aa2b4" }} aria-hidden="true" />{bar.ticker}</span><span className="cmp-track" aria-hidden="true" /><span className="cmp-val muted">sem dado</span></li>)}
    </ul>
    <p className="cmp-scale" aria-hidden="true"><span>{low === 0 ? "0" : text(low)}</span>{mid !== null && <span className="cmp-scale-mid" style={{ left: `${at(mid)}%` }}>mediana {text(mid)}</span>}<span>{text(high)}</span></p>
    {outlier && <p className="tab-foot">Escala até {text(high)}, pelo segundo maior valor; a barra de {known[0].ticker} ({text(known[0].value)}) aparece cortada.</p>}
  </div>;
}

const resultTabs: ResultTab[] = [
  { id: "visao-geral", label: "Visão geral", available: always, render: ({ ready, companies, single, color }) => <>
    {single ? companies.map(company => company.indicators_available ? <div key={company.ticker} className="groups">{Object.entries(company.metric_groups).map(([group, metrics]) => <section className="group" key={group}><h4>{group}</h4><Metrics items={metrics} /></section>)}</div> : <Notice key={company.ticker}>Sem indicadores para {company.ticker}.</Notice>)
      : <><h4 className="tab-sub">Indicadores <small>% para rentabilidade, margens e crescimento · × para múltiplos · negrito ▴▾ maior e menor da coluna</small></h4>
        <Table caption="Indicadores financeiros" extremes swatch={row => color((row as { ticker: string }).ticker)} columns={[{ key: "ticker", label: "Ticker" }, ...ready.comparison_indicators.map(item => ({ key: item.key, label: item.label, unit: item.unit, digits: 2 } as Column))]} rows={companies.map(company => ({ ticker: company.ticker, ...company.indicators }))} /></>}
    <h4 className="tab-sub">Mercado <small>Fundamentus</small></h4>
    <Table caption="Informações de mercado" swatch={row => color((row as { ticker: string }).ticker)} columns={marketColumns} rows={companies.map(company => ({ ticker: company.ticker, ...company.market }))} />
  </> },
  { id: "saude", label: "Saúde financeira", available: always, render: ({ ready, companies }) => <>
    <p className="tab-lead">{ready.health_summary}</p>
    <div className="health">{companies.map(company => <section key={company.ticker} className="health-row"><h4>{company.ticker}<small>{company.identity.company}</small></h4>
      {company.health.metrics.length ? <><Metrics items={company.health.metrics} /><Diagnostics items={company.health.diagnostics} /></> : <Notice>Dados de endividamento não disponíveis via Fundamentus para este ticker.</Notice>}
    </section>)}</div>
  </> },
  { id: "comparacao", label: "Comparação", available: (_, single) => !single, render: ({ ready, companies, comparison, setComparison, color }) => {
    const choice = ready.comparison_indicators.find(item => item.key === comparison) ?? ready.comparison_indicators[0];
    return <>
      <div className="tab-tools"><label>Indicador<select value={choice.key} onChange={event => setComparison(event.target.value)}>{ready.comparison_indicators.map(item => <option value={item.key} key={item.key}>{item.label}</option>)}</select></label>
        <span className="muted">Do maior para o menor · tracejado âmbar: mediana da seleção</span></div>
      <ComparisonBars companies={companies} indicator={choice} color={color} />
    </>;
  } },
  { id: "preco-alvo", label: "Preço-alvo", available: always, render: ({ ready, busy, onRefreshConsensus }) => <>
    <div className="tab-tools"><button type="button" disabled={busy} onClick={onRefreshConsensus}>Atualizar consensos</button><span className="muted">Yahoo Finance · potencial sobre o alvo médio</span></div>
    {ready.consensus.has_targets && <ul className="ladder" aria-label="Faixas de preço-alvo por ativo">{ready.consensus.rows.map(row => {
      const upside = typeof row.upside === "number" && Number.isFinite(row.upside) ? row.upside : null;
      const tone = upside === null ? "neutral" : upside >= 0 ? "positive" : "negative";
      return <li key={String(row.ticker)}>
        <b>{String(row.ticker)}</b>
        <TargetRange row={row} />
        <span className={`chg is-${tone}`}>{upside === null ? "—" : <><span aria-hidden="true">{upside >= 0 ? "▲" : "▼"}</span>{upside > 0 ? "+" : ""}{formatValue(upside, "percent", 1)}</>}</span>
        <span className="muted">{typeof row.analyst_count === "number" ? `${row.analyst_count} ${row.analyst_count === 1 ? "opinião" : "opiniões"}` : "Opiniões —"}{typeof row.recommendation === "string" && row.recommendation ? ` · ${row.recommendation}` : ""}</span>
      </li>;
    })}</ul>}
    {ready.consensus.has_targets && <p className="legend"><i className="lg-band" aria-hidden="true" />Faixa alvo mínimo a máximo <i className="lg-mean" aria-hidden="true" />Alvo médio <i className="lg-price" aria-hidden="true" />Cotação de referência</p>}
    <Table caption="Consenso de preço-alvo" columns={consensusColumns} rows={ready.consensus.rows} />
    {ready.consensus.failed_tickers.length ? <Notice tone="warning">Consenso indisponível para: {ready.consensus.failed_tickers.join(", ")}.</Notice> : !ready.consensus.has_targets && <Notice>O Yahoo Finance não informa preços-alvo para as ações selecionadas.</Notice>}
    <p className="tab-foot">Opiniões e recomendações são agregadas pela fonte e não constituem recomendação de investimento.</p>
  </> },
  { id: "sintese", label: "Síntese", available: always, render: ({ companies }) => {
    const categories = Array.from(new Set(companies.flatMap(company => Object.keys(company.synthesis.categorias))));
    const name = (category: string) => category === "Yield" ? "Dividendos" : category;
    return <>
      <ScrollX label="Síntese setorial por ativo"><table className="grid synth-table">
        <thead><tr><th scope="col">Ativo</th><th scope="col">Setor</th><th scope="col" className="num">Áreas fav.</th>{categories.map(category => <th scope="col" key={category}>{name(category)}</th>)}</tr></thead>
        <tbody>{companies.map(company => <tr key={company.ticker}>
          <th scope="row">{company.ticker}</th>
          <td className="ell">{company.synthesis.sector ?? "Setor indisponível"}</td>
          <td className="num">{company.synthesis.favorable_categories}/{company.synthesis.categorias_validas}</td>
          {categories.map(category => {
            const entry = company.synthesis.categorias[category];
            if (!entry) return <td key={category} className="muted">—</td>;
            const tone = entry.veredicto === "Favorável" ? "fav" : entry.veredicto === "Desfavorável" ? "unfav" : "neutral";
            return <td key={category}><span className={`pctl is-${tone}`} role="meter" aria-label={`Percentil de ${name(category)}: ${entry.veredicto}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={entry.percentil}><span className="pctl-track"><span style={{ width: `${entry.percentil}%` }} /></span><span className="pctl-text">{entry.percentil.toFixed(0)}</span><span className="pctl-verdict">{tone === "fav" ? "fav." : tone === "unfav" ? "desf." : "neutro"}</span></span></td>;
          })}
        </tr>)}</tbody>
      </table></ScrollX>
      <p className="legend">Percentil no setor (0 a 100) e leitura da síntese: fav. favorável · neutro · desf. desfavorável, em âmbar</p>
      {companies.filter(company => company.synthesis.categorias_validas < 2).map(company => <Notice key={company.ticker}>{company.ticker}: dados insuficientes para uma conclusão setorial (são necessárias ao menos duas áreas com indicadores e três observações).</Notice>)}
      {companies.map(company => company.synthesis.details.length ? <details className="disclosure" key={company.ticker}><summary>{company.ticker} · {company.synthesis.details.length} indicadores frente ao setor</summary>
        <Table caption={`Indicadores de ${company.ticker} frente ao setor`} rowKey="label" columns={[{ key: "label", label: "Indicador" }, { key: "verdict", label: "Leitura" }, { key: "value", label: "Valor", unit: "number" }, { key: "unit", label: "Unid." }, { key: "median", label: "Mediana setor", unit: "number" }, { key: "mean", label: "Média aparada", unit: "number" }, { key: "percentile", label: "Percentil", unit: "number", digits: 1 }, { key: "observations", label: "Obs.", unit: "integer" }]} rows={company.synthesis.details.map(detail => ({ ...detail, unit: detail.unit === "percent" ? "%" : "×" }))} />
      </details> : null)}
      <p className="tab-foot">Percentil é a proporção de empresas do setor superadas, sem contar a própria; empates contam pela metade. Cada empresa entra uma vez, pela classe mais líquida. Média do setor aparada em 10% em cada ponta. Fonte: Fundamentus. Não constitui recomendação de investimento.</p>
    </>;
  } },
];

const tabKeys = ["ArrowLeft", "ArrowRight", "Home", "End"];

export default function HomeResults({ ready, selection, busy, onRefreshConsensus }: Props) {
  const [comparison, setComparison] = useState("pl");
  const [selectedTab, setSelectedTab] = useState(resultTabs[0].id);
  const single = selection.length === 1;
  const companies = ready.companies;
  const tabs = resultTabs.filter(tab => tab.available(ready, single));
  const activeTab = tabs.find(tab => tab.id === selectedTab) ?? tabs[0];
  const color = (ticker: string) => { const index = selection.indexOf(ticker); return index < 0 ? undefined : tickerColor(index); };
  function moveFocus(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    if (!tabKeys.includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    setSelectedTab(tabs[next].id);
    document.getElementById(`home-tab-${tabs[next].id}`)?.focus();
  }
  return <Pane title="Resultado da análise" area="res" className="pane-result" meta={`${companies.length} ${companies.length === 1 ? "ativo" : "ativos"} · Fundamentus`} actions={<ExportButton path={`/api/home/${ready.id}/fundamentals.csv`}>CSV</ExportButton>} flush>
    <div className="tabs" role="tablist" aria-label="Seções da análise">{tabs.map((tab, index) => {
      const selected = tab.id === activeTab.id;
      return <button type="button" role="tab" key={tab.id} id={`home-tab-${tab.id}`} aria-controls={`home-panel-${tab.id}`} aria-selected={selected} tabIndex={selected ? 0 : -1} onClick={() => setSelectedTab(tab.id)} onKeyDown={event => moveFocus(event, index)}>{tab.label}</button>;
    })}</div>
    {tabs.map(tab => <div className="tab-panel" role="tabpanel" key={tab.id} id={`home-panel-${tab.id}`} aria-labelledby={`home-tab-${tab.id}`} tabIndex={0} hidden={tab.id !== activeTab.id}>
      {tab.id === activeTab.id && tab.render({ ready, companies, single, busy, comparison, setComparison, onRefreshConsensus, color })}
    </div>)}
  </Pane>;
}
