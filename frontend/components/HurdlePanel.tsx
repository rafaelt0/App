"use client";

import { useEffect, useState } from "react";
import Chart, { type PlotFigure } from "./Chart";
import { Loading, Metrics, Notice, Table, type Column, type Metric } from "./ui";
import { api } from "../lib/api";

type UnitMetric = { value: number | null; unit: "fraction" | "pp" | "ratio" };

type AssetTaxRow = {
  ticker: string;
  asset_class: string;
  asset_class_label: string;
  tax_category: string;
  currency: string;
  tax_rule: string;
  tax_rule_label: string;
  applied_rate: number;
  weight: number;
  gross_return: number;
  taxable_gain: number;
  tax_amount: number;
  fx_drag_amount: number;
  total_drag: number;
};

type HurdleResponse = {
  metrics: Record<string, UnitMetric>;
  asset_breakdown: AssetTaxRow[];
  assumptions: string[];
  warnings: string[];
  chart: PlotFigure;
  regime: {
    base_currency: string;
    hurdle: string;
    hurdle_label: string;
    hurdle_kind: string;
    tax_profile: string;
    tax_profile_label: string;
    real_return: boolean;
    net_of_tax: boolean;
    holding_days: number;
    observations: number;
    inflation_source: string | null;
  };
  hurdle: { key: string; label: string; source: string; total_return_gross: number | null; total_return_net: number | null };
  sample: { start: string; end: string; observations: number };
  tax: { gross_return: number; tax_amount: number; fx_drag: number; total_drag: number; net_return: number };
};

type HurdleKey = "CDI" | "IPCA_PLUS_6" | "SOFR" | "US_CPI";
type TaxProfileKey = "BR_PF" | "US_TAXABLE" | "NONE";

const hurdleOptions: { value: HurdleKey; label: string }[] = [
  { value: "CDI", label: "CDI" },
  { value: "IPCA_PLUS_6", label: "IPCA + 6%" },
  { value: "SOFR", label: "SOFR" },
  { value: "US_CPI", label: "CPI EUA" },
];

const taxProfileOptions: { value: TaxProfileKey; label: string }[] = [
  { value: "BR_PF", label: "Brasil PF" },
  { value: "US_TAXABLE", label: "EUA Tributável" },
  { value: "NONE", label: "Isento" },
];

const columns: Column[] = [
  { key: "ticker", label: "Ativo" },
  { key: "asset_class_label", label: "Classe" },
  { key: "tax_rule_label", label: "Regra tributária" },
  { key: "weight", label: "Peso", unit: "fraction", digits: 1 },
  { key: "applied_rate", label: "Alíquota", unit: "fraction", digits: 1 },
  { key: "gross_return", label: "Retorno no período", unit: "fraction", digits: 2, signed: true },
  { key: "tax_amount", label: "IR estimado", unit: "fraction", digits: 2 },
  { key: "total_drag", label: "Custo total", unit: "fraction", digits: 2 },
];

function metric(items: Record<string, UnitMetric>, key: string): number | null {
  const value = items[key]?.value;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** After-tax and real-return hurdle comparison for the owned portfolio. */
export default function HurdlePanel({ revision, portfolioId }: { revision: number; portfolioId: string }) {
  const [realReturn, setRealReturn] = useState(false);
  const [netOfTax, setNetOfTax] = useState(true);
  const [hurdle, setHurdle] = useState<HurdleKey>("CDI");
  const [taxProfile, setTaxProfile] = useState<TaxProfileKey>("BR_PF");
  const [result, setResult] = useState<HurdleResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!portfolioId) return;
    let cancelled = false;
    setBusy(true);
    setError(null);
    api<HurdleResponse>("/api/portfolio/hurdle", {
      revision,
      portfolio_id: portfolioId,
      real_return: realReturn,
      net_of_tax: netOfTax,
      hurdle,
      tax_profile: taxProfile,
    })
      .then(value => { if (!cancelled) setResult(value); })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setResult(null);
        setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [hurdle, netOfTax, portfolioId, realReturn, revision, taxProfile]);

  const hurdleLabel = result?.regime.hurdle_label ?? "CDI";
  const metricItems: Metric[] = result ? [
    { label: realReturn ? "Retorno real total" : "Retorno total", value: metric(result.metrics, "total_return"), unit: "fraction", digits: 2, signed: true, help: "Retorno acumulado do portfólio no período analisado." },
    { label: "Retorno anualizado (CAGR)", value: metric(result.metrics, "cagr"), unit: "fraction", digits: 2, signed: true, help: "Retorno composto anualizado, base 252 dias úteis." },
    { label: `Hurdle · ${hurdleLabel}`, value: metric(result.metrics, "hurdle_total_return"), unit: "fraction", digits: 2, signed: true, help: result.hurdle.source },
    { label: "CAGR do hurdle", value: metric(result.metrics, "hurdle_cagr"), unit: "fraction", digits: 2, signed: true },
    { label: "Excesso de retorno", value: metric(result.metrics, "excess_return"), unit: "fraction", digits: 2, signed: true, help: "Retorno do portfólio menos o retorno do hurdle, em pontos percentuais do período." },
    { label: "Excesso anualizado", value: metric(result.metrics, "excess_cagr"), unit: "fraction", digits: 2, signed: true },
    { label: "Sharpe do excesso", value: metric(result.metrics, "excess_sharpe"), unit: "ratio", digits: 2, signed: true, help: "Excesso anualizado dividido pela volatilidade anualizada do portfólio." },
    { label: "IR efetivo sobre o ganho", value: metric(result.metrics, "effective_tax_rate"), unit: "fraction", digits: 1 },
    { label: "Custo tributário", value: metric(result.metrics, "tax_drag_pp"), unit: "percent", digits: 2, help: "IR estimado mais custo cambial/IOF, em pontos percentuais do período." },
    { label: "Volatilidade anualizada", value: metric(result.metrics, "volatility"), unit: "fraction", digits: 2 },
  ] : [];

  return <div className="param-group">
    <h3 className="pane-sub">Hurdle após impostos</h3>
    <p className="pane-lead">Compare o retorno da carteira com um benchmark depois do IR e, se quiser, acima da inflação.</p>
    <div className="segmented" role="group" aria-label="Ajustes do retorno">
      <button type="button" aria-pressed={realReturn} onClick={() => setRealReturn(value => !value)}>Retorno real (IPCA)</button>
      <button type="button" aria-pressed={netOfTax} onClick={() => setNetOfTax(value => !value)}>Líquido de IR</button>
    </div>
    <label>Benchmark / hurdle
      <select value={hurdle} onChange={event => setHurdle(event.target.value as HurdleKey)}>
        {hurdleOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
    <label>Regime tributário
      <select value={taxProfile} onChange={event => setTaxProfile(event.target.value as TaxProfileKey)}>
        {taxProfileOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
    {busy && <p role="status" className="param-hint">Comparando portfólio e hurdle…</p>}
    {error && <Notice tone="negative">{error}</Notice>}
    {!result && busy && <Loading>Calculando retorno após impostos…</Loading>}
    {result && <>
      <Metrics items={metricItems} />
      <Chart figure={result.chart} title="Riqueza acumulada base 100 — portfólio vs hurdle" height={320} />
      <p className="param-hint">
        {result.sample.start} a {result.sample.end} · {result.sample.observations} dias úteis · {result.regime.tax_profile_label}
        {result.regime.real_return ? " · retorno real" : " · retorno nominal"}
        {result.regime.net_of_tax ? " · líquido de IR" : " · bruto de IR"}
      </p>
      <Table columns={columns} rows={result.asset_breakdown} rowKey="ticker" caption="Tributação estimada por ativo" sortable />
      {result.warnings.length > 0 && <Notice tone="warning">{result.warnings.join(" ")}</Notice>}
      <details className="disclosure">
        <summary>Premissas do cálculo tributário</summary>
        <ul>{result.assumptions.map((line, index) => <li key={`${index}-${line}`}>{line}</li>)}</ul>
        {result.regime.inflation_source && <p className="param-hint">Inflação: {result.regime.inflation_source}.</p>}
      </details>
    </>}
  </div>;
}
