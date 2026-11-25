"use client";

import { useEffect, useRef, useState } from "react";
import Chart from "./Chart";
import type { PlotFigure } from "./Chart";
import { Loading, Metrics, Notice, ScrollX, Table, ValueText, formatValue } from "./ui";
import { api } from "../lib/api";

type ShockKey = "selic_bps" | "ipca_bps" | "usd_brl_pct" | "us10y_bps" | "sp500_pct" | "dxy_pct";
type Shock = Record<ShockKey, number>;

type Preset = { key: string; label: string; shock: Shock };

type FactorContribution = { label: string; impact_brl: number; impact_pct: number };

type AssetImpact = {
  ticker: string;
  weight: number;
  impact_pct: number;
  impact_brl: number;
  method: "analytical_duration" | "factor_regression" | "fx_direct" | "insufficient_history";
  method_label: string;
  method_detail: string;
  r2_or_duration: number | null;
  betas: Record<string, number>;
  factors_used: string[];
  observations: number | null;
};

type MacroShockResult = {
  shock: Shock;
  capital: number;
  portfolio_impact_pct: number;
  portfolio_impact_brl: number;
  post_shock_capital_brl: number;
  worst_asset: AssetImpact | null;
  best_asset: AssetImpact | null;
  factor_contributions: Record<ShockKey, FactorContribution>;
  asset_impacts: AssetImpact[];
  presets: Preset[];
  missing_factors: string[];
  warnings: string[];
  methodology: string;
  as_of: string;
  chart: PlotFigure | null;
};

const FIELDS: { key: ShockKey; label: string; unit: string; min: number; max: number; step: number }[] = [
  { key: "selic_bps", label: "Selic", unit: "bps", min: -600, max: 600, step: 25 },
  { key: "ipca_bps", label: "IPCA", unit: "bps", min: -500, max: 500, step: 25 },
  { key: "usd_brl_pct", label: "USD/BRL", unit: "%", min: -40, max: 60, step: 1 },
  { key: "us10y_bps", label: "US 10Y", unit: "bps", min: -300, max: 300, step: 25 },
  { key: "sp500_pct", label: "S&P 500", unit: "%", min: -50, max: 50, step: 1 },
  { key: "dxy_pct", label: "DXY", unit: "%", min: -30, max: 30, step: 1 },
];

const ZERO_SHOCK: Shock = { selic_bps: 0, ipca_bps: 0, usd_brl_pct: 0, us10y_bps: 0, sp500_pct: 0, dxy_pct: 0 };

function asShock(value: Partial<Record<ShockKey, unknown>> | undefined): Shock {
  return FIELDS.reduce<Shock>((shock, field) => {
    const number = Number(value?.[field.key] ?? 0);
    shock[field.key] = Number.isFinite(number) ? number : 0;
    return shock;
  }, { ...ZERO_SHOCK });
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function signedGlyph(value: number): string {
  return value > 0 ? "▲" : value < 0 ? "▼" : "";
}

function tone(value: number): string {
  return value > 0 ? " is-pos" : value < 0 ? " is-neg" : "";
}

function numericOrZero(raw: string, min: number, max: number): number {
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? clamp(parsed, min, max) : 0;
}

/** Per-row duration/R² is a different unit per method; keep it readable instead of inventing one unit. */
function methodMeasure(row: AssetImpact): string {
  if (row.r2_or_duration === null || row.r2_or_duration === undefined) return "—";
  if (row.method === "analytical_duration") return `${row.r2_or_duration.toFixed(2)} anos`;
  return `R² ${row.r2_or_duration.toFixed(2)}`;
}

export default function MacroShockPanel({ revision, portfolioId, capital }: { revision: number; portfolioId: string; capital: number }) {
  const [shock, setShock] = useState<Shock>({ ...ZERO_SHOCK });
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetKey, setPresetKey] = useState<string | null>(null);
  const [presetError, setPresetError] = useState<string | null>(null);
  const [result, setResult] = useState<MacroShockResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const identity = useRef("");

  useEffect(() => {
    let cancelled = false;
    api<Preset[]>("/api/stress/presets")
      .then(value => {
        if (cancelled) return;
        const normalized = Array.isArray(value)
          ? value.map(preset => ({ key: String(preset.key), label: String(preset.label), shock: asShock(preset.shock) }))
          : [];
        setPresets(normalized);
        setPresetError(null);
      })
      .catch(reason => {
        if (!cancelled) setPresetError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const next = `${revision}:${portfolioId}`;
    if (identity.current === next) return;
    identity.current = next;
    generation.current += 1;
    setResult(null);
    setError(null);
    setPresetKey(null);
    setBusy(false);
  }, [revision, portfolioId]);

  useEffect(() => {
    // The capital is the base of every R$ figure, so a new base invalidates the displayed result.
    generation.current += 1;
    setResult(null);
    setBusy(false);
  }, [capital]);

  async function simulate(nextShock: Shock) {
    const current = ++generation.current;
    setBusy(true);
    setError(null);
    try {
      const value = await api<MacroShockResult>("/api/stress/macro", {
        revision,
        portfolio_id: portfolioId,
        capital,
        ...nextShock,
      });
      if (current === generation.current) setResult(value);
    } catch (reason) {
      if (current === generation.current) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (current === generation.current) setBusy(false);
    }
  }

  function changeField(key: ShockKey, value: number) {
    setShock(previous => ({ ...previous, [key]: value }));
    setPresetKey(null);
  }

  function applyPreset(preset: Preset) {
    setShock(preset.shock);
    setPresetKey(preset.key);
    setError(null);
    void simulate(preset.shock);
  }

  const worst = result?.worst_asset;
  const factorRows = result
    ? (Object.entries(result.factor_contributions) as [ShockKey, FactorContribution][]).map(([key, contribution]) => ({ key, ...contribution }))
    : [];

  return <div className="macro-shock">
    <div className="param-group">
      <p className="pane-lead">Aplique um cenário pronto ou ajuste os fatores. O impacto é calculado sobre a carteira carregada, sem previsão.</p>
      {presets.length > 0 && <div className="presets" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))" }} role="group" aria-label="Cenários macroeconômicos predefinidos">
        {presets.map(preset => <button type="button" key={preset.key} disabled={busy} aria-pressed={presetKey === preset.key} onClick={() => applyPreset(preset)}>{preset.label}</button>)}
      </div>}
      {presetError && <Notice tone="warning">Não foi possível carregar os cenários predefinidos ({presetError}). Ajuste os fatores manualmente.</Notice>}
      <div className="param-pairs" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
        {FIELDS.map(field => <label key={field.key}>{field.label} ({field.unit})
          <span style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 88px", gap: "8px", alignItems: "center" }}>
            <input type="range" min={field.min} max={field.max} step={field.step} value={shock[field.key]} disabled={busy}
              aria-label={`${field.label} em ${field.unit}`} onChange={event => changeField(field.key, numericOrZero(event.target.value, field.min, field.max))} />
            <input type="number" min={field.min} max={field.max} step={field.step} value={shock[field.key]} disabled={busy}
              aria-label={`${field.label} em ${field.unit}, valor exato`} onChange={event => changeField(field.key, numericOrZero(event.target.value, field.min, field.max))} />
          </span>
        </label>)}
      </div>
      <div className="param-run">
        <button type="button" className="primary" disabled={busy} onClick={() => { void simulate(shock); }}>{busy ? "Simulando…" : "Simular choque"}</button>
      </div>
      <p className="param-hint">Choques de Selic e IPCA precificam os títulos públicos; S&amp;P 500, USD/BRL, US 10Y e DXY movem os ativos de mercado; ativos em USD somam a tradução cambial.</p>
    </div>

    {error && <Notice tone="negative">{error}</Notice>}
    {busy && <Loading>Calculando o impacto do choque macroeconômico…</Loading>}

    {result && <>
      <Metrics size="lg" items={[
        { label: "Impacto no portfólio", value: result.portfolio_impact_pct, unit: "fraction", signed: true },
        { label: "Impacto em R$", value: result.portfolio_impact_brl, unit: "money", signed: true },
        { label: "Patrimônio pós-choque", value: result.post_shock_capital_brl, unit: "money" },
        { label: "Maior impacto", value: worst?.impact_pct ?? null, unit: "fraction", signed: true, help: worst ? `${worst.ticker}: ${formatValue(worst.impact_brl, "money", 2)}` : undefined },
      ]} />

      {result.warnings.map((warning, index) => <Notice tone="warning" key={index}>{warning}</Notice>)}

      {result.chart && <Chart figure={result.chart} height={Math.max(260, 44 * result.asset_impacts.length + 100)} />}

      <div className="split">
        <div>
          <h3 className="pane-sub" style={{ textTransform: "none", letterSpacing: 0, padding: "4px 0 0", color: "var(--text-2)", font: "600 13px var(--sans)" }}>Contribuição por fator</h3>
          <Table caption="Contribuição por fator" rowKey="key" columns={[
            { key: "label", label: "Fator", unit: "text" },
            { key: "impact_brl", label: "Contribuição (R$)", unit: "money", signed: true },
            { key: "impact_pct", label: "Contribuição", unit: "fraction", signed: true },
          ]} rows={factorRows} />
        </div>
        <div>
          <h3 className="pane-sub" style={{ textTransform: "none", letterSpacing: 0, padding: "4px 0 0", color: "var(--text-2)", font: "600 13px var(--sans)" }}>Impacto por ativo</h3>
          <ScrollX label="Impacto do choque por ativo">
            <table className="grid">
              <thead><tr>
                <th scope="col">Ativo</th>
                <th scope="col" className="num">Peso</th>
                <th scope="col" className="num">Impacto</th>
                <th scope="col" className="num">Impacto (R$)</th>
                <th scope="col">Método</th>
                <th scope="col" className="num">R² / duration</th>
              </tr></thead>
              <tbody>{result.asset_impacts.map(row => <tr key={row.ticker}>
                <th scope="row">{row.ticker}</th>
                <td className="num"><ValueText value={row.weight} unit="fraction" digits={1} /></td>
                <td className={`num${tone(row.impact_pct)}`}><span aria-hidden="true">{signedGlyph(row.impact_pct)}</span> <ValueText value={row.impact_pct} unit="fraction" digits={2} /></td>
                <td className={`num${tone(row.impact_brl)}`}><span aria-hidden="true">{signedGlyph(row.impact_brl)}</span> <ValueText value={row.impact_brl} unit="money" digits={2} /></td>
                <td><span className="asset-badge" title={row.method_detail}>{row.method_label}</span></td>
                <td className="num">{methodMeasure(row)}</td>
              </tr>)}</tbody>
            </table>
          </ScrollX>
        </div>
      </div>

      <p className="pane-foot">
        {result.methodology} Cenário em {result.as_of}; base de patrimônio {formatValue(result.capital, "money", 2)}.
        {result.missing_factors.length > 0 ? ` Fatores ausentes: ${result.missing_factors.join(", ")}.` : ""}
      </p>
    </>}
  </div>;
}
