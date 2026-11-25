"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Chart, { type PlotFigure } from "./Chart";
import { Idle, Loading, Metrics, Notice, Pane, Table } from "./ui";
import type { Column, Metric } from "./ui";
import { api } from "../lib/api";
import { formatPercent } from "../lib/formatting";

type Contributor = { ticker: string; gross_brl: number; net_brl: number };
type CalendarMonth = { month_label: string; month_index: number; year: number; gross_brl: number; net_brl: number; contributors: Contributor[] };
type IncomeAsset = {
  ticker: string; asset_class: string; asset_class_label: string; currency: string; weight: number;
  position_value_brl: number; income_mode: string; status: string; distributions_12m: number;
  dividend_per_share_brl: number; shares_estimate: number | null; price_brl: number | null; price_date: string | null;
  annual_gross_brl: number; annual_net_brl: number; monthly_gross_brl: number[]; monthly_net_brl: number[];
  tax_rate: number; tax_note: string | null; yield_gross: number; yield_net: number;
  source: string | null; note: string | null;
};
type IncomeResult = {
  portfolio_id: string; revision: number; as_of: string; capital: number; cost_basis_brl: number | null; jcp_share: number;
  currency: string;
  window: { trailing_start: string; trailing_end: string; projection_start: string; projection_end: string };
  annual_gross_brl: number; annual_net_brl: number; monthly_average_net_brl: number;
  portfolio_yield_gross: number; portfolio_yield_net: number; yield_on_cost_net: number | null;
  cdi_annual_rate: number | null; cdi_net_rate: number | null; cdi_ir_rate: number; yield_vs_cdi_ratio: number | null;
  assets: IncomeAsset[]; calendar: CalendarMonth[]; trailing_calendar: CalendarMonth[];
  missing_tickers: string[]; no_distribution_tickers: string[]; not_applicable_tickers: string[];
  chart: PlotFigure; methodology: string; tax_notes: string[];
};

type IncomeCalendarProps = { revision: number; portfolioId: string; capital: number };

const dateText = new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "2-digit", year: "numeric" });

function formatDate(value: string | null): string {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "—" : dateText.format(parsed);
}

function parseNumber(text: string): number | null {
  const cleaned = text.trim().replace(/\s/g, "").replace(",", ".");
  if (!cleaned) return null;
  const value = Number(cleaned);
  return Number.isFinite(value) ? value : null;
}

function statusLabel(asset: IncomeAsset): string {
  if (asset.status === "ok") return "Projetado";
  if (asset.status === "modeled") return "Premissa de modelo";
  if (asset.status === "missing") return "Sem dados";
  if (asset.income_mode === "no_intermediate_coupons") return "Sem cupons";
  if (asset.status === "no_distributions") return "Sem distribuições";
  return "Não aplicável";
}

const columns: Column[] = [
  { key: "ticker", label: "Ativo" },
  { key: "asset_class_label", label: "Classe" },
  { key: "status_label", label: "Situação", unit: "text" },
  { key: "distributions_12m", label: "Proventos 12m", unit: "integer" },
  { key: "dividend_per_share_brl", label: "Por cota", unit: "money" },
  { key: "position_value_brl", label: "Posição", unit: "money" },
  { key: "annual_gross_brl", label: "Bruto 12m", unit: "money" },
  { key: "annual_net_brl", label: "Líquido 12m", unit: "money" },
  { key: "tax_rate", label: "IR na fonte", unit: "fraction" },
  { key: "yield_net", label: "Yield líquido", unit: "fraction" },
];

/** Unified proventos calendar: trailing distributions, forward seasonality and net cash flow in BRL. */
export function IncomeCalendar({ revision, portfolioId, capital }: IncomeCalendarProps) {
  const [costBasis, setCostBasis] = useState("");
  const [jcpPercent, setJcpPercent] = useState("0");
  const [result, setResult] = useState<IncomeResult | null>(null);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);

  useEffect(() => {
    generation.current += 1;
    setResult(null);
    setLoadedKey(null);
    setBusy(false);
    setError(null);
  }, [revision, portfolioId]);

  const cost = parseNumber(costBasis);
  const jcp = parseNumber(jcpPercent) ?? 0;
  const capitalValid = Number.isFinite(capital) && capital >= 100 && capital <= 10_000_000;
  const costValid = costBasis.trim() === "" || (cost !== null && cost > 0);
  const jcpValid = jcp >= 0 && jcp <= 100;
  const requestKey = `${revision}|${portfolioId}|${capital}|${costBasis.trim()}|${jcpPercent.trim()}`;
  const stale = !!result && loadedKey !== requestKey;

  const run = useCallback(async () => {
    if (!capitalValid || !costValid || !jcpValid) return;
    const request = ++generation.current;
    setBusy(true);
    setError(null);
    try {
      const value = await api<IncomeResult>("/api/income/calendar", {
        revision,
        portfolio_id: portfolioId,
        capital,
        ...(cost !== null ? { cost_basis_brl: cost } : {}),
        jcp_share: jcp / 100,
      });
      if (request === generation.current) {
        setResult(value);
        setLoadedKey(requestKey);
      }
    } catch (cause) {
      if (request === generation.current) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    } finally {
      if (request === generation.current) setBusy(false);
    }
  }, [capital, capitalValid, cost, costValid, jcp, jcpValid, portfolioId, requestKey, revision]);

  const metrics: Metric[] = result ? [
    { label: "Renda líquida 12m", value: result.annual_net_brl, unit: "money", compact: true, help: "Projeção pelo ritmo histórico, já líquida de IR na fonte." },
    { label: "Média mensal líquida", value: result.monthly_average_net_brl, unit: "money", compact: true, help: "Renda líquida anual dividida por 12." },
    { label: "Yield líquido", value: result.portfolio_yield_net, unit: "fraction", help: "Renda líquida anual sobre o capital informado." },
    {
      label: "Yield vs. CDI",
      value: result.yield_vs_cdi_ratio,
      unit: "multiple",
      help: result.cdi_annual_rate === null
        ? "CDI indisponível no BCB."
        : `Renda líquida sobre CDI líquido (Selic bruta de ${formatPercent(result.cdi_annual_rate, 2)} com premissa de ${formatPercent(result.cdi_ir_rate, 0)} de IR).`,
    },
  ] : [];

  const secondary: Metric[] = result ? [
    { label: "Renda bruta 12m", value: result.annual_gross_brl, unit: "money", compact: true },
    { label: "Yield bruto", value: result.portfolio_yield_gross, unit: "fraction" },
    { label: "Yield sobre custo (líq.)", value: result.yield_on_cost_net, unit: "fraction", help: result.yield_on_cost_net === null ? "Informe o custo de aquisição para calcular." : undefined },
    { label: "CDI anual (BCB)", value: result.cdi_annual_rate, unit: "fraction", digits: 2, help: "Selic anualizada do BCB, bruta de IR." },
  ] : [];

  const rows = result ? result.assets.map(asset => {
    const projected = asset.status === "ok" || asset.status === "modeled";
    return {
      ...asset,
      status_label: statusLabel(asset),
      distributions_12m: projected ? asset.distributions_12m : null,
      dividend_per_share_brl: projected ? asset.dividend_per_share_brl : null,
      annual_gross_brl: projected ? asset.annual_gross_brl : null,
      annual_net_brl: projected ? asset.annual_net_brl : null,
      tax_rate: projected ? asset.tax_rate : null,
      yield_net: projected ? asset.yield_net : null,
    };
  }) : [];

  const coupons = result?.assets.filter(asset => asset.income_mode === "no_intermediate_coupons") ?? [];
  const excluded = result?.assets.filter(asset => asset.status === "not_applicable" && asset.income_mode !== "no_intermediate_coupons") ?? [];
  const withoutData = result?.assets.filter(asset => asset.status === "no_distributions") ?? [];

  return <div className="board" style={{ gridColumn: "1 / -1" }}>
    <Pane
      title="Proventos e fluxo de caixa"
      className="span-12"
      meta={result ? `${formatDate(result.window.trailing_start)} a ${formatDate(result.window.trailing_end)} · projeção até ${formatDate(result.window.projection_end)}` : "Renda em reais, líquida de IR na fonte"}
    >
      <div className="param-group param-fields">
        <div className="param-pairs">
          <label>Custo de aquisição (opcional)
            <input type="number" min={0} step="0.01" inputMode="decimal" placeholder="Opcional" value={costBasis}
              disabled={busy} aria-invalid={costValid ? undefined : true}
              onChange={event => setCostBasis(event.target.value)} />
          </label>
          <label>% de JCP estimado
            <input type="number" min={0} max={100} step="1" inputMode="decimal" value={jcpPercent}
              disabled={busy} aria-invalid={jcpValid ? undefined : true}
              onChange={event => setJcpPercent(event.target.value)} />
          </label>
        </div>
        <p className="param-hint">
          O custo de aquisição habilita o yield sobre custo. O percentual de JCP divide os proventos de ações B3 entre dividendos isentos e JCP tributado em 15% na fonte. FIIs são isentos; ativos dos EUA sofrem retenção de 30%.
        </p>
        {!costValid && <p className="field-error">Informe um custo de aquisição positivo ou deixe em branco.</p>}
        {!jcpValid && <p className="field-error">Informe um percentual de JCP entre 0 e 100.</p>}
        {!capitalValid && <p className="field-error">Informe um capital entre R$ 100 e R$ 10.000.000.</p>}
      </div>
      <div className="param-run">
        <button type="button" className="primary" disabled={busy || !capitalValid || !costValid || !jcpValid} onClick={() => void run()}>
          {busy ? "Lendo proventos…" : result ? "Atualizar calendário" : "Calcular calendário"}
        </button>
      </div>
      {error && <Notice tone="negative">{error}</Notice>}
      {stale && <Notice tone="warning">Parâmetros alterados. Atualize o calendário para recalcular.</Notice>}
      {busy && <Loading>Buscando proventos por ativo e montando o fluxo mensal em reais…</Loading>}
      {!result && !busy && <Idle>O calendário reúne os proventos dos últimos 12 meses completos e projeta a sazonalidade mensal sobre a carteira atual.</Idle>}
      {result && <>
        <Metrics items={metrics} size="lg" />
        <Metrics items={secondary} />
        <p className="pane-foot">
          Janela real de {formatDate(result.window.trailing_start)} a {formatDate(result.window.trailing_end)}; projeção de {formatDate(result.window.projection_start)} a {formatDate(result.window.projection_end)}. {result.methodology}
        </p>
      </>}
    </Pane>

    {result && <>
      {(result.missing_tickers.length > 0 || withoutData.length > 0 || coupons.length > 0 || excluded.length > 0) && <div className="span-12 board-notes">
        {result.missing_tickers.length > 0 && <Notice tone="warning">Sem dados de proventos ou preço atual para {result.missing_tickers.join(", ")}. Nenhum valor foi estimado para esses ativos.</Notice>}
        {withoutData.length > 0 && <Notice>Sem distribuições nos 12 meses analisados: {withoutData.map(asset => asset.ticker).join(", ")}.</Notice>}
        {coupons.map(asset => <Notice key={asset.ticker}>{asset.ticker}: {asset.note ?? "Título sem cupons semestrais — fluxo concentrado no vencimento."}</Notice>)}
        {excluded.length > 0 && <Notice>{excluded.map(asset => `${asset.ticker} (${asset.asset_class_label})`).join(", ")} não têm proventos de dividendos observáveis e ficam fora do fluxo.</Notice>}
      </div>}

      <Pane title="Fluxo mensal líquido" className="span-8" meta="Projeção pela sazonalidade dos 12 meses anteriores">
        <Chart figure={result.chart} title="Renda líquida projetada por mês, em reais" height={320} />
        <p className="pane-foot">Barras claras: renda bruta. Barras de destaque: renda líquida de IR na fonte. A projeção repete o ritmo histórico observado; não é garantia de pagamento.</p>
      </Pane>
      <Pane title="Leitura do calendário" className="span-4">
        <Metrics items={[
          { label: "Meses com proventos", value: result.calendar.filter(month => month.net_brl > 0).length, unit: "integer" },
          { label: "Maior mês líquido", value: result.calendar.reduce((best, month) => month.net_brl > best.net_brl ? month : best, result.calendar[0])?.net_brl ?? 0, unit: "money", compact: true },
          { label: "JCP estimado", value: result.jcp_share, unit: "fraction" },
          { label: "Custo de aquisição", value: result.cost_basis_brl, unit: "money", compact: true },
        ]} />
        <ul className="pane-foot">
          {result.tax_notes.map(note => <li key={note}>{note}</li>)}
        </ul>
      </Pane>

      <Pane id="income-por-ativo" title="Proventos por ativo" className="span-12" meta={`${result.assets.length} ativos na carteira`} flush>
        <Table caption="Proventos por ativo, valores em reais" columns={columns} rows={rows} />
        <p className="pane-foot">Valores por cota já convertidos para reais. A quantidade é estimada pelo valor da posição dividido pelo preço atual. “—” indica ativo sem provento aplicável, não provento zero.</p>
      </Pane>
    </>}
  </div>;
}

export default IncomeCalendar;
