import type { PlotFigure } from "../components/Chart";

export type Strategy = "min_volatility" | "markowitz" | "hrp" | "manual";
export type Signal = "positive" | "negative" | "warning" | "neutral" | "info";
export const strategies: { value: Strategy; label: string }[] = [
  { value: "min_volatility", label: "Otimização de Mínima Volatilidade" },
  { value: "markowitz", label: "Otimização de Markowitz (Média-Variância)" },
  { value: "hrp", label: "Otimização Hierarchical Risk Parity (HRP)" },
  { value: "manual", label: "Alocação Manual" },
];
export interface PortfolioInputs {
  strategy: Strategy; capital: number; lookback_days: number;
  manual_shares: Record<string, number>; gamma: number;
}
export interface Sample { count: number; start: string; end: string }
export interface StressResult {
  rows: { crisis: string; period: string; portfolio: number; benchmark: number | null; difference_pp: number | null }[];
  warnings: string[]; gaps: Record<string, Record<string, string>>; chart: PlotFigure | null; history_start: string;
}
export interface PortfolioResult {
  id: string; portfolio_id: string; revision: number; saved: boolean;
  strategy: Strategy; mode: string; allocation_label: string; capital: number; lookback_days: number; tickers: string[];
  weights: { ticker: string; weight: number }[];
  sample: Sample & { selected_count: number; zero_excluded: boolean };
  comparison_sample: Sample | null;
  source: { adjusted_prices: string; trade_quotes: string; selic_source: string; selic_daily: number; selic_annual: number };
  trade_suggestions: { ticker: string; price: number | null; quote_date: string | null; shares: number | null; effective_value: number | null; suggested_value: number; suggested_weight: number; effective_weight: number | null }[];
  trade_totals: { allocated: number; cash: number; efficiency: number } | null;
  metrics: Record<string, number | null>; metric_status: Record<string, string>;
  monthly_returns: (Record<string, number | null> & { year: number })[];
  benchmark_available: boolean;
  charts: Record<string, PlotFigure>;
  warnings: string[];
  health: { score: number | null; coverage: number; label: string; diagnostics: { text: string; status: Signal }[] } | null;
  stress: StressResult | null;
  regime: { label: string; portfolio: number; benchmark: number; difference_pp: number; volatility: number; advice: string } | null;
  concentration: { hhi: number; effective_assets: number; selected_assets: number; largest_weight: number; level: string; diagnosis: string; diagnostic_status: Signal; largest_status: string } | null;
  asset_drawdowns: { ticker: string; max_drawdown: number; date: string }[];
}
export function period(sample: Sample): string {
  const format = (value: string) => value.split("-").reverse().join("/");
  return `${format(sample.start)} a ${format(sample.end)}`;
}
export function metricSignal(label: string): Signal {
  if (["Positivo", "Baixa", "Excelente", "Bom", "Controlado", "OK"].includes(label)) return "positive";
  if (["Negativo", "Alta", "Baixo", "Severo", "Concentrado", "Alto"].includes(label)) return "negative";
  return label === "N/D" ? "neutral" : "warning";
}
export function handoffTickers(raw: string, universe: Set<string>): string[] {
  return [...new Set(raw.split(",").map(value => value.trim().toUpperCase().replace(/\.SA$/, "")).filter(value => universe.has(value)))].slice(0, 20);
}
