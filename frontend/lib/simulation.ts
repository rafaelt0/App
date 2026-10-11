import type { PlotFigure } from "../components/Chart";

export type CalibrationWindow = "portfolio" | 2 | 3 | 5;
export interface SimulationInputs {
  calibration: CalibrationWindow;
  simulations: number;
  initial_capital: number;
  years: number;
}
export const initialInputs: SimulationInputs = { calibration: "portfolio", simulations: 2000, initial_capital: 10000, years: 1 };
export const simulationPresets = [
  { label: "Rápida", simulations: 500, initial_capital: 10000, years: 1, description: "500 trajetórias · 1 ano (cauda menos estável)" },
  { label: "Padrão", simulations: 2000, initial_capital: 10000, years: 1, description: "2.000 trajetórias · 1 ano" },
  { label: "Longo prazo", simulations: 2000, initial_capital: 10000, years: 5, description: "2.000 trajetórias · 5 anos" },
];
export interface SimulationPortfolio {
  id: string | null;
  tickers: string[];
  weights: Record<string, number>;
  mode: string | null;
  revision: number;
  simulation_ready: boolean;
}
export interface CalibrationPeriod {
  count: number;
  start: string | null;
  end: string | null;
  annual_log_mean: number;
  iid_standard_error: number;
}
export interface SimulationResult {
  id: string;
  portfolio_id: string;
  revision: number;
  mode: string;
  inputs: SimulationInputs & { revision: number; portfolio_id: string };
  allocation: { ticker: string; weight: number }[];
  calibration: CalibrationPeriod & { window: CalibrationWindow; sensitivity: (CalibrationPeriod & { years: number })[] };
  stats: {
    mean_terminal: number; gain_probability: number; mean_implied_cagr: number; p75_implied_cagr: number;
    p5_terminal: number; p5_return: number; lower_tail_mean: number; minimum: number; maximum: number;
  };
  comparison: { model: string; p5: number; p95: number; gain_probability: number }[];
  distribution: { minimum: number; q1: number; median: number; q3: number; maximum: number; mean: number; standard_deviation: number };
  warnings: string[];
  monte_carlo_error_pp: number;
  spread_fraction: number;
  charts: { individual: PlotFigure; fan: PlotFigure; histogram: PlotFigure };
}
export function sameInputs(a: SimulationInputs, b: SimulationInputs): boolean {
  return a.calibration === b.calibration && a.simulations === b.simulations && a.initial_capital === b.initial_capital && a.years === b.years;
}
export function displayedTrajectories(figure: PlotFigure, count: number): PlotFigure {
  return { ...figure, data: figure.data.slice(0, count) };
}
export function periodLabel(period: { start: string | null; end: string | null }): string {
  const date = (value: string) => value.split("-").reverse().join("/");
  return period.start && period.end ? `${date(period.start)} a ${date(period.end)}` : "datas não disponíveis";
}
