import type { PlotFigure } from "../components/Chart";

export const intensityLevels = ["Alto", "Médio-Alto", "Médio", "Baixo-Médio", "Baixo", "Evidência limitada"] as const;
export const sortModes = ["Mais importantes", "Mais recentes", "Maior intensidade", "Mais otimistas", "Mais pessimistas"] as const;
export type NewsSort = typeof sortModes[number];
export type Sentiment = "Otimista" | "Neutro" | "Pessimista";
interface ArticleBase {
  id: string; tickers: string[]; title: string; link: string | null; provider: string;
  published: string; pub_time: string; summary: string; text_source: string;
  importance_rank: number; importance_label: string; sentiment: Sentiment; score: number;
  intensity: string; raw_text_length: number;
}
export type NewsArticle = ArticleBase & (
  {is_finbert: true; engine: "FinBERT-PT-BR"; scores: {label: string; score: number}[]; evidence_count: null; pos_terms: string[]; neg_terms: string[]} |
  {is_finbert: false; engine: "PLN Léxico (fallback)"; scores: []; evidence_count: number; pos_terms: string[]; neg_terms: string[]}
);
export interface NewsTone {
  ticker: string; state: string; count: number; positive: number; neutral: number; negative: number;
  mean: number | null; covered_share: number | null; contribution: number | null;
}
export interface NewsSnapshot {
  id: string; revision: number; portfolio_id: string; tickers: string[]; weights: Record<string, number>;
  as_of: string; window_days: number; max_per_ticker: number;
  feeds: {ticker: string; ok: boolean; recent_count: number}[];
  articles: NewsArticle[];
  model: {available: boolean; name: string; engines_used: string[]; retry_after: number | null};
  summary: {raw_score: number | null; score100: number | null; label: string; coverage: number;
    status: "positive" | "warning" | "negative" | "neutral";
    counts: {positive: number; neutral: number; negative: number}; observation_count: number;
    distribution: {ticker: string; positive: number; neutral: number; negative: number}[];
    tone: NewsTone[]; chart: PlotFigure};
}
export interface NewsFilters {ticker: string; sentiment: string; intensity: string; sort: NewsSort}
export const initialFilters: NewsFilters = {ticker: "Todos", sentiment: "Todos", intensity: "Todos", sort: "Mais importantes"};
export function filterArticles(articles: NewsArticle[], filters: NewsFilters): NewsArticle[] {
  const result = articles.filter(article =>
    (filters.ticker === "Todos" || article.tickers.includes(filters.ticker)) &&
    (filters.sentiment === "Todos" || article.sentiment === filters.sentiment) &&
    (filters.intensity === "Todos" || article.intensity === filters.intensity));
  switch (filters.sort) {
    case "Mais importantes": return result.sort((a, b) => b.importance_rank - a.importance_rank || Date.parse(b.published) - Date.parse(a.published));
    case "Mais recentes": return result.sort((a, b) => Date.parse(b.published) - Date.parse(a.published));
    case "Maior intensidade": return result.sort((a, b) => intensityLevels.indexOf(a.intensity as typeof intensityLevels[number]) - intensityLevels.indexOf(b.intensity as typeof intensityLevels[number]));
    case "Mais otimistas": return result.sort((a, b) => b.score - a.score);
    case "Mais pessimistas": return result.sort((a, b) => a.score - b.score);
  }
}
export function safeNewsLink(link: string | null): string | undefined {
  if (!link) return undefined;
  try { const url = new URL(link); return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : undefined; }
  catch { return undefined; }
}
