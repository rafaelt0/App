type Searchable = { ticker: string; company: string; sector: string };

const combiningMarks = /[̀-ͯ]/g;
const wordSeparator = /[^\p{L}\p{N}]+/u;

/** Lowercase, accent-free text for matching; "Petróleo" and "petroleo" compare equal. */
export const normalizeSearch = (value: string) => value.normalize("NFD").replace(combiningMarks, "").toLocaleLowerCase("pt-BR");

type Sortable = { ticker: string; company: string; sector: string; liquidity: number | null };
export type StockSort = "liquidity" | "ticker" | "company" | "sector";
export const stockSorts: { id: StockSort; label: string }[] = [
  { id: "liquidity", label: "Liquidez" },
  { id: "ticker", label: "Ticker (A–Z)" },
  { id: "company", label: "Empresa (A–Z)" },
  { id: "sector", label: "Setor" },
];

const byTicker = (a: Sortable, b: Sortable) => a.ticker.localeCompare(b.ticker, "pt-BR");
const byLiquidity = (a: Sortable, b: Sortable) => (a.liquidity === null ? 1 : 0) - (b.liquidity === null ? 1 : 0) || (b.liquidity ?? 0) - (a.liquidity ?? 0) || byTicker(a, b);
const emptyLast = (a: string, b: string) => (a ? 0 : 1) - (b ? 0 : 1);
const stockComparators: Record<StockSort, (a: Sortable, b: Sortable) => number> = {
  liquidity: byLiquidity,
  ticker: byTicker,
  company: (a, b) => emptyLast(a.company, b.company) || a.company.localeCompare(b.company, "pt-BR", { sensitivity: "base" }) || byTicker(a, b),
  sector: (a, b) => emptyLast(a.sector, b.sector) || a.sector.localeCompare(b.sector, "pt-BR") || byLiquidity(a, b),
};

/** Returns a new array ordered by the chosen mode; ties always fall back to ticker order. */
export function sortStocks<T extends Sortable>(stocks: T[], sort: StockSort): T[] {
  return [...stocks].sort(stockComparators[sort]);
}

/**
 * Ranks stocks for a typed query: exact ticker, ticker prefix, company prefix, company word,
 * company substring, then sector. Without a query the input order is kept. The sort is stable,
 * so ties within a rank follow the input order; sort the input with sortStocks first.
 */
export function rankStocks<T extends Searchable>(stocks: T[], query: string, limit: number): { items: T[]; total: number } {
  const needle = normalizeSearch(query.trim());
  if (!needle) return { items: stocks.slice(0, limit), total: stocks.length };
  const ranked: { stock: T; rank: number }[] = [];
  for (const stock of stocks) {
    const ticker = normalizeSearch(stock.ticker);
    const company = normalizeSearch(stock.company ?? "");
    const rank = ticker === needle ? 0 : ticker.startsWith(needle) ? 1 : company.startsWith(needle) ? 2
      : company.split(wordSeparator).some(word => word.startsWith(needle)) ? 3 : company.includes(needle) ? 4
      : normalizeSearch(stock.sector ?? "").includes(needle) ? 5 : -1;
    if (rank >= 0) ranked.push({ stock, rank });
  }
  ranked.sort((a, b) => a.rank - b.rank);
  return { items: ranked.slice(0, limit).map(item => item.stock), total: ranked.length };
}
