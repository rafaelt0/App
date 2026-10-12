type Searchable = { ticker: string; company: string; sector: string };

const combiningMarks = /[̀-ͯ]/g;
const wordSeparator = /[^\p{L}\p{N}]+/u;

/** Lowercase, accent-free text for matching; "Petróleo" and "petroleo" compare equal. */
export const normalizeSearch = (value: string) => value.normalize("NFD").replace(combiningMarks, "").toLocaleLowerCase("pt-BR");

/**
 * Ranks stocks for a typed query: exact ticker, ticker prefix, company prefix, company word,
 * company substring, then sector. Without a query the input order (liquidity) is kept.
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
