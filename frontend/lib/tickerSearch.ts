import type { AssetClass } from "../components/AppProvider";

type Searchable = { ticker: string; company: string; sector: string; asset_class?: AssetClass; currency?: string };

const combiningMarks = /[̀-ͯ]/g;
const wordSeparator = /[^\p{L}\p{N}]+/u;

export const ASSET_CLASS_BADGES: Record<AssetClass, string> = {
  acao_br: "Ação B3",
  fii: "FII",
  acao_us: "Ação EUA",
  reit: "REIT",
  titulo: "Título / RF",
  etf: "ETF / BDR",
  outros: "Global",
};

export const ASSET_CLASS_FILTERS: { key: "" | AssetClass; label: string }[] = [
  { key: "", label: "Todos" },
  { key: "acao_br", label: "Ações B3" },
  { key: "fii", label: "FIIs" },
  { key: "acao_us", label: "Ações EUA" },
  { key: "reit", label: "REITs" },
  { key: "titulo", label: "Títulos e RF" },
  { key: "etf", label: "ETFs e BDRs" },
  { key: "outros", label: "Cripto, commodities e câmbio" },
];

export function assetBadge(stock: { asset_class?: AssetClass; currency?: string }): string {
  const base = stock.asset_class ? ASSET_CLASS_BADGES[stock.asset_class] ?? "Ativo" : "Ação B3";
  return stock.currency === "USD" && !base.includes("EUA") ? `${base} · USD` : base;
}

/** Lowercase, accent-free text for matching; "Petróleo" and "petroleo" compare equal. */
export const normalizeSearch = (value: string) => value.normalize("NFD").replace(combiningMarks, "").toLocaleLowerCase("pt-BR");

type Sortable = { ticker: string; company: string; sector: string; liquidity: number | null; asset_class?: AssetClass; currency?: string };
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
 * company substring, then sector or asset class. Without a query the input order is kept.
 */
export function rankStocks<T extends Searchable>(stocks: T[], query: string, limit: number): { items: T[]; total: number } {
  const needle = normalizeSearch(query.trim());
  if (!needle) return { items: stocks.slice(0, limit), total: stocks.length };
  const category = /^(acao|acoes) americanas?$/.test(needle) ? "acao_us"
    : needle === "fiis" ? "fii" : needle === "etfs" ? "etf" : needle === "reits" ? "reit" : undefined;
  const ranked: { stock: T; rank: number }[] = [];
  for (const stock of stocks) {
    const ticker = normalizeSearch(stock.ticker);
    const company = normalizeSearch(stock.company ?? "");
    const badge = normalizeSearch(stock.asset_class ? ASSET_CLASS_BADGES[stock.asset_class] ?? "" : "");
    const rank = ticker === needle ? 0 : ticker.startsWith(needle) ? 1 : company.startsWith(needle) ? 2
      : company.split(wordSeparator).some(word => word.startsWith(needle)) ? 3 : company.includes(needle) ? 4
      : (category !== undefined && stock.asset_class === category) || normalizeSearch(stock.sector ?? "").includes(needle) || badge.includes(needle) ? 5 : -1;
    if (rank >= 0) ranked.push({ stock, rank });
  }
  ranked.sort((a, b) => a.rank - b.rank);
  return { items: ranked.slice(0, limit).map(item => item.stock), total: ranked.length };
}

const validCustomPattern = /^\^?[A-Z0-9][A-Z0-9.\-=+]{0,23}$/;

export function synthesizeCustomCandidate(rawQuery: string): Omit<Sortable, "liquidity"> | null {
  const cleaned = rawQuery.trim().toUpperCase().replace(/\.SA$/, "").replace(/,/g, ".");
  if (!cleaned || cleaned.length > 24) return null;
  const compact = cleaned.replace(/\s+/g, "");
  const cdiMatch = compact.match(/^(?:(CDB|LCI|LCA|CRI|CRA|DEB|RDB|LC)-?)?(\d{2,3}(?:\.\d+)?)%?CDI$/);
  if (cdiMatch) {
    const prefix = cdiMatch[1] || "CDB";
    const pct = cdiMatch[2];
    const ticker = `${prefix}-${pct}CDI`;
    return { ticker, company: `${prefix} (${pct}% do CDI)`, sector: "Renda Fixa · Pós-fixado (CDI)", asset_class: "titulo", currency: "BRL" };
  }
  const ipcaMatch = compact.match(/^(?:(DEB|CRI|CRA|TESOURO)-?)?IPCA\+(\d{1,2}(?:\.\d+)?)%?$/);
  if (ipcaMatch) {
    const rate = ipcaMatch[2];
    const ticker = `IPCA+${rate}`;
    return { ticker, company: `Renda Fixa (IPCA + ${rate.replace(".", ",")}% a.a.)`, sector: "Renda Fixa · Inflação (IPCA+)", asset_class: "titulo", currency: "BRL" };
  }
  const preMatch = compact.match(/^(?:PRE|PREFIXADO|LTN)-?(\d{1,2}(?:\.\d+)?)%?$/);
  if (preMatch) {
    const rate = preMatch[1];
    const ticker = `PRE-${rate}`;
    return { ticker, company: `Renda Fixa Prefixada (${rate.replace(".", ",")}% a.a.)`, sector: "Renda Fixa · Prefixado", asset_class: "titulo", currency: "BRL" };
  }
  if (!validCustomPattern.test(compact)) return null;
  const isB3 = /^[A-Z]{4}(?:3|4|5|6|11|31|32|33|34|35|39)$/.test(compact);
  const isFii = isB3 && compact.endsWith("11");
  const marketOnly = compact.startsWith("^") || /(?:=F|=X|-USD)$/.test(compact);
  return {
    ticker: compact,
    company: compact,
    sector: isFii ? "FII · Fundo Imobiliário B3" : isB3 ? "Ações / BDR B3" : "Ativo Global (Yahoo Finance)",
    asset_class: isFii ? "fii" : isB3 ? "acao_br" : marketOnly ? "outros" : "acao_us",
    currency: isB3 || ["^BVSP","BRL=X","USDBRL=X"].includes(compact) ? "BRL" : "USD",
  };
}
