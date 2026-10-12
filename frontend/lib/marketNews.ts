import { api } from "./api";
import { safeNewsLink } from "./news";

type MarketHeadline = {
  id: string;
  title: string;
  link: string | null;
  provider: string;
  published: string;
  importance_label: string;
};

export type MarketNews = {
  as_of: string;
  window_hours: number;
  articles: MarketHeadline[];
};

export async function loadMarketNews(): Promise<MarketNews> {
  const data = await api<unknown>("/api/news/market", undefined, "POST");
  const invalid = () => new Error("Não foi possível ler as notícias de mercado. Tente novamente.");
  if (!data || typeof data !== "object" || !("as_of" in data) || typeof data.as_of !== "string"
    || !Number.isFinite(Date.parse(data.as_of)) || !("window_hours" in data) || typeof data.window_hours !== "number"
    || !("articles" in data) || !Array.isArray(data.articles)) throw invalid();
  const articles: MarketHeadline[] = data.articles.map((item: unknown) => {
    if (!item || typeof item !== "object"
      || !("id" in item) || typeof item.id !== "string"
      || !("title" in item) || typeof item.title !== "string"
      || !("link" in item) || (item.link !== null && typeof item.link !== "string")
      || !("provider" in item) || typeof item.provider !== "string"
      || !("published" in item) || typeof item.published !== "string" || !Number.isFinite(Date.parse(item.published))
      || !("importance_label" in item) || typeof item.importance_label !== "string") throw invalid();
    return { id: item.id, title: item.title, link: safeNewsLink(item.link) ?? null,
      provider: item.provider, published: item.published, importance_label: item.importance_label };
  });
  return { as_of: data.as_of, window_hours: data.window_hours, articles };
}
