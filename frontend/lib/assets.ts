const assetLabels = {
  us_stock: "Ação americana",
  fii: "FII",
  etf: "ETF",
  treasury: "Título do Tesouro Direto",
  reit: "REIT",
};

export type CatalogAsset = {
  ticker: string;
  company: string;
  asset_class: keyof typeof assetLabels;
  market: "B3" | "US" | "TD";
};

export function assetLabel(asset: Pick<CatalogAsset, "asset_class" | "market">): string {
  const label = assetLabels[asset.asset_class];
  return asset.market === "TD" || asset.asset_class === "us_stock" ? label : `${label} · ${asset.market === "US" ? "EUA" : "B3"}`;
}

export function validQuantity(value: number, treasury: boolean): boolean {
  if (!Number.isFinite(value) || value < 0) return false;
  if (!treasury) return Number.isInteger(value);
  return Number(value.toFixed(2)) === value;
}
