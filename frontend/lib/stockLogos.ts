import issuers from "./stock-logo-manifest.json";

const logoByTicker = new Map<string, string>();

for (const issuer of issuers) {
  for (const ticker of issuer.tickers) {
    logoByTicker.set(ticker, `/stock-logos/${issuer.asset}`);
  }
}

export function stockLogoForTicker(ticker: string): string | undefined {
  return logoByTicker.get(ticker.trim().toUpperCase());
}
