import { formatValue } from "./ui";

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const position = (value: number, min: number, max: number) => `${Math.min(100, Math.max(0, (value - min) / (max - min || 1) * 100))}%`;

/** Where the quote sits inside the 52-week range; positions are presentation geometry over source values. */
export function PriceRange({ low, high, price, currency = "BRL", date }: { low: unknown; high: unknown; price: unknown; currency?: string; date?: string | null }) {
  if (!finite(low) || !finite(high) || !finite(price) || high <= low) return null;
  const money = (value: number) => formatValue(value, "money", 2, currency);
  const label = `Cotação ${money(price)}${date ? ` em ${date}` : ""}; mínima de 52 semanas ${money(low)}; máxima ${money(high)}.`;
  return <div className="range" role="img" aria-label={label}>
    <div className="range-head"><span>Faixa de 52 semanas</span>{date && <span>Fundamentus · {date.split("-").reverse().join("/")}</span>}</div>
    <div className="range-track"><span className="range-fill" style={{ width: position(price, low, high) }} /><span className="range-price" style={{ left: position(price, low, high) }} /></div>
    <div className="range-scale"><span>{money(low)}</span><span>{money(high)}</span></div>
  </div>;
}

/** Analyst target band (low to high), mean target and the source's own reference quote on one scale. */
export function TargetRange({ row }: { row: Record<string, unknown> }) {
  const { target_low: low, target_high: high, target_mean: mean, price } = row;
  const currency = typeof row.currency === "string" ? row.currency : "BRL";
  if (!finite(low) || !finite(high) || !finite(price)) return <span className="range-missing">Sem preço-alvo informado</span>;
  const min = Math.min(low, price), max = Math.max(high, price);
  const pad = (max - min) * .06 || 1;
  const [start, end] = [min - pad, max + pad];
  const money = (value: number) => formatValue(value, "money", 2, currency);
  const label = `Cotação ${money(price)}; alvo mínimo ${money(low)}${finite(mean) ? `, alvo médio ${money(mean)}` : ""}, alvo máximo ${money(high)}.`;
  return <div className="range range-target" role="img" aria-label={label}>
    <div className="range-track">
      <span className="range-band" style={{ left: position(low, start, end), width: `calc(${position(high, start, end)} - ${position(low, start, end)})` }} />
      {finite(mean) && <span className="range-mean" style={{ left: position(mean, start, end) }} />}
      <span className="range-price" style={{ left: position(price, start, end) }} />
    </div>
    <div className="range-scale"><span>{money(low)}</span>{finite(mean) && <span className="range-scale-mean">Médio {money(mean)}</span>}<span>{money(high)}</span></div>
  </div>;
}
