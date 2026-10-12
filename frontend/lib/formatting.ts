export type Unit = "money" | "fraction" | "percent" | "ratio" | "multiple" | "number" | "integer" | "text";

/** The one representation for a missing or non-finite value, in tables, metrics and text. */
export const missingValue = "—";
const minus = "−";

function localize(value: number, options: Intl.NumberFormatOptions): string {
  return new Intl.NumberFormat("pt-BR", options).format(value).replace(/-/g, minus);
}

export function formatValue(value: unknown, unit: Unit = "number", digits = 2, currency = "BRL", signed = false): string {
  if (value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value))) return missingValue;
  if (typeof value !== "number" || unit === "text") return String(value);
  const options: Intl.NumberFormatOptions = { signDisplay: signed ? "exceptZero" : "negative", minimumFractionDigits: unit === "integer" ? 0 : digits, maximumFractionDigits: unit === "integer" ? 0 : digits };
  if (unit === "money") { options.style = "currency"; options.currency = currency; }
  if (unit === "fraction") options.style = "percent";
  return localize(value, options) + (unit === "percent" ? "%" : unit === "multiple" ? "×" : "");
}

/** Shortens large counts and money for dense UI values. Callers keep the exact value in their data. */
export function formatCompactValue(value: unknown, unit: Unit = "number", digits = 1, currency = "BRL", signed = false): string {
  if (value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value))) return missingValue;
  if (typeof value !== "number" || !["money", "number", "integer"].includes(unit)) return formatValue(value, unit, undefined, currency, signed);

  const magnitude = Math.abs(value);
  const scale = magnitude >= 1_000_000_000 ? 1_000_000_000 : magnitude >= 1_000_000 ? 1_000_000 : magnitude >= 1_000 ? 1_000 : 1;
  if (scale === 1) return formatValue(value, unit, undefined, currency, signed);

  const compactUnit: Unit = unit === "integer" ? "number" : unit;
  const compactDigits = Math.min(1, Math.max(0, digits));
  const amount = formatValue(value / scale, compactUnit, compactDigits, currency, signed).replace(/,0$/, "");
  const suffix = scale === 1_000_000_000 ? " bi" : scale === 1_000_000 ? " mi" : " mil";
  return `${amount}${suffix}`;
}

/** Fraction shown as a percent with one explicit precision; use it wherever a share or return is printed. */
export function formatPercent(value: unknown, digits = 1): string {
  return formatValue(value, "fraction", digits);
}
