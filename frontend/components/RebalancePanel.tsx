"use client";

import { useId, useState, type ChangeEvent } from "react";
import { api } from "../lib/api";
import { CapitalInput } from "./CapitalInput";
import { Loading, Metrics, Notice, Table, formatValue } from "./ui";
import type { Column, Metric } from "./ui";

type ImportedPosition = { ticker: string; quantity: number; price: number | null; value: number | null };
type SkippedRow = { line: number; value: string; reason: string };
type ImportResult = {
  positions: ImportedPosition[];
  skipped: SkippedRow[];
  format_detected: "b3_area_investidor" | "generic_csv";
  warnings: string[];
};
type PlanRow = {
  ticker: string;
  current_qty: number;
  current_value: number;
  current_weight: number;
  target_weight: number;
  buy_qty: number;
  buy_value: number;
  post_qty: number;
  post_value: number;
  post_weight: number;
  drift_before: number;
  drift_after: number;
  unit_price: number | null;
  share_step: number | null;
};
type PlanSummary = {
  contribution: number;
  allocated_cash: number;
  leftover_cash: number;
  l2_drift_before: number;
  l2_drift_after: number;
  total_value_before: number;
  total_value_after: number;
  zero_sell_guaranteed: boolean;
};
type PlanResult = {
  rows: PlanRow[];
  summary: PlanSummary;
  warnings: string[];
  portfolio_id: string;
  revision: number;
  price_source: string | null;
};

const quantityColumns: Column[] = [
  { key: "ticker", label: "Ativo" },
  { key: "unit_price", label: "Preço", unit: "money", digits: 2 },
  { key: "current_qty", label: "Posição atual", unit: "number", digits: 2 },
  { key: "buy_qty", label: "Comprar", unit: "number", digits: 2 },
  { key: "buy_value", label: "Valor da compra", unit: "money", digits: 2 },
  { key: "current_weight", label: "Peso atual", unit: "fraction", digits: 2 },
  { key: "target_weight", label: "Peso alvo", unit: "fraction", digits: 2 },
  { key: "post_weight", label: "Peso após", unit: "fraction", digits: 2 },
];

const quantityFormat = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 8 });

/** Reads a Brazilian or US decimal typed into a quantity field; blank means zero. */
function parseQuantity(text: string): number | null {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  if (!/\d/.test(trimmed)) return null;
  let normalized = trimmed.replace(/\s/g, "").replace(/^R\$/i, "").replace(/%/g, "");
  normalized = normalized.replace(/[^0-9,.\-+]/g, "");
  if (!normalized) return null;
  if (normalized.includes(",") && normalized.includes(".")) {
    normalized = normalized.lastIndexOf(",") > normalized.lastIndexOf(".")
      ? normalized.replace(/\./g, "").replace(",", ".")
      : normalized.replace(/,/g, "");
  } else if (normalized.includes(",")) {
    normalized = normalized.split(",").length === 2 ? normalized.replace(",", ".") : normalized.replace(/,/g, "");
  } else if ((normalized.match(/\./g) ?? []).length > 1 || /^[+-]?\d{1,3}\.\d{3}$/.test(normalized)) {
    normalized = normalized.replace(/\./g, "");
  }
  const value = Number(normalized);
  return Number.isFinite(value) ? value : null;
}

function WarningList({ items }: { items: string[] }) {
  if (!items.length) return null;
  return <Notice tone="warning"><div>{items.map((item, index) => <p key={`${index}-${item}`}>{item}</p>)}</div></Notice>;
}

export function RebalancePanel({ revision, portfolioId, tickers }: { revision: number; portfolioId: string; tickers: readonly string[] }) {
  const contentId = useId();
  const fileId = useId();
  const [content, setContent] = useState("");
  const [quantities, setQuantities] = useState<Record<string, string>>(() =>
    Object.fromEntries(tickers.map(ticker => [ticker, ""])));
  const [contribution, setContribution] = useState(1000);
  const [importing, setImporting] = useState(false);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [result, setResult] = useState<PlanResult | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const contributionValid = Number.isFinite(contribution) && contribution > 0 && contribution <= 10_000_000;
  const extraTickers = Object.keys(quantities).filter(ticker => !tickers.includes(ticker) && quantities[ticker]?.trim());

  async function importPositions() {
    if (!content.trim() || importing) return;
    setImporting(true);
    setError(null);
    try {
      const imported = await api<ImportResult>("/api/rebalance/import", { content });
      setImportResult(imported);
      setQuantities(Object.fromEntries(imported.positions.map(position => [position.ticker, quantityFormat.format(position.quantity)])));
      setResult(null);
      setFileError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setImporting(false);
    }
  }

  async function readFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;
    setFileError(null);
    if (file.size > 200_000) {
      setFileError("O arquivo excede o limite de 200 KB. Envie uma tabela menor.");
      return;
    }
    try {
      setContent(await file.text());
    } catch {
      setFileError("Não foi possível ler o arquivo. Verifique se ele está em UTF-8.");
    }
  }

  async function calculate() {
    if (planning || importing || !contributionValid) return;
    const currentQuantities: Record<string, number> = {};
    const names = [...new Set([...tickers, ...Object.keys(quantities)])];
    for (const ticker of names) {
      const parsed = parseQuantity(quantities[ticker] ?? "");
      if (parsed === null || parsed < 0) {
        setError(`Informe uma quantidade válida e não negativa para ${ticker}.`);
        return;
      }
      if (parsed > 0) currentQuantities[ticker] = parsed;
    }
    setPlanning(true);
    setError(null);
    setResult(null);
    try {
      const planned = await api<PlanResult>("/api/rebalance/plan", {
        revision,
        portfolio_id: portfolioId,
        contribution,
        current_quantities: currentQuantities,
        allow_fractional_us: false,
      });
      setResult(planned);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setPlanning(false);
    }
  }

  const buyRows = result?.rows.filter(row => row.buy_qty > 0) ?? [];
  const metrics: Metric[] = result ? [
    { label: "Aporte do mês", value: result.summary.contribution, unit: "money", digits: 2 },
    { label: "Alocado agora", value: result.summary.allocated_cash, unit: "money", digits: 2 },
    {
      label: "Sobra em caixa", value: result.summary.leftover_cash, unit: "money", digits: 2,
      status: result.summary.leftover_cash > 0 ? "warning" : "positive",
      classification: result.summary.leftover_cash > 0 ? "Sobra" : "Tudo alocado",
    },
    { label: "Desvio L2 antes", value: result.summary.l2_drift_before, unit: "number", digits: 5 },
    {
      label: "Desvio L2 depois", value: result.summary.l2_drift_after, unit: "number", digits: 5,
      status: result.summary.l2_drift_after <= result.summary.l2_drift_before ? "positive" : "negative",
      classification: result.summary.l2_drift_after <= result.summary.l2_drift_before ? "Melhor" : "Pior",
    },
    { label: "Valor após o aporte", value: result.summary.total_value_after, unit: "money", digits: 2 },
  ] : [];

  return <div className="param-group">
    <div className="param-group">
      <h3 className="pane-sub">Importar posições</h3>
      <p className="param-hint">Cole a tabela da Área do Investidor (B3) ou de outra corretora, ou envie o arquivo CSV/TSV. Aceita ponto e vírgula, tabulação e vírgula como separador.</p>
      <label htmlFor={contentId}>Tabela de posições<textarea id={contentId} rows={6} value={content} disabled={importing || planning}
        placeholder="Código de Negociação;Quantidade;Preço de Fechamento&#10;PETR4;100;38,50"
        onChange={event => setContent(event.currentTarget.value)} /></label>
      <label htmlFor={fileId}>Arquivo CSV/TSV<input id={fileId} type="file" accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain"
        disabled={importing || planning} onChange={event => void readFile(event)} /></label>
      {fileError && <Notice tone="negative">{fileError}</Notice>}
      <div className="param-run">
        <button type="button" disabled={!content.trim() || importing || planning} onClick={() => void importPositions()}>
          {importing ? "Importando…" : "Importar posições"}
        </button>
      </div>
      {importing && <Loading>Lendo a tabela de posições…</Loading>}
    </div>

    {importResult && <>
      <Notice tone={importResult.skipped.length ? "warning" : "positive"}>
        {importResult.positions.length} {importResult.positions.length === 1 ? "posição importada" : "posições importadas"} de {importResult.format_detected === "b3_area_investidor" ? "Área do Investidor (B3)" : "CSV de corretora"}.
      </Notice>
      <WarningList items={importResult.warnings} />
      {importResult.skipped.length > 0 && <Notice tone="warning">
        <div>
          {importResult.skipped.slice(0, 6).map(row => <p key={`${row.line}-${row.value}`}>Linha {row.line}: {row.value || "sem ativo"} — {row.reason}</p>)}
          {importResult.skipped.length > 6 && <p>Mais {importResult.skipped.length - 6} linha(s) ignorada(s).</p>}
        </div>
      </Notice>}
    </>}

    <div className="param-group">
      <h3 className="pane-sub">Quantidades atuais</h3>
      <p className="param-hint">Informe as cotas que você já possui. Ações, FIIs e ETFs usam cotas inteiras; títulos do Tesouro Direto usam múltiplos de 0,01.</p>
      <div className="param-pairs">
        {tickers.map(ticker => <label key={ticker}>Cotas de {ticker}<input type="text" inputMode="decimal" autoComplete="off"
          value={quantities[ticker] ?? ""} disabled={planning || importing} placeholder="0"
          onChange={event => setQuantities(previous => ({ ...previous, [ticker]: event.currentTarget.value }))} /></label>)}
      </div>
      {extraTickers.length > 0 && <p className="param-hint">Também entram no cálculo: {extraTickers.join(", ")}.</p>}
    </div>

    <div className="param-group">
      <h3 className="pane-sub">Aporte do mês</h3>
      <CapitalInput label="Aporte do mês em R$" value={contribution} disabled={planning || importing} max={10_000_000}
        help="Entre R$ 1,00 e R$ 10.000.000,00." onChange={setContribution}
        error={contributionValid ? undefined : "Informe um aporte entre R$ 1,00 e R$ 10.000.000,00."} />
      <p className="param-hint">O plano é só de compra: nenhum ativo é vendido.</p>
      <div className="param-run">
        <button type="button" className="primary" disabled={planning || importing || !contributionValid || tickers.length === 0}
          onClick={() => void calculate()}>
          {planning ? "Calculando…" : "Calcular aporte"}
        </button>
      </div>
      {planning && <Loading>Calculando o aporte e o desvio dos pesos…</Loading>}
    </div>

    {error && <Notice tone="negative">{error}</Notice>}

    {result && <>
      <Metrics items={metrics} />
      {buyRows.length > 0 ? <>
        <Table columns={quantityColumns} rows={buyRows} rowKey="ticker" sortable
          caption="Ordem de compra do aporte, com preço de negociação e pesos antes e depois" />
        <p className="param-hint">Total de {formatValue(result.summary.allocated_cash, "money", 2)} em {buyRows.length} {buyRows.length === 1 ? "ativo" : "ativos"}. O peso após considera o valor total informado, inclusive eventual caixa não alocado.</p>
      </> : <Notice tone="warning">Nenhum ativo recebeu compra com este aporte. Ajuste o valor ou confira os preços e as posições atuais.</Notice>}
      {result.price_source && <p className="param-hint">{result.price_source}.</p>}
      <WarningList items={result.warnings} />
    </>}
  </div>;
}
