"""Parse pasted or uploaded broker position tables into canonical B3Lab holdings.

The importer accepts the two layouts users actually have at hand:

* B3 Área do Investidor exports, usually ``;``-separated and written with
  Brazilian decimal commas (``1.234,56``).
* Generic brokerage CSV/TSV files with English headers such as ``ticker``,
  ``quantity``, ``shares``, ``avg_price``, ``price`` and ``value``.

Rows are normalised through :func:`utils.assets.normalize_ticker`, verified
with :func:`utils.assets.is_valid_asset_ticker`, and duplicate tickers are
consolidated by summing quantities and weighting the unit price.  Invalid
rows are reported instead of being silently dropped.
"""

from __future__ import annotations

import csv
import io
import math
import re
import unicodedata

from utils.assets import is_valid_asset_ticker, normalize_ticker

MAX_IMPORT_BYTES = 200_000
MAX_IMPORT_ROWS = 500
_MAX_HEADER_SCAN = 40

# Alias rank decides which column wins when a file carries both a B3 trading
# code and a descriptive product name (or both close and average prices).
_ALIASES: dict[str, dict[str, int]] = {
    "ticker": {
        "codigo de negociacao": 0,
        "ticker": 0,
        "symbol": 0,
        "codigo": 1,
        "ativo": 1,
        "asset": 1,
        "papel": 2,
        "produto": 3,
    },
    "quantity": {
        "quantidade": 0,
        "qtd": 0,
        "qtde": 0,
        "quantity": 0,
        "shares": 0,
        "qty": 0,
        "quantidade disponivel": 1,
        "quantidade atual": 1,
        "posicao": 2,
        "position": 2,
    },
    "price": {
        "preco medio": 0,
        "avg price": 0,
        "average price": 0,
        "preco de fechamento": 1,
        "preco atual": 2,
        "price": 2,
        "preco unitario": 3,
        "unit price": 3,
        "preco": 4,
        "cotacao": 4,
        "close": 5,
        "closing price": 5,
    },
    "value": {
        "valor atualizado": 0,
        "valor total": 1,
        "market value": 1,
        "total value": 1,
        "valor": 2,
        "value": 2,
        "current value": 2,
        "valor da posicao": 2,
    },
}

# A header that only exists in the B3 export decides the reported format.
_B3_HEADER_KEYS = frozenset({
    "codigo de negociacao",
    "produto",
    "preco de fechamento",
    "preco medio",
    "valor atualizado",
})
_B3_TOKEN = re.compile(r"(?<![A-Z0-9])([A-Z]{4}\d{1,2})(?![A-Z0-9])")
_TREASURY_TOKEN = re.compile(r"(TD-[A-Z]+-\d{4})")
_DELIMITERS = (";", "\t", ",", "|")
_BLANK_MARKERS = {"", "-", "--", "—", "n/d", "nd", "na", "n/a"}


def _header_key(value: object) -> str:
    """Accent-, case- and punctuation-insensitive header key."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _parse_number(value: object) -> float | None:
    """Read a Brazilian or US decimal, tolerating currency marks and blanks."""
    if value is None:
        return None
    text = str(value).strip().replace("\u00a0", " ").replace(" ", "")
    if text.lower() in _BLANK_MARKERS:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"^R\$", "", text, flags=re.IGNORECASE).replace("%", "")
    cleaned = re.sub(r"[^0-9,.\-+]", "", text)
    if not cleaned or not re.search(r"\d", cleaned):
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".") if cleaned.count(",") == 1 else cleaned.replace(",", "")
    elif cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    elif re.fullmatch(r"[+-]?\d{1,3}\.\d{3}", cleaned):
        # ``1.234`` is a Brazilian thousands group, not a three-decimal value.
        cleaned = cleaned.replace(".", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if negative:
        number = -number
    return number if math.isfinite(number) else None


def _plausible_ticker(ticker: str) -> bool:
    """``is_valid_asset_ticker`` plus a shape check that rejects free text."""
    if not is_valid_asset_ticker(ticker):
        return False
    if re.fullmatch(r"[A-Z]{4}\d{1,2}", ticker):
        return True
    if re.fullmatch(r"[A-Z]{1,5}", ticker):
        return True
    return bool(re.search(r"[.\-=+^]", ticker))


def _resolve_ticker(raw: object) -> str | None:
    """Canonical ticker from a trading code or from a descriptive product name."""
    text = str(raw or "").strip()
    if not text:
        return None
    direct = normalize_ticker(text)
    if _plausible_ticker(direct):
        return direct
    upper = text.upper()
    for pattern in (_TREASURY_TOKEN, _B3_TOKEN):
        for candidate in pattern.findall(upper):
            canonical = normalize_ticker(candidate)
            if _plausible_ticker(canonical):
                return canonical
    return None


def _detect_header(lines: list[str]) -> tuple[int, str, dict[str, int], str]:
    """Find the header row and delimiter; return line index, delimiter, columns and format."""
    best: tuple[tuple[int, int, int, int], int, str, dict[str, int], str] | None = None
    for index, line in enumerate(lines[:_MAX_HEADER_SCAN]):
        if not line.strip():
            continue
        for order, delimiter in enumerate(_DELIMITERS):
            try:
                cells = next(csv.reader([line], delimiter=delimiter, skipinitialspace=True))
            except csv.Error:
                continue
            if len(cells) < 2:
                continue
            keys = [_header_key(cell) for cell in cells]
            columns: dict[str, int] = {}
            for field, aliases in _ALIASES.items():
                chosen: tuple[int, int] | None = None
                for cell_index, key in enumerate(keys):
                    rank = aliases.get(key)
                    if rank is not None and (chosen is None or rank < chosen[0]):
                        chosen = (rank, cell_index)
                if chosen is not None:
                    columns[field] = chosen[1]
            if "ticker" not in columns or "quantity" not in columns:
                continue
            format_detected = (
                "b3_area_investidor" if any(key in _B3_HEADER_KEYS for key in keys) else "generic_csv"
            )
            score = (len(columns), len(cells), -index, -order)
            candidate = (score, index, delimiter, columns, format_detected)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise ValueError(
            "Não foi possível identificar as colunas de ativo e quantidade. "
            "Use cabeçalhos como “Código de Negociação” e “Quantidade”, ou “ticker” e “quantity”."
        )
    _, index, delimiter, columns, format_detected = best
    return index, delimiter, columns, format_detected


def parse_positions(content: str | bytes) -> dict:
    """Parse a position table into ``positions``, ``skipped``, format and warnings.

    Raises ``ValueError`` with a Brazilian Portuguese message when the content
    is empty, too large, over the row limit, or has no recognisable columns.
    """
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("A tabela precisa estar em UTF-8. Salve o arquivo novamente e tente de novo.") from exc
    text = str(content or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("Cole ou envie uma tabela de posições para importar.")
    if len(text.encode("utf-8")) > MAX_IMPORT_BYTES:
        raise ValueError("A tabela excede o limite de 200 KB. Envie um arquivo menor.")

    lines = text.split("\n")
    header_index, delimiter, columns, format_detected = _detect_header(lines)
    reader = csv.reader(lines[header_index:], delimiter=delimiter, skipinitialspace=True)
    try:
        next(reader)
    except StopIteration as exc:  # pragma: no cover - _detect_header already found a row
        raise ValueError("A tabela não tem linhas de posição.") from exc
    raw_rows = [
        row for row in reader
        if any(str(cell).strip() for cell in row)
    ]
    if len(raw_rows) > MAX_IMPORT_ROWS:
        raise ValueError("A tabela excede o limite de 500 linhas. Envie uma posição mais enxuta.")

    aggregated: dict[str, dict[str, float]] = {}
    skipped: list[dict[str, object]] = []
    maximum_index = max(columns.values())

    for offset, row in enumerate(raw_rows, start=1):
        line_number = header_index + offset + 1
        if len(row) <= maximum_index:
            skipped.append({"line": line_number, "value": delimiter.join(row), "reason": "Linha incompleta."})
            continue
        ticker = _resolve_ticker(row[columns["ticker"]])
        if ticker is None:
            skipped.append({
                "line": line_number,
                "value": str(row[columns["ticker"]]).strip(),
                "reason": "Ativo não reconhecido.",
            })
            continue
        quantity = _parse_number(row[columns["quantity"]])
        if quantity is None:
            skipped.append({"line": line_number, "value": ticker, "reason": "Quantidade ausente ou inválida."})
            continue
        if quantity <= 0:
            skipped.append({"line": line_number, "value": ticker, "reason": "A quantidade deve ser positiva."})
            continue
        price = _parse_number(row[columns["price"]]) if "price" in columns else None
        reported_value = _parse_number(row[columns["value"]]) if "value" in columns else None
        if price is None and reported_value is not None and reported_value > 0:
            price = reported_value / quantity
        if price is not None and price <= 0:
            price = None

        entry = aggregated.get(ticker)
        if entry is None:
            entry = {
                "quantity": 0.0,
                "priced_quantity": 0.0,
                "priced_value": 0.0,
                "reported_value": 0.0,
                "rows": 0,
            }
            aggregated[ticker] = entry
        entry["quantity"] += quantity
        entry["rows"] += 1
        if reported_value is not None and reported_value > 0:
            entry["reported_value"] += reported_value
        if price is not None:
            entry["priced_quantity"] += quantity
            entry["priced_value"] += quantity * price

    if not aggregated:
        raise ValueError("Nenhuma posição válida foi encontrada na tabela. Confira os tickers e as quantidades.")

    positions: list[dict] = []
    warnings: list[str] = []
    for ticker in sorted(aggregated):
        entry = aggregated[ticker]
        quantity = entry["quantity"]
        if entry["priced_quantity"] > 0:
            price = entry["priced_value"] / entry["priced_quantity"]
        elif entry["reported_value"] > 0 and quantity > 0:
            price = entry["reported_value"] / quantity
        else:
            price = None
        value = quantity * price if price is not None else (entry["reported_value"] or None)
        positions.append({
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "value": value,
        })
        if entry["rows"] > 1:
            warnings.append(
                f"{ticker}: {int(entry['rows'])} linhas consolidadas (quantidade somada e preço médio ponderado)."
            )
    return {
        "positions": positions,
        "skipped": skipped,
        "format_detected": format_detected,
        "warnings": warnings,
    }
