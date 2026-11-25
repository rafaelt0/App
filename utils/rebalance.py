"""Deterministic buy-only monthly contribution planner ("Aporte do Mês").

The planner never sells.  It receives target weights from the owned portfolio,
current quantities and unadjusted trade prices (Yahoo purchase references or
Tesouro Direto PU Compra Manhã), then buys whole B3 lots, 0.01 Treasury titles,
or optionally 0.0001 US shares while reducing squared L2 weight drift.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_DOWN

from utils.assets import asset_currency, asset_metadata, normalize_ticker, supports_share_allocation
from utils.treasury_data import TREASURY_SHARE_STEP, TREASURY_TITLES

_LOT_STEP = Decimal("1")
_TREASURY_STEP = Decimal(str(TREASURY_SHARE_STEP))
_US_FRACTIONAL_STEP = Decimal("0.0001")
_ZERO = Decimal("0")
_MAX_WATERFILL_PASSES = 10_000


def _number(value: object, message: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(number):
        raise ValueError(message)
    return number


def _canonical(values: dict[str, float], message: str, *, aggregate: bool = True) -> dict[str, float]:
    """Normalise ticker keys and combine repeated entries deterministically."""
    result: dict[str, float] = {}
    for raw, value in (values or {}).items():
        ticker = normalize_ticker(raw)
        if not ticker:
            raise ValueError(message)
        number = _number(value, message)
        if number < 0:
            raise ValueError(message)
        result[ticker] = result.get(ticker, 0.0) + number if aggregate else number
    return result


def share_step(ticker: str, allow_fractional_us: bool = False) -> Decimal | None:
    """Lot size for one purchase step, or ``None`` when the asset has no share allocation."""
    canonical = normalize_ticker(ticker)
    if canonical in TREASURY_TITLES:
        return _TREASURY_STEP
    if not supports_share_allocation(canonical):
        return None
    is_us = (
        asset_metadata(canonical).get("asset_class") in {"acao_us", "reit"}
        or asset_currency(canonical) == "USD"
    )
    if is_us and allow_fractional_us:
        return _US_FRACTIONAL_STEP
    return _LOT_STEP


def _waterfill(budget: Decimal, deficits: dict[str, Decimal]) -> dict[str, Decimal]:
    """Continuous cash allocation that equalises the largest remaining target deficits."""
    if budget <= 0 or not deficits:
        return {}
    total = sum(deficits.values(), _ZERO)
    if total <= budget:
        return dict(deficits)
    ordered = sorted(deficits.items(), key=lambda item: (-item[1], item[0]))
    prefix = _ZERO
    level = _ZERO
    for index, (_, deficit) in enumerate(ordered):
        prefix += deficit
        count = index + 1
        level = (prefix - budget) / Decimal(count)
        next_deficit = ordered[index + 1][1] if index + 1 < len(ordered) else _ZERO
        if level >= next_deficit:
            break
    return {ticker: max(_ZERO, deficit - level) for ticker, deficit in deficits.items()}


def _as_float(value: Decimal, step: Decimal | None = None) -> float:
    if step is not None:
        value = value.quantize(step)
    return float(value)


def plan_buy_only_rebalance(
    target_weights: dict[str, float],
    current_holdings: dict[str, float] | None = None,
    trade_prices: dict[str, float] | None = None,
    contribution: float = 0.0,
    allow_fractional_us: bool = False,
) -> dict:
    """Return a buy-only monthly contribution ticket and the resulting drift.

    ``total_value_after`` includes any leftover cash, while ``post_weight`` is
    each asset's share of that full amount.  ``allocated_cash`` is always the
    sum of the emitted buy values and ``allocated_cash + leftover_cash`` equals
    the contribution.
    """
    targets = _canonical(
        target_weights,
        "Informe pesos-alvo finitos e não negativos com soma positiva.",
    )
    target_total = sum(targets.values())
    if not math.isfinite(target_total) or target_total <= 0:
        raise ValueError("Informe pesos-alvo finitos e não negativos com soma positiva.")
    targets = {ticker: weight / target_total for ticker, weight in targets.items()}

    holdings = _canonical(
        current_holdings or {},
        "As quantidades atuais devem ser finitas e não negativas.",
    )
    contribution_value = _number(contribution, "O aporte deve ser um valor positivo e finito.")
    if contribution_value <= 0:
        raise ValueError("O aporte deve ser um valor positivo e finito.")

    prices: dict[str, float] = {}
    invalid_prices: list[str] = []
    for raw, value in (trade_prices or {}).items():
        ticker = normalize_ticker(raw)
        try:
            number = float(value)
        except (TypeError, ValueError):
            invalid_prices.append(ticker or str(raw))
            continue
        if not math.isfinite(number) or number <= 0:
            invalid_prices.append(ticker or str(raw))
            continue
        prices[ticker] = number

    universe = sorted(set(targets) | set(holdings))
    contribution_decimal = Decimal(str(contribution_value))
    prices_decimal = {ticker: Decimal(str(price)) for ticker, price in prices.items()}
    current_values: dict[str, Decimal] = {}
    for ticker in universe:
        quantity = Decimal(str(holdings.get(ticker, 0.0)))
        price = prices_decimal.get(ticker)
        current_values[ticker] = quantity * price if price is not None else _ZERO
    total_before = sum(current_values.values(), _ZERO)
    total_after = total_before + contribution_decimal

    steps: dict[str, Decimal | None] = {
        ticker: share_step(ticker, allow_fractional_us) for ticker in universe
    }
    unit_costs: dict[str, Decimal] = {}
    eligible: list[str] = []
    for ticker in universe:
        if targets.get(ticker, 0.0) <= 0:
            continue
        step = steps[ticker]
        if step is None or ticker not in prices_decimal:
            continue
        unit_costs[ticker] = step * prices_decimal[ticker]
        if unit_costs[ticker] > 0:
            eligible.append(ticker)

    target_values = {
        ticker: total_after * Decimal(str(targets.get(ticker, 0.0))) for ticker in universe
    }

    allocated_units: dict[str, int] = {ticker: 0 for ticker in eligible}
    remaining = contribution_decimal
    for _ in range(_MAX_WATERFILL_PASSES):
        deficits: dict[str, Decimal] = {}
        for ticker in eligible:
            if unit_costs[ticker] > remaining:
                continue
            value_now = current_values[ticker] + Decimal(allocated_units[ticker]) * unit_costs[ticker]
            deficit = target_values[ticker] - value_now
            if deficit > 0:
                deficits[ticker] = deficit
        if not deficits:
            break
        continuous = _waterfill(remaining, deficits)
        progress = False
        for ticker, available in continuous.items():
            units = int((available / unit_costs[ticker]).to_integral_value(rounding=ROUND_DOWN))
            cap = int((deficits[ticker] / unit_costs[ticker]).to_integral_value(rounding=ROUND_DOWN))
            units = min(units, cap)
            if units <= 0:
                continue
            cost = Decimal(units) * unit_costs[ticker]
            if cost > remaining:
                continue
            allocated_units[ticker] += units
            remaining -= cost
            progress = True
        if progress:
            continue

        # Only fractional remains: buy the single unit with the best drift reduction.
        best: tuple[tuple[float, str], str] | None = None
        denominator = float(total_after)
        for ticker in deficits:
            value_now = current_values[ticker] + Decimal(allocated_units[ticker]) * unit_costs[ticker]
            weight_before = float(value_now) / denominator
            weight_after = float(value_now + unit_costs[ticker]) / denominator
            delta = (weight_after - targets[ticker]) ** 2 - (weight_before - targets[ticker]) ** 2
            if delta < 0:
                candidate = ((delta, ticker), ticker)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            break
        ticker = best[1]
        allocated_units[ticker] += 1
        remaining -= unit_costs[ticker]

    allocated_cash = contribution_decimal - remaining
    l2_before = 0.0
    l2_after = 0.0
    rows: list[dict] = []
    for ticker in universe:
        current_value = current_values[ticker]
        buy_value = Decimal(allocated_units.get(ticker, 0)) * unit_costs.get(ticker, _ZERO)
        post_value = current_value + buy_value
        current_weight = float(current_value / total_before) if total_before > 0 else 0.0
        post_weight = float(post_value / total_after) if total_after > 0 else 0.0
        target = targets.get(ticker, 0.0)
        drift_before = (current_weight - target) ** 2
        drift_after = (post_weight - target) ** 2
        l2_before += drift_before
        l2_after += drift_after
        step = steps[ticker]
        buy_quantity = (
            Decimal(allocated_units.get(ticker, 0)) * step if step is not None else _ZERO
        )
        current_quantity = holdings.get(ticker, 0.0)
        unit_price = prices.get(ticker)
        rows.append({
            "ticker": ticker,
            "current_qty": current_quantity,
            "current_value": float(current_value),
            "current_weight": current_weight,
            "target_weight": target,
            "buy_qty": _as_float(buy_quantity, step),
            "buy_value": float(buy_value),
            "post_qty": current_quantity + _as_float(buy_quantity, step),
            "post_value": float(post_value),
            "post_weight": post_weight,
            "drift_before": drift_before,
            "drift_after": drift_after,
            "unit_price": unit_price,
            "share_step": float(step) if step is not None else None,
        })

    warnings: list[str] = []
    invalid_price_set = set(invalid_prices)
    for ticker in universe:
        target = targets.get(ticker, 0.0)
        quantity = holdings.get(ticker, 0.0)
        step = steps[ticker]
        price = prices.get(ticker)
        if target > 0 and step is None:
            warnings.append(
                f"{ticker}: renda fixa modelada ou ativo sem cota negociável; não entra no aporte por cotas."
            )
        elif price is None and ticker not in invalid_price_set:
            if quantity > 0 and target > 0:
                warnings.append(
                    f"{ticker}: sem preço de negociação; a posição atual conta como zero e o ativo ficou fora do aporte."
                )
            elif quantity > 0:
                warnings.append(f"{ticker}: sem preço de negociação; a posição atual conta como zero.")
            elif target > 0:
                warnings.append(f"{ticker}: sem preço de negociação; o ativo ficou fora do aporte.")
    for ticker in sorted(invalid_price_set):
        warnings.append(f"{ticker}: preço de negociação inválido; o ativo ficou fora do aporte.")
    if not eligible:
        warnings.append("Nenhum ativo elegível tem preço de negociação para receber este aporte.")
    elif allocated_cash <= 0:
        warnings.append("O aporte não foi alocado: os lotes mínimos não couberam no valor informado.")
    elif remaining > 0:
        warnings.append("Sobrou caixa porque os lotes mínimos não couberam no aporte restante.")

    return {
        "rows": rows,
        "summary": {
            "contribution": contribution_value,
            "allocated_cash": float(allocated_cash),
            "leftover_cash": float(remaining),
            "l2_drift_before": l2_before,
            "l2_drift_after": l2_after,
            "total_value_before": float(total_before),
            "total_value_after": float(total_after),
            "zero_sell_guaranteed": True,
        },
        "warnings": warnings,
    }
