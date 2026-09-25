from utils.portfolio_data import (
    align_benchmark_returns,
    align_weights_to_columns,
    bound_efficient_return,
    estimate_markowitz_inputs,
)


def test_weights_align_to_return_columns_by_ticker_not_position():
    import pandas as pd

    returns = pd.DataFrame({"BBB.SA": [-0.1], "AAA.SA": [0.2]})
    weights = {"AAA.SA": 0.8, "BBB.SA": 0.2}

    result = returns.dot(align_weights_to_columns(weights, returns.columns))

    assert result.iloc[0] == 0.14


def test_weights_alignment_fails_when_ticker_weight_is_missing():
    import pytest

    with pytest.raises(ValueError, match="AAA.SA"):
        align_weights_to_columns({"BBB.SA": 1.0}, ["AAA.SA"])


def test_bound_efficient_return_clamps_to_feasible_range():
    assert bound_efficient_return(2.0, 0.0, 1.0) == 0.999999
    assert bound_efficient_return(-2.0, 0.0, 1.0) == 0.000001


def test_bound_efficient_return_returns_none_for_non_finite_values():
    assert bound_efficient_return(float("nan"), 0.0, 1.0) is None
    assert bound_efficient_return(0.5, float("inf"), 1.0) is None


def test_bound_efficient_return_returns_none_for_degenerate_range():
    assert bound_efficient_return(0.5, 1.0, 1.0) is None


def test_historical_stress_calculates_available_crises_and_skips_missing_history():
    import pandas as pd

    from utils.portfolio_data import calculate_historical_stress

    dates = pd.bdate_range("2024-11-26", periods=7)
    prices = pd.DataFrame(
        {
            "AAA.SA": [100, 90, 80, 80, 80, 80, 80],
            "BBB.SA": [100, 100, 100, 100, 100, 100, 100],
        },
        index=dates,
    )
    benchmark = pd.Series(100.0, index=dates)
    crises = {
        "Crise fiscal brasileira (2024)": ("2024-11-26", "2024-12-30"),
        "Sem histórico": ("2018-01-01", "2018-01-31"),
    }

    results = calculate_historical_stress(
        prices, benchmark, {"AAA.SA": 0.5, "BBB.SA": 0.5}, crises
    )

    expected = (
        1 + prices.pct_change(fill_method=None).dropna().dot([0.5, 0.5])
    ).prod() - 1
    assert len(results) == 1
    assert results[0]["Crise"] == "Crise fiscal brasileira (2024)"
    assert abs(results[0]["Portfólio"] - expected) < 1e-12
    assert results[0]["IBOV"] == 0


def test_missing_benchmark_price_never_becomes_zero_return():
    import pandas as pd

    dates = pd.bdate_range("2025-01-01", periods=35)
    portfolio = pd.Series(0.01, index=dates)
    benchmark = pd.Series(100.0, index=dates)
    benchmark.iloc[10] = float("nan")
    benchmark.iloc[11:] = 110.0

    aligned_portfolio, aligned_benchmark = align_benchmark_returns(portfolio, benchmark)

    assert dates[10] not in aligned_benchmark.index
    assert dates[11] not in aligned_benchmark.index
    assert len(aligned_portfolio) == len(aligned_benchmark) == 32


def test_markowitz_inputs_ignore_missing_quote_gap():
    import pandas as pd

    dates = pd.bdate_range("2025-01-01", periods=40)
    prices = pd.DataFrame(
        {
            "AAA.SA": [100.0] * 15 + [None] + [110.0] * 24,
            "BBB.SA": [50.0 + i for i in range(40)],
        },
        index=dates,
    )
    returns = prices.pct_change(fill_method=None).dropna()
    mu, covariance = estimate_markowitz_inputs(returns)

    assert dates[15] not in returns.index and dates[16] not in returns.index
    assert mu["AAA.SA"] == 0
    assert covariance.loc["AAA.SA", "AAA.SA"] == 0


def test_missing_stress_price_does_not_bridge_gap():
    import pandas as pd

    from utils.portfolio_data import calculate_historical_stress

    dates = pd.bdate_range("2020-01-01", periods=8)
    prices = pd.DataFrame({"AAA.SA": [100, 100, 100, None, 110, 110, 110, 110]}, index=dates)
    result = calculate_historical_stress(
        prices, None, {"AAA.SA": 1.0}, {"Crise": ("2020-01-01", "2020-01-31")}
    )

    assert len(result) == 1
    assert result[0]["Portfólio"] == 0
    assert result[0]["IBOV"] is None
