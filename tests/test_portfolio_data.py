from utils.portfolio_data import align_weights_to_columns, bound_efficient_return


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
