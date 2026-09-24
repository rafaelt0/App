from utils.portfolio_data import bound_efficient_return


def test_bound_efficient_return_clamps_to_feasible_range():
    assert bound_efficient_return(2.0, 0.0, 1.0) == 0.999999
    assert bound_efficient_return(-2.0, 0.0, 1.0) == 0.000001


def test_bound_efficient_return_returns_none_for_non_finite_values():
    assert bound_efficient_return(float("nan"), 0.0, 1.0) is None
    assert bound_efficient_return(0.5, float("inf"), 1.0) is None


def test_bound_efficient_return_returns_none_for_degenerate_range():
    assert bound_efficient_return(0.5, 1.0, 1.0) is None
