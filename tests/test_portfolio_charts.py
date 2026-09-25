import numpy as np

from utils import portfolio_charts


def test_random_portfolios_are_uniform_on_the_simplex(monkeypatch):
    class FakeEfficientFrontier:
        def __init__(self, mu, covariance):
            self.weights = {str(i): 1 / len(mu) for i in range(len(mu))}

        def min_volatility(self):
            return self.weights

        def efficient_return(self, target):
            return None

        def clean_weights(self):
            return self.weights

    monkeypatch.setattr(portfolio_charts, "EfficientFrontier", FakeEfficientFrontier)
    portfolio_charts._compute_frontier_data.clear()
    results = portfolio_charts._compute_frontier_data(
        (0.0, 1.0, 0.0),
        ((0.04, 0.0, 0.0), (0.0, 0.04, 0.0), (0.0, 0.0, 0.04)),
        (1 / 3, 1 / 3, 1 / 3),
        0.02,
        20_000,
    )[0]

    target_asset_weights = results[1]
    assert np.all((target_asset_weights >= 0) & (target_asset_weights <= 1))
    assert np.isclose(target_asset_weights.mean(), 1 / 3, atol=0.01)
    assert 0.08 < np.mean(target_asset_weights < 0.05) < 0.12


def test_frontier_labels_selected_l2_allocation_without_calling_it_max_sharpe(monkeypatch):
    monkeypatch.setattr(
        portfolio_charts,
        "_compute_frontier_data",
        lambda *args: (
            np.array([[0.2], [0.1], [0.25]]),
            np.array([0.2]),
            np.array([0.1]),
            0.1,
            0.2,
            0.25,
            0.08,
            0.1,
        ),
    )

    fig = portfolio_charts.plot_efficient_frontier_and_random_portfolios(
        np.array([0.1]), np.array([[0.04]]), {"AAA.SA": 1.0}, 0.05, "Retorno-alvo com L2"
    )

    assert any(trace.name == "Retorno-alvo com L2" for trace in fig.data)
    assert "Max Sharpe" not in str(fig.to_plotly_json())
