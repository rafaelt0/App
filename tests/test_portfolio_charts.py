import numpy as np

from utils import portfolio_charts


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
