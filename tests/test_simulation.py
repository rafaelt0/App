from utils.simulation import simulate_portfolio


def test_annualized_log_return_stats_uses_more_daily_observations_to_reduce_iid_se():
    import numpy as np
    from utils.simulation import annualized_log_return_stats

    returns = np.array([[0.01], [-0.02], [0.03], [0.0]])
    annual_mean, standard_error = annualized_log_return_stats(returns, [1.0])
    daily_log = np.log1p(returns[:, 0])
    assert np.isclose(annual_mean, 252 * daily_log.mean())
    assert np.isclose(standard_error, 252 * daily_log.std(ddof=1) / np.sqrt(4))

    repeated_returns = np.tile(returns, (4, 1))
    _, longer_standard_error = annualized_log_return_stats(repeated_returns, [1.0])
    repeated_log = np.log1p(repeated_returns[:, 0])
    expected = 252 * repeated_log.std(ddof=1) / np.sqrt(len(repeated_log))
    assert np.isclose(longer_standard_error, expected)
    assert longer_standard_error < standard_error


def test_simulation_daily_rebalances_and_does_not_change_global_rng():
    import numpy as np
    original = np.random.get_state()
    paths = simulate_portfolio(
        (np.log(1.1), np.log(0.9)), ((0.0, 0.0), (0.0, 0.0)),
        (0.5, 0.5), 2, 3, 100.0, "2026-01-01"
    )
    after = np.random.get_state()
    assert all((a == b).all() if hasattr(a, "shape") else a == b for a, b in zip(original, after))
    assert paths.shape == (3, 3)
    # Equal-weight daily returns cancel; buy-and-hold would end at R$101.
    assert (paths.iloc[-1] == 100).all()


def test_bootstrap_joint_days_respond_to_changed_history():
    import numpy as np
    from utils.simulation import bootstrap_terminal_values

    # Independently drawing assets would create +/-20% days; joint rows cancel exactly.
    returns = np.array([[0.2, -0.2], [-0.2, 0.2]])
    args = ((0.5, 0.5), 5, 100, 100.0)
    first = bootstrap_terminal_values(returns, *args, chunk_size=7)
    np.testing.assert_allclose(first, 100.0)

    changed = np.array([[0.2, -0.2], [-0.2, 0.0]])
    second = bootstrap_terminal_values(changed, *args, chunk_size=7)
    assert (second < 100).any()
    crash_days = np.array([[-0.3], [0.0]])
    crash_paths = bootstrap_terminal_values(crash_days, (1.0,), 1, 100, 100.0)
    assert set(np.round(crash_paths, 2)) == {70.0, 100.0}
    with np.testing.assert_raises(ValueError):
        bootstrap_terminal_values(np.array([[-1.0, 0.0], [0.0, 0.0]]), *args)


