import numpy as np
import pandas as pd

from src.portfolio_analysis import (
    annualized_portfolio_metrics,
    compute_log_returns,
    efficient_frontier,
    normality_summary,
    optimize_max_sharpe,
    optimize_min_volatility,
    simulate_portfolios,
    validate_weights,
)


def test_compute_log_returns_uses_log_price_ratio():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 121.0], "MSFT": [50.0, 55.0, 60.5]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    returns = compute_log_returns(prices)

    assert returns.shape == (2, 2)
    assert np.isclose(returns.iloc[0]["AAPL"], np.log(110.0 / 100.0))
    assert np.isclose(returns.iloc[1]["MSFT"], np.log(60.5 / 55.0))


def test_validate_weights_rejects_over_cap():
    weights = np.array([0.41, 0.10, 0.10, 0.10, 0.10, 0.07, 0.01, 0.01])

    assert validate_weights(weights, max_weight=0.40) is False


def test_annualized_portfolio_metrics_matches_manual_calculation():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, -0.01],
            "MSFT": [0.00, 0.01, 0.02],
        }
    )
    weights = np.array([0.6, 0.4])

    metrics = annualized_portfolio_metrics(returns, weights, risk_free_rate=0.01)

    portfolio_daily = returns.to_numpy() @ weights
    expected_return = portfolio_daily.mean() * 252
    expected_volatility = portfolio_daily.std(ddof=1) * np.sqrt(252)
    expected_sharpe = (expected_return - 0.01) / expected_volatility

    assert np.isclose(metrics["return"], expected_return)
    assert np.isclose(metrics["volatility"], expected_volatility)
    assert np.isclose(metrics["sharpe"], expected_sharpe)


def test_normality_summary_contains_expected_columns():
    returns = pd.DataFrame(
        {
            "AAPL": np.linspace(-0.02, 0.02, 40),
            "MSFT": np.linspace(0.02, -0.02, 40),
        }
    )

    summary = normality_summary(returns)

    assert list(summary.columns) == [
        "ticker",
        "observations",
        "mean",
        "std",
        "skewness",
        "kurtosis",
        "skew_pvalue",
        "kurtosis_pvalue",
        "normaltest_pvalue",
        "reject_normality_5pct",
    ]
    assert set(summary["ticker"]) == {"AAPL", "MSFT"}


def test_optimizers_respect_weight_cap_and_sum_to_one():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(120, 4)),
        columns=["A", "B", "C", "D"],
    )

    max_sharpe = optimize_max_sharpe(returns, risk_free_rate=0.01, max_weight=0.40)
    min_vol = optimize_min_volatility(returns, risk_free_rate=0.01, max_weight=0.40)

    assert validate_weights(max_sharpe.weights, max_weight=0.40)
    assert validate_weights(min_vol.weights, max_weight=0.40)


def test_simulate_portfolios_returns_requested_count_and_valid_weights():
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(80, 5)),
        columns=["A", "B", "C", "D", "E"],
    )

    simulated = simulate_portfolios(returns, count=25, risk_free_rate=0.01, max_weight=0.40, seed=123)

    assert simulated.shape[0] == 25
    assert {"return", "volatility", "sharpe"}.issubset(simulated.columns)
    for _, row in simulated.iterrows():
        weights = row[["A", "B", "C", "D", "E"]].to_numpy(dtype=float)
        assert validate_weights(weights, max_weight=0.40)


def test_efficient_frontier_returns_monotonic_targets():
    rng = np.random.default_rng(15)
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.010, size=(140, 5)),
        columns=["A", "B", "C", "D", "E"],
    )

    frontier = efficient_frontier(returns, points=8, risk_free_rate=0.01, max_weight=0.40)

    assert frontier.shape[0] >= 3
    assert {"target_return", "volatility", "sharpe"}.issubset(frontier.columns)
    assert frontier["target_return"].is_monotonic_increasing
