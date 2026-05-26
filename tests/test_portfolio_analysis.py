import numpy as np
import pandas as pd

from src.portfolio_analysis import (
    annualized_portfolio_metrics,
    compute_log_returns,
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
