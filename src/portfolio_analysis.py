from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class PortfolioResult:
    name: str
    weights: np.ndarray
    annual_return: float
    annual_volatility: float
    sharpe: float


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    return returns


def validate_weights(weights: np.ndarray, max_weight: float = 0.40, tolerance: float = 1e-8) -> bool:
    return (
        abs(float(np.sum(weights)) - 1.0) <= tolerance
        and bool(np.all(weights >= -tolerance))
        and bool(np.all(weights <= max_weight + tolerance))
    )


def annualized_portfolio_metrics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.01,
) -> dict[str, float]:
    daily_portfolio_returns = returns.to_numpy() @ weights
    annual_return = float(np.mean(daily_portfolio_returns) * TRADING_DAYS)
    annual_volatility = float(np.std(daily_portfolio_returns, ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((annual_return - risk_free_rate) / annual_volatility)
    return {"return": annual_return, "volatility": annual_volatility, "sharpe": sharpe}
