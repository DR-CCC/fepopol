from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


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


def normality_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        _, skew_p = stats.skewtest(series)
        _, kurt_p = stats.kurtosistest(series)
        _, normal_p = stats.normaltest(series)
        rows.append(
            {
                "ticker": ticker,
                "observations": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "skewness": float(stats.skew(series)),
                "kurtosis": float(stats.kurtosis(series)),
                "skew_pvalue": float(skew_p),
                "kurtosis_pvalue": float(kurt_p),
                "normaltest_pvalue": float(normal_p),
                "reject_normality_5pct": bool(normal_p < 0.05),
            }
        )
    return pd.DataFrame(rows)


def _initial_weights(asset_count: int) -> np.ndarray:
    return np.repeat(1.0 / asset_count, asset_count)


def _bounds(asset_count: int, max_weight: float) -> tuple[tuple[float, float], ...]:
    return tuple((0.0, max_weight) for _ in range(asset_count))


def _sum_to_one_constraint() -> dict[str, object]:
    return {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}


def optimize_max_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.01,
    max_weight: float = 0.40,
) -> PortfolioResult:
    asset_count = len(returns.columns)

    def objective(weights: np.ndarray) -> float:
        return -annualized_portfolio_metrics(returns, weights, risk_free_rate)["sharpe"]

    result = optimize.minimize(
        objective,
        _initial_weights(asset_count),
        method="SLSQP",
        bounds=_bounds(asset_count, max_weight),
        constraints=(_sum_to_one_constraint(),),
    )
    if not result.success:
        raise RuntimeError(f"max Sharpe optimization failed: {result.message}")
    weights = np.asarray(result.x)
    metrics = annualized_portfolio_metrics(returns, weights, risk_free_rate)
    return PortfolioResult("Maximum Sharpe", weights, metrics["return"], metrics["volatility"], metrics["sharpe"])


def optimize_min_volatility(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.01,
    max_weight: float = 0.40,
) -> PortfolioResult:
    asset_count = len(returns.columns)

    def objective(weights: np.ndarray) -> float:
        return annualized_portfolio_metrics(returns, weights, risk_free_rate)["volatility"]

    result = optimize.minimize(
        objective,
        _initial_weights(asset_count),
        method="SLSQP",
        bounds=_bounds(asset_count, max_weight),
        constraints=(_sum_to_one_constraint(),),
    )
    if not result.success:
        raise RuntimeError(f"minimum volatility optimization failed: {result.message}")
    weights = np.asarray(result.x)
    metrics = annualized_portfolio_metrics(returns, weights, risk_free_rate)
    return PortfolioResult("Minimum Volatility", weights, metrics["return"], metrics["volatility"], metrics["sharpe"])
