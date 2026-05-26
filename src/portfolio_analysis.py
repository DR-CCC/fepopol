from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


def random_capped_weights(asset_count: int, max_weight: float, rng: np.random.Generator) -> np.ndarray:
    for _ in range(10_000):
        weights = rng.dirichlet(np.ones(asset_count))
        if np.all(weights <= max_weight):
            return weights
    raise RuntimeError("could not generate capped random weights")


def simulate_portfolios(
    returns: pd.DataFrame,
    count: int = 10_000,
    risk_free_rate: float = 0.01,
    max_weight: float = 0.40,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(count):
        weights = random_capped_weights(len(returns.columns), max_weight, rng)
        metrics = annualized_portfolio_metrics(returns, weights, risk_free_rate)
        row = {
            "return": metrics["return"],
            "volatility": metrics["volatility"],
            "sharpe": metrics["sharpe"],
        }
        row.update({ticker: float(weight) for ticker, weight in zip(returns.columns, weights)})
        rows.append(row)
    return pd.DataFrame(rows)


def save_normality_charts(returns: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(series, bins=50, density=True, alpha=0.65, color="#4C78A8")
        x_values = np.linspace(series.min(), series.max(), 200)
        axes[0].plot(
            x_values,
            stats.norm.pdf(x_values, series.mean(), series.std(ddof=1)),
            color="#E45756",
            linewidth=2,
        )
        axes[0].set_title(f"{ticker} Log Returns")
        axes[0].set_xlabel("Daily log return")
        axes[0].set_ylabel("Density")
        stats.probplot(series, dist="norm", plot=axes[1])
        axes[1].set_title(f"{ticker} Q-Q Plot")
        fig.tight_layout()
        fig.savefig(output_dir / f"normality_{ticker}.png", dpi=180)
        plt.close(fig)


def save_simulation_chart(simulated: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        simulated["volatility"],
        simulated["return"],
        c=simulated["sharpe"],
        cmap="viridis",
        s=14,
        alpha=0.75,
    )
    ax.set_title("Portfolio Simulation")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def efficient_frontier(
    returns: pd.DataFrame,
    points: int = 40,
    risk_free_rate: float = 0.01,
    max_weight: float = 0.40,
) -> pd.DataFrame:
    asset_count = len(returns.columns)
    asset_annual_returns = returns.mean().to_numpy() * TRADING_DAYS
    min_target = float(np.min(asset_annual_returns))
    max_target = float(np.max(asset_annual_returns))
    targets = np.linspace(min_target, max_target, points)
    rows: list[dict[str, float]] = []

    for target in targets:
        constraints = (
            _sum_to_one_constraint(),
            {
                "type": "eq",
                "fun": lambda weights, target=target: (returns.mean().to_numpy() * TRADING_DAYS) @ weights
                - target,
            },
        )

        result = optimize.minimize(
            lambda weights: annualized_portfolio_metrics(returns, weights, risk_free_rate)["volatility"],
            _initial_weights(asset_count),
            method="SLSQP",
            bounds=_bounds(asset_count, max_weight),
            constraints=constraints,
        )
        if result.success:
            weights = np.asarray(result.x)
            metrics = annualized_portfolio_metrics(returns, weights, risk_free_rate)
            row = {
                "target_return": target,
                "return": metrics["return"],
                "volatility": metrics["volatility"],
                "sharpe": metrics["sharpe"],
            }
            row.update({ticker: float(weight) for ticker, weight in zip(returns.columns, weights)})
            rows.append(row)

    if not rows:
        raise RuntimeError("efficient frontier optimization failed for all target returns")
    return pd.DataFrame(rows).sort_values("target_return").reset_index(drop=True)


def save_frontier_chart(
    simulated: pd.DataFrame,
    frontier: pd.DataFrame,
    max_sharpe: PortfolioResult,
    min_vol: PortfolioResult,
    output_path: Path,
    risk_free_rate: float = 0.01,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        simulated["volatility"],
        simulated["return"],
        c=simulated["sharpe"],
        cmap="viridis",
        s=10,
        alpha=0.35,
        label="Simulated portfolios",
    )
    ax.plot(frontier["volatility"], frontier["return"], color="#E45756", linewidth=2.5, label="Efficient frontier")
    ax.scatter(
        max_sharpe.annual_volatility,
        max_sharpe.annual_return,
        marker="*",
        s=260,
        color="#F2CF5B",
        edgecolor="black",
        label="Max Sharpe",
    )
    ax.scatter(
        min_vol.annual_volatility,
        min_vol.annual_return,
        marker="D",
        s=110,
        color="#54A24B",
        edgecolor="black",
        label="Min Volatility",
    )

    cml_x = np.linspace(0, max(frontier["volatility"].max(), max_sharpe.annual_volatility) * 1.05, 100)
    cml_slope = (max_sharpe.annual_return - risk_free_rate) / max_sharpe.annual_volatility
    cml_y = risk_free_rate + cml_slope * cml_x
    ax.plot(cml_x, cml_y, color="#B279A2", linestyle="--", linewidth=2, label="Capital market line")

    ax.set_title("Efficient Frontier and Capital Market Line")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_weights_comparison_chart(
    tickers: list[str],
    max_sharpe: PortfolioResult,
    min_vol: PortfolioResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_positions = np.arange(len(tickers))
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x_positions - bar_width / 2,
        max_sharpe.weights,
        width=bar_width,
        label="Maximum Sharpe",
        color="#4C78A8",
    )
    ax.bar(
        x_positions + bar_width / 2,
        min_vol.weights,
        width=bar_width,
        label="Minimum Volatility",
        color="#54A24B",
    )
    ax.axhline(0.40, color="#E45756", linestyle="--", linewidth=1.4, label="Weight cap 40%")
    ax.set_title("Optimized Portfolio Weights")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Portfolio weight")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tickers)
    ax.set_ylim(0, 0.45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_normality_pvalue_chart(normality: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = normality.sort_values("normaltest_pvalue", ascending=False).copy()
    plot_data["minus_log10_pvalue"] = -np.log10(plot_data["normaltest_pvalue"].clip(lower=1e-300))
    rejection_line = -np.log10(0.05)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(plot_data["ticker"], plot_data["minus_log10_pvalue"], color="#F58518")
    ax.axhline(rejection_line, color="#E45756", linestyle="--", linewidth=1.8, label="5% rejection threshold")
    ax.set_title("Normality Test Summary")
    ax.set_xlabel("Asset")
    ax.set_ylabel("-log10(p-value)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_asset_risk_return_chart(returns: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        annual_return = float(series.mean() * TRADING_DAYS)
        annual_volatility = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS))
        rows.append({"ticker": ticker, "annual_return": annual_return, "annual_volatility": annual_volatility})

    asset_metrics = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(asset_metrics["annual_volatility"], asset_metrics["annual_return"], s=95, color="#4C78A8")
    for _, row in asset_metrics.iterrows():
        ax.annotate(
            row["ticker"],
            (row["annual_volatility"], row["annual_return"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Individual Asset Risk-Return Profile")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return asset_metrics
