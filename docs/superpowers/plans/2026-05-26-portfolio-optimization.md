# Portfolio Optimization Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python workflow that analyzes an eight-asset U.S. stock/ETF portfolio, generates all charts/tables, summarizes results, and later creates a PPT draft after user approval.

**Architecture:** Keep the analysis pipeline script-based and easy to inspect. Separate reusable finance calculations into `src/portfolio_analysis.py`, command-line execution into `src/run_analysis.py`, tests into `tests/`, and generated artifacts into `outputs/`. PPT generation is isolated in `src/build_presentation.py` and must run only after the user reviews the analysis outputs.

**Tech Stack:** Python, pandas, numpy, scipy, matplotlib, pypdf/python-pptx where needed, yfinance or stooq-backed fallback for market data, pytest.

---

## File Structure

- Create `requirements.txt`: pinned or minimum runtime dependencies.
- Create `src/portfolio_analysis.py`: pure functions for returns, normality tests, portfolio metrics, simulations, optimization, frontier, and chart generation.
- Create `src/run_analysis.py`: executable analysis pipeline that fetches data, calls pure functions, writes outputs, and prints a concise summary.
- Create `src/build_presentation.py`: PPT builder that consumes generated outputs after approval.
- Create `tests/test_portfolio_analysis.py`: unit tests for calculations and constraints.
- Create `outputs/charts/`: generated PNG charts.
- Create `outputs/tables/`: generated CSV tables.
- Create `outputs/reports/`: generated Markdown report.
- Later modify `PPT_TEMPLATE/효율적자본시장구현 TEMPLATE.pptx` only by reading it as a template and writing generated PPT copies under `outputs/ppt/`.

## Task 1: Project Dependencies and Directories

**Files:**
- Create: `requirements.txt`
- Create: `src/.gitkeep`
- Create: `tests/.gitkeep`

- [ ] **Step 1: Add runtime dependencies**

Create `requirements.txt` with:

```text
pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
yfinance>=0.2
python-pptx>=0.6.23
pytest>=8.0
```

- [ ] **Step 2: Create source and test folders**

Run:

```powershell
New-Item -ItemType Directory -Force -Path src,tests | Out-Null
New-Item -ItemType File -Force -Path src/.gitkeep,tests/.gitkeep | Out-Null
```

- [ ] **Step 3: Verify dependency file exists**

Run:

```powershell
Get-Content requirements.txt
```

Expected: the seven dependencies above are printed.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/.gitkeep tests/.gitkeep
git commit -m "chore: add analysis project structure"
```

## Task 2: Core Finance Calculations

**Files:**
- Create: `tests/test_portfolio_analysis.py`
- Create: `src/portfolio_analysis.py`

- [ ] **Step 1: Write failing tests for return and metric calculations**

Create `tests/test_portfolio_analysis.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
```

Expected: FAIL because `src.portfolio_analysis` does not exist yet.

- [ ] **Step 3: Implement core functions**

Create `src/portfolio_analysis.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/portfolio_analysis.py tests/test_portfolio_analysis.py
git commit -m "feat: add portfolio metric calculations"
```

## Task 3: Normality Tests and Portfolio Optimization

**Files:**
- Modify: `tests/test_portfolio_analysis.py`
- Modify: `src/portfolio_analysis.py`

- [ ] **Step 1: Add failing tests for normality and optimization constraints**

Append to `tests/test_portfolio_analysis.py`:

```python
from src.portfolio_analysis import (
    normality_summary,
    optimize_max_sharpe,
    optimize_min_volatility,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
```

Expected: FAIL because new functions are not implemented.

- [ ] **Step 3: Implement normality and optimization functions**

Append to `src/portfolio_analysis.py`:

```python

def normality_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        skew_stat, skew_p = stats.skewtest(series)
        kurt_stat, kurt_p = stats.kurtosistest(series)
        normal_stat, normal_p = stats.normaltest(series)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/portfolio_analysis.py tests/test_portfolio_analysis.py
git commit -m "feat: add normality tests and optimizers"
```

## Task 4: Simulation, Frontier, and Chart Functions

**Files:**
- Modify: `tests/test_portfolio_analysis.py`
- Modify: `src/portfolio_analysis.py`

- [ ] **Step 1: Add failing tests for simulation output**

Append to `tests/test_portfolio_analysis.py`:

```python
from src.portfolio_analysis import simulate_portfolios


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
```

Expected: FAIL because `simulate_portfolios` is not implemented.

- [ ] **Step 3: Implement simulation and charts**

Append to `src/portfolio_analysis.py`:

```python

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
        x = np.linspace(series.min(), series.max(), 200)
        axes[0].plot(x, stats.norm.pdf(x, series.mean(), series.std(ddof=1)), color="#E45756", linewidth=2)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/portfolio_analysis.py tests/test_portfolio_analysis.py
git commit -m "feat: add portfolio simulation and chart helpers"
```

## Task 5: Analysis Pipeline and Report

**Files:**
- Create: `src/run_analysis.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add output policy to `.gitignore`**

Ensure `.gitignore` includes generated outputs but keeps directories trackable through `.gitkeep` if needed:

```text
outputs/
data/cache/
```

- [ ] **Step 2: Create the analysis runner**

Create `src/run_analysis.py` with:

```python
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from portfolio_analysis import (
    normality_summary,
    optimize_max_sharpe,
    optimize_min_volatility,
    save_normality_charts,
    save_simulation_chart,
    simulate_portfolios,
)


TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "GLD", "TLT", "SCHD"]
RISK_FREE_RATE = 0.01
MAX_WEIGHT = 0.40


def download_prices() -> pd.DataFrame:
    raw = yf.download(TICKERS, period="5y", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            prices = raw.xs("Close", axis=1, level=1)
    else:
        prices = raw
    prices = prices[TICKERS].dropna(how="any")
    if prices.empty:
        raise RuntimeError("downloaded price data is empty")
    return prices


def weights_table(result, tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portfolio": result.name,
            "ticker": tickers,
            "weight": result.weights,
        }
    )


def write_report(
    report_path: Path,
    prices: pd.DataFrame,
    normality: pd.DataFrame,
    max_sharpe,
    min_vol,
    tickers: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    max_weights = ", ".join(f"{ticker}: {weight:.1%}" for ticker, weight in zip(tickers, max_sharpe.weights))
    min_weights = ", ".join(f"{ticker}: {weight:.1%}" for ticker, weight in zip(tickers, min_vol.weights))
    text = f"""# Portfolio Optimization Analysis Summary

Generated: {date.today().isoformat()}

## Data

- Assets: {", ".join(tickers)}
- Date range: {prices.index.min().date()} to {prices.index.max().date()}
- Observations: {len(prices)}
- Risk-free rate: {RISK_FREE_RATE:.1%}
- Per-asset weight cap: {MAX_WEIGHT:.0%}

## Normality

Most financial return series are expected to reject normality because daily returns often contain skewness and fat tails. See `outputs/tables/normality_tests.csv` and `outputs/charts/normality_*.png`.

## Maximum Sharpe Portfolio

- Annualized return: {max_sharpe.annual_return:.2%}
- Annualized volatility: {max_sharpe.annual_volatility:.2%}
- Sharpe ratio: {max_sharpe.sharpe:.2f}
- Weights: {max_weights}

## Minimum Volatility Portfolio

- Annualized return: {min_vol.annual_return:.2%}
- Annualized volatility: {min_vol.annual_volatility:.2%}
- Sharpe ratio: {min_vol.sharpe:.2f}
- Weights: {min_weights}

## Generated Charts

- `outputs/charts/portfolio_simulation.png`
- `outputs/charts/normality_<TICKER>.png`
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    charts_dir = root / "outputs" / "charts"
    tables_dir = root / "outputs" / "tables"
    reports_dir = root / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)

    prices = download_prices()
    prices.to_csv(tables_dir / "prices.csv")

    from portfolio_analysis import compute_log_returns

    returns = compute_log_returns(prices)
    returns.to_csv(tables_dir / "log_returns.csv")

    normality = normality_summary(returns)
    normality.to_csv(tables_dir / "normality_tests.csv", index=False)
    save_normality_charts(returns, charts_dir)

    simulated = simulate_portfolios(returns, count=10_000, risk_free_rate=RISK_FREE_RATE, max_weight=MAX_WEIGHT)
    simulated.to_csv(tables_dir / "portfolio_simulation.csv", index=False)
    save_simulation_chart(simulated, charts_dir / "portfolio_simulation.png")

    max_sharpe = optimize_max_sharpe(returns, risk_free_rate=RISK_FREE_RATE, max_weight=MAX_WEIGHT)
    min_vol = optimize_min_volatility(returns, risk_free_rate=RISK_FREE_RATE, max_weight=MAX_WEIGHT)

    portfolios = pd.concat(
        [weights_table(max_sharpe, TICKERS), weights_table(min_vol, TICKERS)],
        ignore_index=True,
    )
    portfolios.to_csv(tables_dir / "optimized_weights.csv", index=False)
    pd.DataFrame(
        [
            {
                "portfolio": max_sharpe.name,
                "annual_return": max_sharpe.annual_return,
                "annual_volatility": max_sharpe.annual_volatility,
                "sharpe": max_sharpe.sharpe,
            },
            {
                "portfolio": min_vol.name,
                "annual_return": min_vol.annual_return,
                "annual_volatility": min_vol.annual_volatility,
                "sharpe": min_vol.sharpe,
            },
        ]
    ).to_csv(tables_dir / "optimized_metrics.csv", index=False)

    write_report(reports_dir / "analysis_summary.md", prices, normality, max_sharpe, min_vol, TICKERS)

    print("Analysis complete.")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Max Sharpe: return={max_sharpe.annual_return:.2%}, vol={max_sharpe.annual_volatility:.2%}, sharpe={max_sharpe.sharpe:.2f}")
    print(f"Min Vol: return={min_vol.annual_return:.2%}, vol={min_vol.annual_volatility:.2%}, sharpe={min_vol.sharpe:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run analysis**

Run:

```powershell
python src/run_analysis.py
```

Expected:

```text
Analysis complete.
Date range: <five-year start> to <latest available trading day>
Max Sharpe: return=<...>, vol=<...>, sharpe=<...>
Min Vol: return=<...>, vol=<...>, sharpe=<...>
```

If `yfinance` fails due to network or dependency issues, stop and report the exact error before changing the data source.

- [ ] **Step 5: Verify generated files**

Run:

```powershell
Get-ChildItem outputs -Recurse -File | Select-Object FullName,Length
```

Expected: CSV files under `outputs/tables`, PNG files under `outputs/charts`, and `outputs/reports/analysis_summary.md`.

- [ ] **Step 6: Commit source code only**

```bash
git add .gitignore src/run_analysis.py src/portfolio_analysis.py tests/test_portfolio_analysis.py requirements.txt
git commit -m "feat: add portfolio analysis pipeline"
```

## Task 6: Efficient Frontier and Capital Market Line

**Files:**
- Modify: `src/portfolio_analysis.py`
- Modify: `src/run_analysis.py`
- Modify: `tests/test_portfolio_analysis.py`

- [ ] **Step 1: Add failing test for frontier output**

Append to `tests/test_portfolio_analysis.py`:

```python
from src.portfolio_analysis import efficient_frontier


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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
```

Expected: FAIL because `efficient_frontier` is not implemented.

- [ ] **Step 3: Implement frontier and CML chart**

Append to `src/portfolio_analysis.py`:

```python

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
            {"type": "eq", "fun": lambda weights, target=target: (returns.mean().to_numpy() * TRADING_DAYS) @ weights - target},
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
    ax.scatter(max_sharpe.annual_volatility, max_sharpe.annual_return, marker="*", s=260, color="#F2CF5B", edgecolor="black", label="Max Sharpe")
    ax.scatter(min_vol.annual_volatility, min_vol.annual_return, marker="D", s=110, color="#54A24B", edgecolor="black", label="Min Volatility")

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
```

- [ ] **Step 4: Wire frontier into runner**

Modify `src/run_analysis.py` imports:

```python
from portfolio_analysis import (
    efficient_frontier,
    normality_summary,
    optimize_max_sharpe,
    optimize_min_volatility,
    save_frontier_chart,
    save_normality_charts,
    save_simulation_chart,
    simulate_portfolios,
)
```

After optimized metrics are computed, add:

```python
    frontier = efficient_frontier(returns, points=40, risk_free_rate=RISK_FREE_RATE, max_weight=MAX_WEIGHT)
    frontier.to_csv(tables_dir / "efficient_frontier.csv", index=False)
    save_frontier_chart(
        simulated,
        frontier,
        max_sharpe,
        min_vol,
        charts_dir / "efficient_frontier_cml.png",
        risk_free_rate=RISK_FREE_RATE,
    )
```

Update the report chart list to include:

```text
- `outputs/charts/efficient_frontier_cml.png`
```

- [ ] **Step 5: Run tests and analysis**

Run:

```powershell
pytest tests/test_portfolio_analysis.py -v
python src/run_analysis.py
```

Expected: tests pass and `outputs/charts/efficient_frontier_cml.png` exists.

- [ ] **Step 6: Commit**

```bash
git add src/portfolio_analysis.py src/run_analysis.py tests/test_portfolio_analysis.py
git commit -m "feat: add efficient frontier and capital market line"
```

## Task 7: Analysis Review Checkpoint

**Files:**
- No source changes required unless analysis exposes a defect.

- [ ] **Step 1: Read summary outputs**

Run:

```powershell
Get-Content outputs/reports/analysis_summary.md -Encoding UTF8
Import-Csv outputs/tables/optimized_metrics.csv | Format-Table
Import-Csv outputs/tables/optimized_weights.csv | Format-Table
Import-Csv outputs/tables/normality_tests.csv | Format-Table ticker,normaltest_pvalue,reject_normality_5pct
```

- [ ] **Step 2: Present results to user**

Report:

- Actual data date range.
- Normality rejection summary.
- Maximum Sharpe portfolio return, volatility, Sharpe, and weights.
- Minimum volatility portfolio return, volatility, Sharpe, and weights.
- Generated chart list.
- Any caveats from the data source or optimizer.

- [ ] **Step 3: Ask whether to continue to PPT**

Ask:

```text
분석 결과는 위와 같습니다. 이 결과를 기준으로 PPT 초안을 만들까요?
```

Stop until the user approves PPT creation.

## Task 8: PPT Draft After User Approval

**Files:**
- Create: `src/build_presentation.py`
- Generated: `outputs/ppt/portfolio_assignment_draft.pptx`

- [ ] **Step 1: Create PPT builder**

Create `src/build_presentation.py` with code that:

- Opens `PPT_TEMPLATE/효율적자본시장구현 TEMPLATE.pptx`.
- Reads `outputs/reports/analysis_summary.md`.
- Reads `outputs/tables/optimized_metrics.csv`.
- Reads `outputs/tables/optimized_weights.csv`.
- Adds slides for normality results, simulation, optimized weights, and efficient frontier.
- Inserts generated chart images from `outputs/charts/`.
- Saves `outputs/ppt/portfolio_assignment_draft.pptx`.

Use `python-pptx` and keep all generated work under `outputs/ppt/`.

- [ ] **Step 2: Run PPT builder**

Run:

```powershell
python src/build_presentation.py
```

Expected: `outputs/ppt/portfolio_assignment_draft.pptx` exists.

- [ ] **Step 3: Inspect generated PPT file size**

Run:

```powershell
Get-Item outputs/ppt/portfolio_assignment_draft.pptx | Select-Object FullName,Length
```

Expected: file exists and length is greater than the original template because charts were inserted.

- [ ] **Step 4: Commit PPT builder source**

```bash
git add src/build_presentation.py
git commit -m "feat: add presentation draft generator"
```

- [ ] **Step 5: Ask whether to export PDF**

Ask:

```text
PPT 초안이 생성되었습니다. 확인 후 PDF 변환까지 진행할까요?
```

Stop until the user approves PDF export.

## Task 9: Optional PDF Export

**Files:**
- Generated: `outputs/ppt/portfolio_assignment_draft.pdf`

- [ ] **Step 1: Check for available converter**

Run:

```powershell
Get-Command soffice,powerpnt -ErrorAction SilentlyContinue | Select-Object Name,Source
```

Expected: at least one converter is available. If none is available, report that PDF export cannot be completed locally.

- [ ] **Step 2: Export PDF**

If LibreOffice is available, run:

```powershell
soffice --headless --convert-to pdf --outdir outputs/ppt outputs/ppt/portfolio_assignment_draft.pptx
```

If PowerPoint automation is required, use a narrow PowerShell script after explicit approval.

- [ ] **Step 3: Verify PDF exists**

Run:

```powershell
Get-Item outputs/ppt/portfolio_assignment_draft.pdf | Select-Object FullName,Length
```

Expected: PDF exists and length is greater than zero.

## Self-Review

- Spec coverage: The plan covers asset universe, recent five-year period, 0-40% long-only constraints, 1% risk-free rate, normality tests, simulation, max Sharpe optimization, min volatility optimization, efficient frontier, capital market line, Markdown summary, user checkpoint before PPT, and optional PDF export after PPT approval.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps are intentionally left in the executable tasks.
- Type consistency: Core functions use pandas DataFrames for prices/returns, numpy arrays for weights, and `PortfolioResult` for optimizer outputs. Later tasks reference the same function and field names.
- Scope check: PPT generation and PDF export are isolated as later gated tasks, matching the user's instruction to decide on those after analysis review.

