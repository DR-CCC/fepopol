from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from portfolio_analysis import (
    compute_log_returns,
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
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
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

    write_report(reports_dir / "analysis_summary.md", prices, max_sharpe, min_vol, TICKERS)

    print("Analysis complete.")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(
        f"Max Sharpe: return={max_sharpe.annual_return:.2%}, "
        f"vol={max_sharpe.annual_volatility:.2%}, sharpe={max_sharpe.sharpe:.2f}"
    )
    print(
        f"Min Vol: return={min_vol.annual_return:.2%}, "
        f"vol={min_vol.annual_volatility:.2%}, sharpe={min_vol.sharpe:.2f}"
    )


if __name__ == "__main__":
    main()
