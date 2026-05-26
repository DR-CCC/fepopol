# Portfolio Optimization Assignment Design

## Context

The project folder currently contains the assignment notice, a cleaned Markdown summary of the PDF, and a PowerPoint template:

- `과제공지/포트폴리오과제.pdf`
- `과제공지/포트폴리오과제.markdown`
- `과제공지/런어스공지.md`
- `PPT_TEMPLATE/효율적자본시장구현 TEMPLATE.pptx`

There is no existing analysis code, dataset, generated chart set, or completed presentation. The folder is not a git repository, so this design document cannot be committed unless a repository is initialized later.

## Goal

Build a finance assignment workflow that analyzes a mixed U.S. stock and ETF portfolio, finds optimized portfolio weights, and prepares materials suitable for a presentation.

The work should proceed in stages. After each major analytical stage, results should be explained to the user before continuing to later deliverables such as PPT creation or PDF conversion.

## Confirmed Scope

### Asset Universe

Use eight U.S.-listed assets:

| Ticker | Role |
|---|---|
| `AAPL` | Large-cap growth stock |
| `MSFT` | Stable large-cap growth stock |
| `NVDA` | AI and semiconductor growth stock |
| `SPY` | Broad U.S. market ETF |
| `QQQ` | Nasdaq/growth index ETF |
| `GLD` | Gold ETF and alternative asset |
| `TLT` | Long-term Treasury bond ETF |
| `SCHD` | Dividend/value ETF |

### Analysis Period

Use the most recent five years of available daily price data at the time the analysis is run.

### Portfolio Constraints

- Long-only portfolio.
- Total portfolio weight must equal 100%.
- Each asset weight must be between 0% and 40%.
- Risk-free rate for the capital market line: 1% annualized.

### Primary Deliverables

1. Analysis code.
2. Downloaded or cached price data if needed.
3. Normality test outputs and charts.
4. Portfolio simulation and optimization outputs.
5. Efficient frontier and capital market line charts.
6. Result summary in Markdown.
7. PPT draft based on the provided template, only after user approval of the analytical results.

### Later Optional Deliverables

- PDF export of the PPT, only after user approval of the PPT draft.

## Non-Goals

- Do not build the PPT before the user reviews the numerical and graphical analysis results.
- Do not convert to PDF until the user confirms the PPT draft.
- Do not add unrelated assets unless the user explicitly changes the asset universe.
- Do not use short selling or leveraged portfolio assumptions.
- Do not treat the selected eight assets as an actual investment recommendation; frame results as assignment analysis.

## Recommended Workflow

### Stage 1: Data and Setup

Create a clear project structure for reproducible outputs:

- `src/` for Python analysis scripts.
- `data/` for downloaded or cached price data.
- `outputs/charts/` for generated charts.
- `outputs/tables/` for CSV result tables.
- `outputs/reports/` for Markdown summaries.
- `outputs/ppt/` for generated presentation files.

Fetch adjusted close price data for the eight tickers over the most recent five years. Use adjusted prices when possible because dividends and splits matter for return calculations.

### Stage 2: Return Construction and Normality Analysis

For each asset:

- Compute daily log returns.
- Compare the return distribution against a normal distribution.
- Generate histogram plus normal density plot.
- Generate Q-Q plot.
- Run skewness test.
- Run kurtosis test.
- Run normality test.
- Summarize P-values and normality rejection results.

Expected interpretation: most financial return series are likely to reject normality, especially due to fat tails and skewness.

### Stage 3: Portfolio Simulation

Run a Monte Carlo simulation of random long-only portfolios under the 0% to 40% per-asset cap.

For each simulated portfolio:

- Compute annualized expected return.
- Compute annualized volatility.
- Compute Sharpe ratio.
- Store asset weights.

Generate a scatter plot of volatility versus return, colored by Sharpe ratio.

### Stage 4: Portfolio Optimization

Use numerical optimization to compute:

- Maximum Sharpe ratio portfolio.
- Minimum volatility portfolio.
- Efficient frontier portfolios across target returns.

Optimization must respect:

- Sum of weights equals 1.
- Each individual weight is between 0 and 0.4.
- No short positions.

Output tables should include:

- Asset weights.
- Annualized return.
- Annualized volatility.
- Sharpe ratio.

### Stage 5: Efficient Frontier and Capital Market Line

Plot:

- Simulated portfolios.
- Efficient frontier.
- Maximum Sharpe portfolio.
- Minimum volatility portfolio.
- Capital market line using a 1% risk-free rate.

If needed for a smooth capital market line demonstration, use interpolation on the efficient frontier. The design should keep the implementation explainable for the assignment rather than overcomplicating the math.

### Stage 6: User Review Checkpoint

Before generating PPT slides, present:

- Selected assets and data period.
- Normality test summary.
- Maximum Sharpe portfolio result.
- Minimum volatility portfolio result.
- Efficient frontier and capital market line chart.
- Short interpretation suitable for presentation.

Ask the user whether to continue to PPT creation.

### Stage 7: PPT Draft

Only after user approval, generate a PPT draft using the existing template as the base.

The template has seven slides, but the final deck may be longer if the analysis needs more space. A likely deck structure:

1. Title.
2. Contents.
3. Theory: normality, mean-variance selection, efficient frontier, capital market line.
4. Data selection: eight assets, roles, period, source.
5. Method: log returns, normality tests, simulation, optimization constraints.
6. Normality analysis results.
7. Portfolio simulation results.
8. Optimization results: maximum Sharpe and minimum volatility.
9. Efficient frontier and capital market line.
10. Final interpretation and conclusion.

### Stage 8: PDF Export Checkpoint

After PPT draft review, ask the user whether to export to PDF. PDF export depends on local PowerPoint or LibreOffice availability.

## Data and Calculation Assumptions

Use daily adjusted close prices. Compute log returns as:

```text
log_return_t = ln(price_t / price_{t-1})
```

Annualize daily metrics with:

```text
annualized_return = mean_daily_return * 252
annualized_volatility = std_daily_return * sqrt(252)
```

Sharpe ratio:

```text
sharpe = (annualized_return - risk_free_rate) / annualized_volatility
```

Use `risk_free_rate = 0.01`.

## Error Handling

If a ticker fails to download:

- Report the failed ticker and cause.
- Retry once.
- If still unavailable, ask the user before substituting an asset.

If missing prices occur:

- Align all assets by common trading dates.
- Drop rows where any selected asset has missing adjusted close data.
- Report the final date range and number of observations.

If optimization fails:

- Report the failed objective.
- Check constraint feasibility.
- Retry with a different initial weight vector.
- If still failing, stop before PPT generation and explain the issue.

## Testing and Verification

Before presenting analysis results:

- Confirm all eight tickers are present.
- Confirm the final data period is approximately five years.
- Confirm portfolio weights sum to 1 within numerical tolerance.
- Confirm no optimized weight is below 0 or above 0.4.
- Confirm all required charts were generated.
- Confirm result tables contain maximum Sharpe and minimum volatility portfolios.

Before PPT generation:

- Confirm analytical results have been shown to and approved by the user.

Before PDF export:

- Confirm PPT draft has been shown to and approved by the user.

## Open Decisions

No open decisions remain for the analysis design. Later continuation gates remain intentional:

- Whether to create the PPT after reviewing analysis results.
- Whether to export the PPT to PDF after reviewing the PPT draft.

