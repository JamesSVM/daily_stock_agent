# V1.2 Portfolio Backtest

V1.2 converts the trade-level mean-reversion signals into a capital-constrained portfolio simulation without changing the signal rules.

## Baseline assumptions

- Initial capital: NT$1,000,000
- 10% of initial capital targeted per position
- Maximum 10 concurrent positions
- Same-day signals are ranked by score when slots are limited
- Commission: configurable, default 0.1425% per side
- Stock transaction tax: default 0.3% on sales
- Slippage: default 0.05% per side as a research assumption
- Daily mark-to-market uses `daily_price.close` from the local SQLite database

These are research defaults, not a statement of the user's actual broker pricing or fill quality.
