# V1.2 Portfolio Backtest

## Purpose

V1.2 converts the trade-level mean-reversion results into a capital-constrained portfolio simulation. The signal rules remain unchanged.

## Baseline assumptions

- Initial capital: NT$1,000,000
- Allocation per position: 10% of initial capital, capped by available cash
- Maximum concurrent positions: 10
- Same-day signals are ranked by score; higher scores receive priority when slots are limited.
- Commission is configurable; the default is 0.1425% on both buy and sell.
- Stock transaction tax defaults to 0.3% on sales.
- Slippage defaults to 0.05% on each side as a research assumption.
- Positions are marked to market daily using SQLite `daily_price` closes.

These are research assumptions, not a statement of the user's actual brokerage fee schedule.
