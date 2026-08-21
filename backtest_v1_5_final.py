from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# V1.5 FINAL CANDIDATE — FROZEN CONFIGURATION
# ============================================================

DB_PATH = "data/database.db"
TRADES_FILE = "trades_relative_strength_v1_3.csv"

INITIAL_CAPITAL = 1_000_000.0

MAX_POSITIONS = 3

# Entry
RS_THRESHOLD = 0.10
PULLBACK_MIN = -0.07
PULLBACK_MAX = -0.02

# Original V1.3 event-driven exits are preserved from CSV.
# Portfolio-level regime risk exit:
REGIME_POLICY = "bull_to_neutral"
REGIME_CONFIRMATION_DAYS = 1

# Transaction cost
ROUND_TRIP_COST = 0.005
BUY_COST = ROUND_TRIP_COST / 2.0
SELL_COST = ROUND_TRIP_COST / 2.0


# ============================================================
# Data classes
# ============================================================

@dataclass
class Position:
    stock_id: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    cost_basis: float
    rs20: float
    score: float
    alpha: float
    original_exit_date: pd.Timestamp
    original_exit_price: float
    original_exit_reason: str


# ============================================================
# Universe
# ============================================================

def get_universe_stock_ids(
    conn: sqlite3.Connection,
) -> list[str]:

    rows = conn.execute(
        """
        SELECT DISTINCT stock_id
        FROM daily_price
        ORDER BY stock_id
        """
    ).fetchall()

    return [
        str(row[0])
        for row in rows
    ]


# ============================================================
# Candidates
# ============================================================

def load_candidates() -> pd.DataFrame:

    df = pd.read_csv(
        TRADES_FILE
    )

    for column in [
        "signal_date",
        "entry_date",
        "exit_date",
    ]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    for column in [
        "entry",
        "exit",
        "return",
        "alpha",
        "rs20",
        "drawdown_20d",
        "score",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    required = [
        "signal_date",
        "entry_date",
        "exit_date",
        "entry",
        "exit",
        "return",
        "alpha",
        "rs20",
        "drawdown_20d",
        "score",
    ]

    df = df.dropna(
        subset=required
    ).copy()

    df["stock_id"] = (
        df["stock_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Frozen V1.5 candidate definition
    # --------------------------------------------------------

    df = df[
        (df["rs20"] > RS_THRESHOLD)
        & (
            df["drawdown_20d"]
            >= PULLBACK_MIN
        )
        & (
            df["drawdown_20d"]
            <= PULLBACK_MAX
        )
    ].copy()

    df = df.sort_values(
        [
            "entry_date",
            "rs20",
            "score",
            "stock_id",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )

    return df


# ============================================================
# Prices
# ============================================================

def load_prices(
    conn: sqlite3.Connection,
    stock_ids: list[str],
) -> pd.DataFrame:

    if not stock_ids:
        return pd.DataFrame()

    placeholders = ",".join(
        ["?"] * len(stock_ids)
    )

    query = f"""
        SELECT
            stock_id,
            date,
            open,
            high,
            low,
            close,
            volume
        FROM daily_price
        WHERE stock_id IN ({placeholders})
        ORDER BY date, stock_id
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=stock_ids,
    )

    if df.empty:
        return df

    df["stock_id"] = (
        df["stock_id"]
        .astype(str)
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "stock_id",
            "date",
            "open",
            "close",
        ]
    )

    df = df[
        np.isfinite(df["open"])
        & np.isfinite(df["close"])
        & (df["open"] > 0)
        & (df["close"] > 0)
    ].copy()

    return df


# ============================================================
# Market benchmark + regime
#
# EXACT frozen definition used throughout V1.4-D/F/G.
# ============================================================

def build_market_regime(
    prices: pd.DataFrame,
) -> pd.DataFrame:

    matrix = (
        prices
        .pivot(
            index="date",
            columns="stock_id",
            values="close",
        )
        .sort_index()
    )

    market_return = (
        matrix
        .pct_change()
        .mean(
            axis=1,
            skipna=True,
        )
        .fillna(0.0)
    )

    market = pd.DataFrame(
        index=market_return.index
    )

    market["return"] = market_return

    market["equity"] = (
        1.0
        + market["return"]
    ).cumprod()

    market["ma20"] = (
        market["equity"]
        .rolling(20)
        .mean()
    )

    market["ma60"] = (
        market["equity"]
        .rolling(60)
        .mean()
    )

    market["regime"] = "neutral"

    bull = (
        (market["equity"] > market["ma60"])
        & (
            market["ma20"]
            > market["ma60"]
        )
    )

    bear = (
        (market["equity"] < market["ma60"])
        & (
            market["ma20"]
            < market["ma60"]
        )
    )

    market.loc[
        bull,
        "regime",
    ] = "bull"

    market.loc[
        bear,
        "regime",
    ] = "bear"

    return market


# ============================================================
# Regime confirmation
#
# Frozen rule:
# Bull -> Neutral
# 1 confirmation day
# exit at next trading day's Open
# ============================================================

def build_regime_exit_schedule(
    market: pd.DataFrame,
) -> tuple[
    dict[pd.Timestamp, str],
    list[dict],
]:

    dates = list(
        market.index
    )

    regimes = (
        market["regime"]
        .astype(str)
    )

    schedule: dict[
        pd.Timestamp,
        str,
    ] = {}

    audit_rows = []

    for i in range(
        1,
        len(dates),
    ):

        previous_date = dates[i - 1]
        transition_date = dates[i]

        previous_regime = regimes.iloc[
            i - 1
        ]

        current_regime = regimes.iloc[
            i
        ]

        # Frozen rule: Bull -> Neutral
        if not (
            previous_regime == "bull"
            and current_regime == "neutral"
        ):
            continue

        # Confirmation = 1 day means the first
        # neutral observation is enough.
        confirmation_end_index = i

        if (
            confirmation_end_index
            >= len(dates)
        ):
            continue

        confirmed_regime = regimes.iloc[
            confirmation_end_index
        ]

        confirmed = (
            confirmed_regime
            == "neutral"
        )

        if not confirmed:
            continue

        # Regime becomes known at transition_date close.
        # Therefore the actual exit is next trading day.
        exit_index = (
            confirmation_end_index + 1
        )

        if exit_index >= len(dates):
            continue

        exit_date = dates[
            exit_index
        ]

        schedule[
            exit_date
        ] = (
            "bull_to_neutral_confirm_1"
        )

        audit_rows.append(
            {
                "transition_date": transition_date,
                "confirmed_date": transition_date,
                "exit_date": exit_date,
                "previous_regime": previous_regime,
                "transition_regime": current_regime,
                "confirmation_regime": confirmed_regime,
                "lookahead_safe": (
                    exit_date
                    > transition_date
                ),
            }
        )

    return schedule, audit_rows


# ============================================================
# Portfolio backtest
# ============================================================

def run_backtest(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if candidates.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    price_lookup = {}

    for stock_id, group in prices.groupby(
        "stock_id"
    ):

        price_lookup[
            str(stock_id)
        ] = (
            group
            .set_index("date")
            .sort_index()
        )

    entry_groups = {
        date: group
        for date, group in candidates.groupby(
            "entry_date"
        )
    }

    # Original V1.3 exit events.
    exit_groups = {
        date: group
        for date, group in candidates.groupby(
            "exit_date"
        )
    }

    start_date = candidates[
        "entry_date"
    ].min()

    end_date = candidates[
        "exit_date"
    ].max()

    trading_dates = market.index[
        (market.index >= start_date)
        & (market.index <= end_date)
    ]

    regime_exit_schedule, _ = (
        build_regime_exit_schedule(
            market
        )
    )

    cash = INITIAL_CAPITAL

    positions: dict[
        str,
        Position,
    ] = {}

    closed_trades = []

    equity_rows = []

    turnover_cash = 0.0

    for current_date in trading_dates:

        # ====================================================
        # 1. Portfolio-level regime exit
        # ====================================================

        regime_reason = (
            regime_exit_schedule.get(
                current_date
            )
        )

        if regime_reason is not None:

            for stock_id in list(
                positions.keys()
            ):

                position = positions[
                    stock_id
                ]

                table = price_lookup.get(
                    stock_id
                )

                if table is None:
                    continue

                if current_date not in table.index:
                    continue

                exit_price = float(
                    table.loc[
                        current_date,
                        "open",
                    ]
                )

                if (
                    not np.isfinite(
                        exit_price
                    )
                    or exit_price <= 0
                ):
                    continue

                gross_proceeds = (
                    position.shares
                    * exit_price
                )

                sell_fee = (
                    gross_proceeds
                    * SELL_COST
                )

                net_proceeds = (
                    gross_proceeds
                    - sell_fee
                )

                cash += net_proceeds

                turnover_cash += (
                    gross_proceeds
                )

                net_return = (
                    net_proceeds
                    / position.cost_basis
                    - 1.0
                )

                closed_trades.append(
                    {
                        "stock_id": stock_id,
                        "entry_date": position.entry_date,
                        "exit_date": current_date,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "gross_return": (
                            exit_price
                            / position.entry_price
                            - 1.0
                        ),
                        "net_return": net_return,
                        "alpha": position.alpha,
                        "score": position.score,
                        "rs20": position.rs20,
                        "exit_reason": "regime_exit",
                        "regime_transition": regime_reason,
                    }
                )

                del positions[
                    stock_id
                ]

        # ====================================================
        # 2. Original V1.3 exit
        # ====================================================

        exits = exit_groups.get(
            current_date,
            pd.DataFrame(),
        )

        if not exits.empty:

            for _, trade in exits.iterrows():

                stock_id = str(
                    trade["stock_id"]
                )

                position = positions.get(
                    stock_id
                )

                if position is None:
                    continue

                exit_price = float(
                    trade["exit"]
                )

                if (
                    not np.isfinite(
                        exit_price
                    )
                    or exit_price <= 0
                ):
                    continue

                gross_proceeds = (
                    position.shares
                    * exit_price
                )

                sell_fee = (
                    gross_proceeds
                    * SELL_COST
                )

                net_proceeds = (
                    gross_proceeds
                    - sell_fee
                )

                cash += net_proceeds

                turnover_cash += (
                    gross_proceeds
                )

                net_return = (
                    net_proceeds
                    / position.cost_basis
                    - 1.0
                )

                closed_trades.append(
                    {
                        "stock_id": stock_id,
                        "entry_date": position.entry_date,
                        "exit_date": current_date,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "gross_return": (
                            exit_price
                            / position.entry_price
                            - 1.0
                        ),
                        "net_return": net_return,
                        "alpha": position.alpha,
                        "score": position.score,
                        "rs20": position.rs20,
                        "exit_reason": str(
                            trade.get(
                                "exit_reason",
                                "normal_exit",
                            )
                        ),
                        "regime_transition": "",
                    }
                )

                del positions[
                    stock_id
                ]

        # ====================================================
        # 3. New entries
        # ====================================================

        entries = entry_groups.get(
            current_date,
            pd.DataFrame(),
        )

        if not entries.empty:

            slots = (
                MAX_POSITIONS
                - len(positions)
            )

            if slots > 0:

                selected = (
                    entries
                    .sort_values(
                        [
                            "rs20",
                            "score",
                            "stock_id",
                        ],
                        ascending=[
                            False,
                            False,
                            True,
                        ],
                    )
                    .head(slots)
                )

                for _, trade in selected.iterrows():

                    stock_id = str(
                        trade["stock_id"]
                    )

                    if stock_id in positions:
                        continue

                    if cash <= 0:
                        break

                    entry_price = float(
                        trade["entry"]
                    )

                    if (
                        not np.isfinite(
                            entry_price
                        )
                        or entry_price <= 0
                    ):
                        continue

                    target_capital = (
                        INITIAL_CAPITAL
                        / MAX_POSITIONS
                    )

                    gross_allocation = min(
                        target_capital,
                        cash
                        / (
                            1.0
                            + BUY_COST
                        ),
                    )

                    if gross_allocation <= 0:
                        continue

                    buy_fee = (
                        gross_allocation
                        * BUY_COST
                    )

                    total_cost = (
                        gross_allocation
                        + buy_fee
                    )

                    shares = (
                        gross_allocation
                        / entry_price
                    )

                    cash -= total_cost

                    turnover_cash += (
                        gross_allocation
                    )

                    positions[
                        stock_id
                    ] = Position(
                        stock_id=stock_id,
                        entry_date=current_date,
                        entry_price=entry_price,
                        shares=shares,
                        cost_basis=total_cost,
                        rs20=float(
                            trade["rs20"]
                        ),
                        score=float(
                            trade["score"]
                        ),
                        alpha=float(
                            trade["alpha"]
                        ),
                        original_exit_date=pd.Timestamp(
                            trade["exit_date"]
                        ),
                        original_exit_price=float(
                            trade["exit"]
                        ),
                        original_exit_reason=str(
                            trade.get(
                                "exit_reason",
                                "normal_exit",
                            )
                        ),
                    )

        # ====================================================
        # 4. Daily MTM
        # ====================================================

        position_value = 0.0

        for stock_id, position in positions.items():

            table = price_lookup.get(
                stock_id
            )

            if table is None:
                continue

            history = table.loc[
                table.index <= current_date
            ]

            if history.empty:
                continue

            close_price = float(
                history[
                    "close"
                ].iloc[-1]
            )

            if (
                np.isfinite(
                    close_price
                )
                and close_price > 0
            ):

                position_value += (
                    position.shares
                    * close_price
                )

        equity = (
            cash
            + position_value
        )

        exposure = (
            position_value
            / equity
            if equity > 0
            else 0.0
        )

        equity_rows.append(
            {
                "date": current_date,
                "equity": equity,
                "cash": cash,
                "position_value": position_value,
                "open_positions": len(
                    positions
                ),
                "exposure": exposure,
                "turnover_cash": turnover_cash,
                "regime": str(
                    market.loc[
                        current_date,
                        "regime",
                    ]
                ),
            }
        )

    equity_df = pd.DataFrame(
        equity_rows
    )

    trades_df = pd.DataFrame(
        closed_trades
    )

    if not equity_df.empty:

        equity_df[
            "daily_return"
        ] = (
            equity_df[
                "equity"
            ]
            .pct_change()
            .fillna(0.0)
        )

    return (
        equity_df,
        trades_df,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark: pd.DataFrame,
    initial_capital: float,
) -> dict:

    if equity.empty:
        return {}

    initial = float(
        initial_capital
    )

    final_equity = float(
        equity["equity"].iloc[-1]
    )

    total_return = (
        final_equity
        / initial
        - 1.0
    )

    elapsed_days = (
        equity["date"].iloc[-1]
        - equity["date"].iloc[0]
    ).days

    years = max(
        elapsed_days / 365.25,
        1.0 / 365.25,
    )

    cagr = (
        final_equity
        / initial
    ) ** (
        1.0 / years
    ) - 1.0

    peak = (
        equity["equity"]
        .cummax()
    )

    drawdown = (
        equity["equity"]
        / peak
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    daily_returns = (
        equity["equity"]
        .pct_change()
        .fillna(0.0)
    )

    std = daily_returns.std(
        ddof=1
    )

    sharpe = (
        daily_returns.mean()
        / std
        * np.sqrt(252)
        if len(daily_returns) > 1
        and std > 0
        else 0.0
    )

    benchmark_period = benchmark.loc[
        (benchmark.index >= equity["date"].iloc[0])
        & (benchmark.index <= equity["date"].iloc[-1])
    ]

    benchmark_returns = (
        pd.to_numeric(
            benchmark_period[
                "return"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    benchmark_return = float(
        (1.0 + benchmark_returns)
        .prod()
        - 1.0
    )

    portfolio_alpha = (
        total_return
        - benchmark_return
    )

    if trades.empty:

        win_rate = 0.0
        avg_return = 0.0
        avg_alpha = 0.0
        profit_factor = 0.0
        forced_exits = 0
        avg_exposure = float(
            equity["exposure"].mean()
        )
        turnover_pct = 0.0

    else:

        returns = (
            pd.to_numeric(
                trades["net_return"],
                errors="coerce",
            )
            .dropna()
        )

        wins = returns[
            returns > 0
        ]

        losses = returns[
            returns < 0
        ]

        gross_profit = float(
            wins.sum()
        )

        gross_loss = abs(
            float(losses.sum())
        )

        win_rate = float(
            (returns > 0).mean()
        )

        avg_return = float(
            returns.mean()
        )

        avg_alpha = float(
            pd.to_numeric(
                trades["alpha"],
                errors="coerce",
            ).mean()
        )

        profit_factor = (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else 0.0
        )

        forced_exits = int(
            (
                trades[
                    "exit_reason"
                ]
                == "regime_exit"
            ).sum()
        )

        avg_exposure = float(
            equity[
                "exposure"
            ].mean()
        )

        turnover_pct = (
            float(
                equity[
                    "turnover_cash"
                ].iloc[-1]
            )
            / initial
            * 100.0
        )

    calmar = (
        cagr
        / abs(max_drawdown)
        if max_drawdown < 0
        else 0.0
    )

    return {
        "initial_capital": round(
            initial,
            2,
        ),
        "final_equity": round(
            final_equity,
            2,
        ),
        "total_return_pct": round(
            total_return * 100,
            2,
        ),
        "cagr_pct": round(
            cagr * 100,
            2,
        ),
        "benchmark_return_pct": round(
            benchmark_return * 100,
            2,
        ),
        "portfolio_alpha_pct": round(
            portfolio_alpha * 100,
            2,
        ),
        "max_drawdown_pct": round(
            max_drawdown * 100,
            2,
        ),
        "sharpe": round(
            float(sharpe),
            2,
        ),
        "calmar": round(
            float(calmar),
            2,
        ),
        "trades": int(
            len(trades)
        ),
        "win_rate_pct": round(
            win_rate * 100,
            2,
        ),
        "avg_trade_return_pct": round(
            avg_return * 100,
            2,
        ),
        "avg_trade_alpha_pct": round(
            avg_alpha * 100,
            2,
        ),
        "profit_factor": round(
            profit_factor,
            2,
        ),
        "forced_regime_exits": forced_exits,
        "avg_exposure_pct": round(
            avg_exposure * 100,
            2,
        ),
        "turnover_pct": round(
            turnover_pct,
            2,
        ),
    }


# ============================================================
# Period runner
#
# Research and OOS each start independently from 1M.
# ============================================================

def run_period(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    period_candidates = candidates[
        (
            candidates[
                "entry_date"
            ].dt.year >= start_year
        )
        & (
            candidates[
                "entry_date"
            ].dt.year <= end_year
        )
    ].copy()

    period_market = market[
        (
            market.index.year
            >= start_year
        )
        & (
            market.index.year
            <= end_year
        )
    ].copy()

    if (
        period_candidates.empty
        or period_market.empty
    ):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )

    equity, trades = run_backtest(
        period_candidates,
        prices,
        period_market,
    )

    metrics = calculate_metrics(
        equity,
        trades,
        period_market,
        INITIAL_CAPITAL,
    )

    return (
        equity,
        trades,
        metrics,
    )


# ============================================================
# Yearly metrics
# ============================================================

def yearly_metrics(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:

    years = sorted(
        candidates[
            "entry_date"
        ]
        .dt.year
        .unique()
        .tolist()
    )

    rows = []

    for year in years:

        _, _, metrics = run_period(
            candidates,
            prices,
            market,
            int(year),
            int(year),
        )

        if not metrics:
            continue

        rows.append(
            {
                "year": int(year),
                "return_pct": metrics.get(
                    "total_return_pct",
                    0.0,
                ),
                "benchmark_pct": metrics.get(
                    "benchmark_return_pct",
                    0.0,
                ),
                "alpha_pct": metrics.get(
                    "portfolio_alpha_pct",
                    0.0,
                ),
                "mdd_pct": metrics.get(
                    "max_drawdown_pct",
                    0.0,
                ),
                "sharpe": metrics.get(
                    "sharpe",
                    0.0,
                ),
                "profit_factor": metrics.get(
                    "profit_factor",
                    0.0,
                ),
                "win_rate_pct": metrics.get(
                    "win_rate_pct",
                    0.0,
                ),
                "trades": metrics.get(
                    "trades",
                    0,
                ),
                "forced_regime_exits": metrics.get(
                    "forced_regime_exits",
                    0,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Final audit
# ============================================================

def run_audits(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
) -> dict:

    regime_schedule, audit_rows = (
        build_regime_exit_schedule(
            market
        )
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    # --------------------------------------------------------
    # Look-ahead audit
    # --------------------------------------------------------

    if audit_df.empty:

        lookahead_pass = False

    else:

        lookahead_pass = bool(
            audit_df[
                "lookahead_safe"
            ].all()
        )

    # --------------------------------------------------------
    # Entry-date sanity
    #
    # Entry date must be strictly after signal date.
    # --------------------------------------------------------

    entry_timing_pass = bool(
        (
            candidates["entry_date"]
            > candidates["signal_date"]
        ).all()
    )

    # --------------------------------------------------------
    # Candidate data sanity
    # --------------------------------------------------------

    candidate_sanity_pass = bool(
        (
            candidates[
                "rs20"
            ] > RS_THRESHOLD
        ).all()
        and
        (
            candidates[
                "drawdown_20d"
            ]
            >= PULLBACK_MIN
        ).all()
        and
        (
            candidates[
                "drawdown_20d"
            ]
            <= PULLBACK_MAX
        ).all()
    )

    # --------------------------------------------------------
    # Universe sanity
    # --------------------------------------------------------

    universe_count = (
        prices["stock_id"]
        .nunique()
    )

    universe_pass = (
        universe_count == 150
    )

    # --------------------------------------------------------
    # Benchmark sanity
    # --------------------------------------------------------

    benchmark_valid = (
        market[
            "return"
        ]
        .notna()
        .all()
    )

    regime_valid = set(
        market[
            "regime"
        ]
        .dropna()
        .unique()
    ).issubset(
        {
            "bull",
            "neutral",
            "bear",
        }
    )

    return {
        "lookahead_pass": lookahead_pass,
        "entry_timing_pass": entry_timing_pass,
        "candidate_sanity_pass": candidate_sanity_pass,
        "universe_pass": universe_pass,
        "benchmark_valid": bool(
            benchmark_valid
        ),
        "regime_valid": bool(
            regime_valid
        ),
        "universe_count": universe_count,
        "regime_exit_events": len(
            audit_df
        ),
        "regime_exit_schedule_events": len(
            regime_schedule
        ),
        "audit_rows": audit_df,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        "=== V1.5 FINAL AUDIT ==="
    )

    print(
        "=" * 100
    )

    print(
        "\n[FROZEN STRATEGY]"
    )

    print(
        "RS20 > 10%"
    )

    print(
        "Pullback -7% ~ -2%"
    )

    print(
        "Top 3 by RS20"
    )

    print(
        "Bull -> Neutral"
    )

    print(
        "1 confirmation day"
    )

    print(
        "T+1 Open regime exit"
    )

    print(
        "Original V1.3 stop/take-profit/time exit"
    )

    print(
        "Round-trip cost 0.50%"
    )

    candidates = load_candidates()

    print(
        f"\nCandidates: "
        f"{len(candidates)}"
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    try:

        universe_ids = (
            get_universe_stock_ids(
                conn
            )
        )

        prices = load_prices(
            conn,
            universe_ids,
        )

    finally:

        conn.close()

    print(
        f"Price universe: "
        f"{prices['stock_id'].nunique()} stocks"
    )

    market = build_market_regime(
        prices
    )

    start_date = candidates[
        "entry_date"
    ].min()

    end_date = candidates[
        "exit_date"
    ].max()

    market = market.loc[
        (
            market.index
            >= start_date
        )
        & (
            market.index
            <= end_date
        )
    ].copy()

    print(
        "\n[REGIME DISTRIBUTION]"
    )

    print(
        market[
            "regime"
        ]
        .value_counts()
        .to_dict()
    )

    # ========================================================
    # Audits
    # ========================================================

    audits = run_audits(
        candidates,
        prices,
        market,
    )

    print(
        "\n[AUDIT CHECKS]"
    )

    print(
        f"Look-ahead audit: "
        f"{'PASS' if audits['lookahead_pass'] else 'FAIL'}"
    )

    print(
        f"Entry timing: "
        f"{'PASS' if audits['entry_timing_pass'] else 'FAIL'}"
    )

    print(
        f"Candidate definition: "
        f"{'PASS' if audits['candidate_sanity_pass'] else 'FAIL'}"
    )

    print(
        f"150-stock universe: "
        f"{'PASS' if audits['universe_pass'] else 'FAIL'}"
    )

    print(
        f"Benchmark data: "
        f"{'PASS' if audits['benchmark_valid'] else 'FAIL'}"
    )

    print(
        f"Regime labels: "
        f"{'PASS' if audits['regime_valid'] else 'FAIL'}"
    )

    print(
        f"Regime exit events: "
        f"{audits['regime_exit_events']}"
    )

    # ========================================================
    # Research
    # ========================================================

    (
        research_equity,
        research_trades,
        research_metrics,
    ) = run_period(
        candidates,
        prices,
        market,
        2021,
        2024,
    )

    # ========================================================
    # OOS
    # ========================================================

    (
        oos_equity,
        oos_trades,
        oos_metrics,
    ) = run_period(
        candidates,
        prices,
        market,
        2025,
        2026,
    )

    # ========================================================
    # Full
    #
    # Full is only a descriptive statistic.
    # Research/OOS are the validation statistics.
    # ========================================================

    (
        full_equity,
        full_trades,
    ) = run_backtest(
        candidates,
        prices,
        market,
    )

    full_metrics = calculate_metrics(
        full_equity,
        full_trades,
        market,
        INITIAL_CAPITAL,
    )

    # ========================================================
    # Metrics
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "=== FULL PERIOD ==="
    )

    print(
        "=" * 100
    )

    for key, value in full_metrics.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "=== RESEARCH 2021-2024 ==="
    )

    print(
        "=" * 100
    )

    for key, value in research_metrics.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "=== OOS 2025-2026 ==="
    )

    print(
        "=" * 100
    )

    for key, value in oos_metrics.items():

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # Yearly
    # ========================================================

    yearly = yearly_metrics(
        candidates,
        prices,
        market,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "=== YEARLY PERFORMANCE ==="
    )

    print(
        "=" * 100
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    # ========================================================
    # Final decision logic
    # ========================================================

    research_positive_alpha = (
        research_metrics.get(
            "portfolio_alpha_pct",
            -999.0,
        )
        > 0
    )

    oos_positive_alpha = (
        oos_metrics.get(
            "portfolio_alpha_pct",
            -999.0,
        )
        > 0
    )

    research_pf_ok = (
        research_metrics.get(
            "profit_factor",
            0.0,
        )
        > 1.0
    )

    oos_pf_ok = (
        oos_metrics.get(
            "profit_factor",
            0.0,
        )
        > 1.0
    )

    audit_pass = all(
        [
            audits["lookahead_pass"],
            audits["entry_timing_pass"],
            audits["candidate_sanity_pass"],
            audits["universe_pass"],
            audits["benchmark_valid"],
            audits["regime_valid"],
        ]
    )

    final_candidate_pass = all(
        [
            audit_pass,
            research_positive_alpha,
            oos_positive_alpha,
            research_pf_ok,
            oos_pf_ok,
        ]
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "=== V1.5 FINAL AUDIT RESULT ==="
    )

    print(
        "=" * 100
    )

    print(
        f"Technical audits: "
        f"{'PASS' if audit_pass else 'FAIL'}"
    )

    print(
        f"Research Alpha > 0: "
        f"{'PASS' if research_positive_alpha else 'FAIL'}"
    )

    print(
        f"OOS Alpha > 0: "
        f"{'PASS' if oos_positive_alpha else 'FAIL'}"
    )

    print(
        f"Research PF > 1: "
        f"{'PASS' if research_pf_ok else 'FAIL'}"
    )

    print(
        f"OOS PF > 1: "
        f"{'PASS' if oos_pf_ok else 'FAIL'}"
    )

    print(
        "\nFINAL CANDIDATE: "
        + (
            "PASS"
            if final_candidate_pass
            else "FAIL"
        )
    )

    # ========================================================
    # Save reports
    # ========================================================

    full_equity.to_csv(
        "portfolio_equity_v1_5_final.csv",
        index=False,
    )

    full_trades.to_csv(
        "portfolio_trades_v1_5_final.csv",
        index=False,
    )

    yearly.to_csv(
        "yearly_performance_v1_5_final.csv",
        index=False,
    )

    audits["audit_rows"].to_csv(
        "regime_exit_audit_v1_5.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                **{
                    f"research_{k}": v
                    for k, v in research_metrics.items()
                },
                **{
                    f"oos_{k}": v
                    for k, v in oos_metrics.items()
                },
                "technical_audit_pass": audit_pass,
                "final_candidate_pass": final_candidate_pass,
            }
        ]
    ).to_csv(
        "final_audit_summary_v1_5.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "portfolio_equity_v1_5_final.csv"
    )

    print(
        "portfolio_trades_v1_5_final.csv"
    )

    print(
        "yearly_performance_v1_5_final.csv"
    )

    print(
        "regime_exit_audit_v1_5.csv"
    )

    print(
        "final_audit_summary_v1_5.csv"
    )


if __name__ == "__main__":
    main()


