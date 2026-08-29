"""Long-only, next-session event backtester with realistic configurable costs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import validate_ohlcv


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 5.0

    @property
    def one_way_fraction(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    holding_sessions: int

    @property
    def return_fraction(self) -> float:
        return self.exit_price / self.entry_price - 1


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    metrics: dict[str, float | int | None]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 or np.isnan(denominator) else float(numerator / denominator)


def performance_metrics(
    equity: pd.Series,
    trades: list[Trade],
    benchmark_equity: pd.Series | None = None,
    invested: pd.Series | None = None,
) -> dict[str, float | int | None]:
    """Compute whole-period metrics without calling a flat strategy a profitable one."""
    returns = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = max(len(returns) / 252, 1 / 252)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(252)) if not returns.empty else 0.0
    excess = returns  # Cash rate intentionally fixed to zero and documented.
    sharpe = _safe_ratio(float(excess.mean() * 252), annualized_volatility)
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(252)
    sortino = _safe_ratio(float(excess.mean() * 252), float(downside))
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = _safe_ratio(cagr, abs(max_drawdown))
    positive = [trade.return_fraction for trade in trades if trade.return_fraction > 0]
    negative = [trade.return_fraction for trade in trades if trade.return_fraction < 0]
    gross_profit = sum(positive)
    gross_loss = abs(sum(negative))
    profit_factor = None if gross_loss == 0 else float(gross_profit / gross_loss)
    win_rate = None if not trades else float(len(positive) / len(trades))
    average_winner = float(np.mean(positive)) if positive else None
    average_loser = float(np.mean(negative)) if negative else None
    win_loss_ratio = (
        None
        if average_winner is None or average_loser in (None, 0)
        else float(abs(average_winner / average_loser))
    )
    expectancy = None if not trades else float(np.mean([trade.return_fraction for trade in trades]))
    exposure = (
        float(invested.reindex(equity.index).fillna(False).mean())
        if invested is not None
        else float((equity.pct_change().fillna(0) != 0).mean())
    )
    peak_index = equity.index[0]
    maximum_recovery = 0
    for index, value in equity.items():
        if value >= equity.loc[peak_index]:
            peak_index = index
        else:
            maximum_recovery = max(maximum_recovery, int((index - peak_index).days))
    metrics: dict[str, float | int | None] = {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_return": float(returns.mean() * 252) if not returns.empty else 0.0,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "maximum_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "average_winner": average_winner,
        "average_loser": average_loser,
        "win_loss_ratio": win_loss_ratio,
        "exposure": exposure,
        "turnover": float(len(trades) * 2 / max(len(equity), 1)),
        "recovery_time_days": maximum_recovery,
        "number_of_trades": len(trades),
        "average_holding_sessions": float(np.mean([trade.holding_sessions for trade in trades]))
        if trades
        else None,
        "alpha": None,
        "beta": None,
        "information_ratio": None,
        "benchmark_total_return": None,
        "benchmark_cagr": None,
    }
    if benchmark_equity is not None:
        benchmark_returns = benchmark_equity.reindex(returns.index).pct_change().dropna()
        aligned = pd.concat(
            [returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1
        ).dropna()
        if len(aligned) >= 2 and float(aligned["benchmark"].var()) > 0:
            beta = float(
                aligned["portfolio"].cov(aligned["benchmark"]) / aligned["benchmark"].var()
            )
            alpha = float((aligned["portfolio"].mean() - beta * aligned["benchmark"].mean()) * 252)
            active = aligned["portfolio"] - aligned["benchmark"]
            metrics["beta"] = beta
            metrics["alpha"] = alpha
            metrics["information_ratio"] = _safe_ratio(
                float(active.mean() * 252), float(active.std(ddof=0) * np.sqrt(252))
            )
        benchmark_total_return = float(benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1)
        metrics["benchmark_total_return"] = benchmark_total_return
        metrics["benchmark_cagr"] = float(
            (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) ** (1 / years) - 1
        )
    return metrics


def run_long_only_backtest(
    ohlcv: pd.DataFrame,
    close_signal: pd.Series,
    holding_period_sessions: int = 10,
    initial_cash: float = 100_000.0,
    costs: CostModel | None = None,
    benchmark_close: pd.Series | None = None,
) -> BacktestResult:
    """Trade the close decision only at next session's open; signals never trade same-bar close."""
    bars = validate_ohlcv(ohlcv)
    effective_costs = costs or CostModel()
    signals = close_signal.reindex(bars.index).fillna(False).astype(bool)
    cash = initial_cash
    shares = 0.0
    entry_time: pd.Timestamp | None = None
    entry_price = 0.0
    held_sessions = 0
    trades: list[Trade] = []
    equity_points: list[float] = [initial_cash]
    equity_index: list[pd.Timestamp] = [bars.index[0]]
    invested_flags: list[bool] = [False]

    for position, timestamp in enumerate(bars.index[1:], start=1):
        prior_signal = bool(signals.iloc[position - 1])
        open_price = float(bars.iloc[position]["open"])
        should_exit = shares > 0 and (not prior_signal or held_sessions >= holding_period_sessions)
        if should_exit and entry_time is not None:
            execution = open_price * (1 - effective_costs.one_way_fraction)
            cash += shares * execution
            trades.append(
                Trade(entry_time, timestamp, entry_price, execution, shares, held_sessions)
            )
            shares, entry_time, entry_price, held_sessions = 0.0, None, 0.0, 0
        if shares == 0 and prior_signal:
            execution = open_price * (1 + effective_costs.one_way_fraction)
            shares = cash / execution
            cash = 0.0
            entry_time, entry_price, held_sessions = timestamp, execution, 0
        if shares > 0:
            held_sessions += 1
        equity_points.append(cash + shares * float(bars.iloc[position]["close"]))
        equity_index.append(timestamp)
        invested_flags.append(shares > 0)

    equity = pd.Series(equity_points, index=pd.DatetimeIndex(equity_index), name="equity")
    benchmark_equity = None
    if benchmark_close is not None:
        benchmark = benchmark_close.reindex(equity.index).ffill().dropna()
        if not benchmark.empty:
            benchmark_equity = benchmark / benchmark.iloc[0] * initial_cash
    invested = pd.Series(invested_flags, index=pd.DatetimeIndex(equity_index), name="invested")
    metrics = performance_metrics(equity, trades, benchmark_equity, invested)
    return BacktestResult(equity=equity, trades=trades, metrics=metrics)


def result_as_dict(result: BacktestResult) -> dict[str, Any]:
    return {
        "metrics": result.metrics,
        "trades": [
            {**asdict(trade), "return_fraction": trade.return_fraction} for trade in result.trades
        ],
    }
