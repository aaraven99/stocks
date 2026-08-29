"""Leakage-safe, next-open cross-asset portfolio simulations for research only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .backtesting import CostModel, performance_metrics
from .data import validate_ohlcv


@dataclass(frozen=True)
class RelativeStrengthSpec:
    momentum_lookback_sessions: int
    trend_lookback_sessions: int
    volatility_lookback_sessions: int
    rebalance_sessions: int
    top_n: int
    risk_assets: tuple[str, ...]
    defensive_asset: str


@dataclass
class PortfolioBacktestResult:
    equity: pd.Series
    target_weights: pd.DataFrame
    daily_turnover: pd.Series
    metrics: dict[str, float | int | None]
    benchmark_metrics: dict[str, dict[str, float | int | None]]


def close_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align completed bars; an incomplete or missing asset cannot be silently forward-filled."""
    closes = {
        ticker: validate_ohlcv(frame)["close"].rename(ticker) for ticker, frame in frames.items()
    }
    aligned = pd.concat(closes.values(), axis=1, join="inner")
    aligned.columns = list(closes)
    if aligned.empty:
        raise ValueError("No common completed sessions across the research instruments")
    return aligned.sort_index()


def _target_weights(prices: pd.DataFrame, position: int, spec: RelativeStrengthSpec) -> pd.Series:
    """Use data through `position` close only; the caller executes at the next session's open."""
    lookback = max(spec.momentum_lookback_sessions, spec.trend_lookback_sessions)
    if position < lookback:
        return pd.Series({spec.defensive_asset: 1.0}, dtype=float)
    current = prices.iloc[position]
    momentum = current / prices.iloc[position - spec.momentum_lookback_sessions] - 1
    trend = current / prices.iloc[position - spec.trend_lookback_sessions] - 1
    volatility = prices.pct_change().iloc[
        position - spec.volatility_lookback_sessions + 1 : position + 1
    ].std(ddof=0)
    eligible = [
        ticker
        for ticker in spec.risk_assets
        if ticker in prices
        and momentum[ticker] > 0
        and trend[ticker] > 0
        and volatility[ticker] > 0
    ]
    ranked = sorted(
        eligible,
        key=lambda ticker: float(momentum[ticker] / volatility[ticker]),
        reverse=True,
    )
    selected = ranked[: spec.top_n]
    weights = pd.Series(0.0, index=prices.columns)
    if not selected:
        weights.loc[spec.defensive_asset] = 1.0
        return weights
    inverse_volatility = (1 / volatility.loc[selected]).replace([np.inf, -np.inf], np.nan).dropna()
    if inverse_volatility.empty:
        weights.loc[spec.defensive_asset] = 1.0
        return weights
    weights.loc[inverse_volatility.index] = inverse_volatility / inverse_volatility.sum()
    return weights


def run_relative_strength_portfolio(
    frames: dict[str, pd.DataFrame],
    spec: RelativeStrengthSpec,
    evaluation_start: datetime | pd.Timestamp,
    evaluation_end: datetime | pd.Timestamp,
    costs: CostModel | None = None,
    initial_cash: float = 100_000.0,
    benchmark_tickers: tuple[str, ...] = ("SPY", "QQQ"),
) -> PortfolioBacktestResult:
    """Execute close signals next session open, with assets held as shares between rebalances."""
    prices = close_matrix(frames)
    opens = pd.concat(
        {ticker: validate_ohlcv(frame)["open"] for ticker, frame in frames.items()},
        axis=1,
        join="inner",
    ).reindex(columns=prices.columns)
    prices = prices.reindex(opens.index).dropna()
    opens = opens.reindex(prices.index).dropna()
    effective_costs = costs or CostModel()
    start = pd.Timestamp(evaluation_start).tz_localize(None).normalize()
    end = pd.Timestamp(evaluation_end).tz_localize(None).normalize()
    start_position = int(prices.index.searchsorted(start, side="left"))
    end_position = int(prices.index.searchsorted(end, side="right")) - 1
    if start_position <= 0 or end_position <= start_position:
        raise ValueError("Evaluation period does not contain enough complete sessions")
    required_history = max(spec.momentum_lookback_sessions, spec.trend_lookback_sessions)
    if start_position - 1 < required_history:
        raise ValueError("Evaluation period does not include the strategy's required prior history")
    shares = pd.Series(0.0, index=prices.columns)
    cash = initial_cash
    equity_points = [initial_cash]
    dates = [prices.index[start_position - 1]]
    turnover_points = [0.0]
    weight_points: list[pd.Series] = [pd.Series(0.0, index=prices.columns)]
    for position in range(start_position, end_position + 1):
        open_prices = opens.iloc[position]
        current_at_open = shares * open_prices
        equity_at_open = cash + float(current_at_open.sum())
        rebalance = (position - start_position) % spec.rebalance_sessions == 0
        turnover = 0.0
        if rebalance:
            target = _target_weights(prices, position - 1, spec)
            desired_values = target * equity_at_open
            for _ in range(2):
                traded_value = float((desired_values - current_at_open).abs().sum())
                cost = traded_value * effective_costs.one_way_fraction
                desired_values = target * max(0.0, equity_at_open - cost)
            turnover = traded_value / max(equity_at_open, 1)
            shares = desired_values / open_prices
            cash = equity_at_open - cost - float(desired_values.sum())
        close_values = shares * prices.iloc[position]
        equity_at_close = cash + float(close_values.sum())
        weight_points.append(close_values / max(equity_at_close, 1))
        equity_points.append(equity_at_close)
        dates.append(prices.index[position])
        turnover_points.append(turnover)
    equity = pd.Series(equity_points, index=pd.DatetimeIndex(dates), name="equity")
    turnover_series = pd.Series(turnover_points, index=equity.index, name="turnover")
    weights = pd.DataFrame(weight_points, index=equity.index).fillna(0.0)
    invested = weights.sum(axis=1) > 0
    metrics = performance_metrics(equity, [], None, invested)
    metrics["turnover"] = float(turnover_series.sum())
    benchmark_metrics: dict[str, dict[str, float | int | None]] = {}
    for ticker in benchmark_tickers:
        if ticker not in prices:
            continue
        benchmark = prices[ticker].reindex(equity.index).ffill()
        benchmark_equity = benchmark / benchmark.iloc[0] * initial_cash
        benchmark_metrics[ticker] = performance_metrics(benchmark_equity, [], None)
        terminal_multiple = float(equity.iloc[-1] / benchmark_equity.iloc[-1])
        benchmark_metrics[ticker]["terminal_wealth_multiple"] = terminal_multiple
        strategy_return = float(metrics["total_return"] or 0)
        benchmark_return = float(benchmark_metrics[ticker]["total_return"] or 0)
        benchmark_metrics[ticker]["relative_total_return"] = strategy_return - benchmark_return
    return PortfolioBacktestResult(equity, weights, turnover_series, metrics, benchmark_metrics)
