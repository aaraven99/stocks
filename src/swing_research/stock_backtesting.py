"""Point-in-time constituent, next-open cross-sectional stock portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .backtesting import CostModel, performance_metrics
from .data import validate_ohlcv
from .portfolio_backtesting import PortfolioBacktestResult


@dataclass(frozen=True)
class CrossSectionalMomentumSpec:
    momentum_lookback_sessions: int
    trend_lookback_sessions: int
    volatility_lookback_sessions: int
    rebalance_sessions: int
    top_n: int


def _price_matrices(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = pd.concat(
        {ticker: validate_ohlcv(frame)["close"] for ticker, frame in frames.items()}, axis=1
    ).sort_index()
    opens = pd.concat(
        {ticker: validate_ohlcv(frame)["open"] for ticker, frame in frames.items()}, axis=1
    ).sort_index()
    if closes.empty or opens.empty:
        raise ValueError("No completed price history supplied")
    return closes, opens.reindex(index=closes.index, columns=closes.columns)


def _target_weights(
    closes: pd.DataFrame,
    membership: pd.Series,
    position: int,
    spec: CrossSectionalMomentumSpec,
) -> pd.Series:
    """Rank only securities present in the previous point-in-time constituent snapshot."""
    weights = pd.Series(0.0, index=closes.columns)
    required = max(
        spec.momentum_lookback_sessions,
        spec.trend_lookback_sessions,
        spec.volatility_lookback_sessions,
    )
    if position < required:
        return weights
    eligible: list[tuple[str, float, float]] = []
    for ticker in membership.index[membership]:
        history = closes[ticker].iloc[position - required : position + 1]
        if history.isna().any():
            continue
        current = float(closes[ticker].iloc[position])
        momentum = (
            current / float(closes[ticker].iloc[position - spec.momentum_lookback_sessions]) - 1
        )
        trend = current / float(closes[ticker].iloc[position - spec.trend_lookback_sessions]) - 1
        volatility = float(
            closes[ticker]
            .pct_change()
            .iloc[position - spec.volatility_lookback_sessions + 1 : position + 1]
            .std(ddof=0)
        )
        if momentum > 0 and trend > 0 and volatility > 0:
            eligible.append((ticker, momentum / volatility, volatility))
    selected = sorted(eligible, key=lambda item: item[1], reverse=True)[: spec.top_n]
    if not selected:
        return weights
    inverse_volatility = pd.Series({ticker: 1 / volatility for ticker, _, volatility in selected})
    weights.loc[inverse_volatility.index] = inverse_volatility / inverse_volatility.sum()
    return weights


def run_point_in_time_cross_sectional_momentum(
    frames: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    spec: CrossSectionalMomentumSpec,
    evaluation_start: datetime | pd.Timestamp,
    evaluation_end: datetime | pd.Timestamp,
    costs: CostModel | None = None,
    initial_cash: float = 100_000.0,
    benchmark_tickers: tuple[str, ...] = ("SPY", "QQQ"),
) -> PortfolioBacktestResult:
    """Trade a dynamic historical universe with prior-close membership and next-open execution.

    Missing open/close bars for a security that is held are an error, not a reason to silently
    forward-fill or drop a delisted holding. A separate provider-coverage audit must run before
    calling this simulation on a broad historical universe.
    """
    if spec.top_n < 1 or spec.rebalance_sessions < 1:
        raise ValueError("top_n and rebalance_sessions must be positive")
    closes, opens = _price_matrices(frames)
    member = (
        membership.reindex(index=closes.index, columns=closes.columns).fillna(False).astype(bool)
    )
    start = pd.Timestamp(evaluation_start).tz_localize(None).normalize()
    end = pd.Timestamp(evaluation_end).tz_localize(None).normalize()
    start_position = int(closes.index.searchsorted(start, side="left"))
    end_position = int(closes.index.searchsorted(end, side="right")) - 1
    required = max(
        spec.momentum_lookback_sessions,
        spec.trend_lookback_sessions,
        spec.volatility_lookback_sessions,
    )
    if start_position - 1 < required or end_position <= start_position:
        raise ValueError("Evaluation period does not contain enough prior completed history")
    effective_costs = costs or CostModel()
    shares = pd.Series(0.0, index=closes.columns)
    cash = initial_cash
    dates = [closes.index[start_position - 1]]
    equity_points = [initial_cash]
    turnover_points = [0.0]
    weight_points: list[pd.Series] = [pd.Series(0.0, index=closes.columns)]
    for position in range(start_position, end_position + 1):
        open_prices = opens.iloc[position]
        held = shares[shares != 0]
        if open_prices.reindex(held.index).isna().any():
            raise ValueError("Missing opening price for a held security; coverage is inadmissible")
        prior_members = member.iloc[position - 1]
        forced_exits = held.index[~prior_members.reindex(held.index).fillna(False)]
        turnover = 0.0
        if len(forced_exits):
            proceeds = float((shares.loc[forced_exits] * open_prices.loc[forced_exits]).sum())
            forced_cost = proceeds * effective_costs.one_way_fraction
            cash += proceeds - forced_cost
            turnover += proceeds / max(cash + float((shares * open_prices).sum()), 1)
            shares.loc[forced_exits] = 0.0
        values_at_open = shares * open_prices
        equity_at_open = cash + float(values_at_open.sum())
        rebalancing = (position - start_position) % spec.rebalance_sessions == 0
        target = _target_weights(closes, prior_members, position - 1, spec)
        if rebalancing:
            desired_values = target * equity_at_open
            for _ in range(2):
                traded_value = float((desired_values - values_at_open).abs().sum())
                cost = traded_value * effective_costs.one_way_fraction
                desired_values = target * max(0.0, equity_at_open - cost)
            turnover += traded_value / max(equity_at_open, 1)
            shares = desired_values / open_prices
            cash = equity_at_open - cost - float(desired_values.sum())
        close_prices = closes.iloc[position]
        held = shares[shares != 0]
        if close_prices.reindex(held.index).isna().any():
            raise ValueError("Missing closing price for a held security; coverage is inadmissible")
        close_values = shares * close_prices
        equity_at_close = cash + float(close_values.sum())
        dates.append(closes.index[position])
        equity_points.append(equity_at_close)
        turnover_points.append(turnover)
        weight_points.append(close_values / max(equity_at_close, 1))
    equity = pd.Series(equity_points, index=pd.DatetimeIndex(dates), name="equity")
    turnover_series = pd.Series(turnover_points, index=equity.index, name="turnover")
    weights = pd.DataFrame(weight_points, index=equity.index).fillna(0.0)
    metrics = performance_metrics(equity, [], None, weights.sum(axis=1) > 0)
    metrics["turnover"] = float(turnover_series.sum())
    benchmark_metrics: dict[str, dict[str, float | int | None]] = {}
    for ticker in benchmark_tickers:
        if ticker not in closes:
            continue
        benchmark_prices = closes[ticker].reindex(equity.index)
        if benchmark_prices.isna().any():
            raise ValueError(f"Benchmark {ticker} has missing prices in the evaluation period")
        benchmark_equity = benchmark_prices / benchmark_prices.iloc[0] * initial_cash
        metrics_for_benchmark = performance_metrics(benchmark_equity, [], None)
        metrics_for_benchmark["terminal_wealth_multiple"] = float(
            equity.iloc[-1] / benchmark_equity.iloc[-1]
        )
        benchmark_metrics[ticker] = metrics_for_benchmark
    return PortfolioBacktestResult(equity, weights, turnover_series, metrics, benchmark_metrics)
