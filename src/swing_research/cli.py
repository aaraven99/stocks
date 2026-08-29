"""Command-line entry points for daily research, demos, backtests, and Actions gating."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from .backtesting import CostModel, result_as_dict, run_long_only_backtest
from .config import load_config
from .data import YFinancePriceProvider
from .features import build_technical_features
from .market_calendar import should_start_daily_workflow
from .pipeline import run_daily_research, run_demo_research
from .reporting import write_equity_curve_svg, write_report
from .storage import PaperLedger


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _command_daily(args: argparse.Namespace) -> int:
    root = _root()
    load_dotenv(root / ".env")
    ledger = PaperLedger(root / args.database)
    try:
        _, _, report = run_daily_research(root / args.config, ledger=ledger)
        output = (
            root / args.output
            if args.output
            else root / "reports" / "daily" / f"{datetime.now(UTC):%Y-%m-%d}.md"
        )
        write_report(output, report)
        print(f"Wrote {output}; total persisted predictions: {ledger.prediction_count()}")
    finally:
        ledger.close()
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    _, _, report = run_demo_research()
    output = _root() / args.output
    write_report(output, report)
    print(f"Wrote deterministic offline demo report: {output}")
    return 0


def _command_backtest(args: argparse.Namespace) -> int:
    root = _root()
    configuration = load_config(root / args.config)
    provider = YFinancePriceProvider()
    end = datetime.now(UTC)
    start = end - timedelta(days=args.years * 366)
    price = provider.fetch_daily(args.ticker, start, end)
    benchmark = provider.fetch_daily("SPY", start, end)
    features = build_technical_features(price, benchmark["close"])
    threshold = float(configuration["strategy"]["strategy"]["thresholds"]["entry_score"])
    result = run_long_only_backtest(
        price,
        features["technical_score"] >= threshold,
        holding_period_sessions=int(
            configuration["strategy"]["strategy"]["holding_period_sessions"]
        ),
        costs=CostModel(**configuration["strategy"]["costs"]),
        benchmark_close=benchmark["close"],
    )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_as_dict(result), indent=2, default=str), encoding="utf-8")
    write_equity_curve_svg(result.equity, output.with_suffix(".svg"))
    print(f"Wrote backtest artifact: {output}")
    print(json.dumps(result.metrics, indent=2))
    return 0


def _command_workflow_gate(_: argparse.Namespace) -> int:
    allowed = should_start_daily_workflow(datetime.now(UTC))
    print(f"run_daily={'true' if allowed else 'false'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swing-research")
    commands = parser.add_subparsers(dest="command", required=True)
    daily = commands.add_parser("daily", help="Run a live-data morning research pipeline")
    daily.add_argument("--config", default="config")
    daily.add_argument("--output")
    daily.add_argument("--database", default="data/paper_ledger.sqlite3")
    daily.set_defaults(handler=_command_daily)
    demo = commands.add_parser("demo", help="Run deterministic offline research fixture")
    demo.add_argument("--output", default="reports/daily/demo.md")
    demo.set_defaults(handler=_command_demo)
    backtest = commands.add_parser(
        "backtest", help="Run a single-ticker cost-aware historical backtest"
    )
    backtest.add_argument("--ticker", default="MSFT")
    backtest.add_argument("--years", type=int, default=3)
    backtest.add_argument("--config", default="config")
    backtest.add_argument("--output", default="reports/backtests/latest-backtest.json")
    backtest.set_defaults(handler=_command_backtest)
    gate = commands.add_parser(
        "workflow-gate", help="Print whether a 5 AM Chicago NYSE run is allowed"
    )
    gate.set_defaults(handler=_command_workflow_gate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
