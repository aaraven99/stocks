"""Command-line entry points for daily research, demos, backtests, and Actions gating."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from .backtesting import CostModel, result_as_dict, run_long_only_backtest
from .config import load_config, load_yaml
from .data import YFinancePriceProvider
from .features import build_technical_features
from .market_calendar import should_start_daily_workflow
from .pipeline import run_daily_research, run_demo_research
from .relative_strength_research import (
    ResearchPeriod,
    experiment_as_dict,
    robust_experiment_as_dict,
    run_predeclared_experiment,
    run_robust_experiment,
)
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


def _command_portfolio_study(args: argparse.Namespace) -> int:
    root = _root()
    document = load_yaml(root / args.config)
    research = document["research"]
    provider = YFinancePriceProvider()
    data_start = datetime.fromisoformat(str(research["data_start"]))
    data_end = datetime.fromisoformat(str(research["holdout_end"]))
    frames = {
        ticker: provider.fetch_daily(ticker, data_start, data_end)
        for ticker in research["instruments"]
    }
    costs = CostModel(**research["costs"])
    experiment = run_predeclared_experiment(
        frames,
        research,
        ResearchPeriod(
            "train",
            datetime.fromisoformat(str(research["train_start"])),
            datetime.fromisoformat(str(research["train_end"])),
        ),
        ResearchPeriod(
            "validation",
            datetime.fromisoformat(str(research["validation_start"])),
            datetime.fromisoformat(str(research["validation_end"])),
        ),
        ResearchPeriod(
            "holdout",
            datetime.fromisoformat(str(research["holdout_start"])),
            data_end,
        ),
        costs,
    )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(experiment_as_dict(experiment), indent=2), encoding="utf-8")
    print(f"Wrote predeclared portfolio study: {output}")
    print(json.dumps(experiment.holdout, indent=2))
    return 0


def _command_robust_portfolio_study(args: argparse.Namespace) -> int:
    root = _root()
    document = load_yaml(root / args.config)
    research = document["research"]
    provider = YFinancePriceProvider()
    data_start = datetime.fromisoformat(str(research["data_start"]))
    data_end = datetime.fromisoformat(str(research["final_holdout_end"]))
    frames = {
        ticker: provider.fetch_daily(ticker, data_start, data_end)
        for ticker in research["instruments"]
    }
    development = [
        ResearchPeriod(
            str(period["name"]),
            datetime.fromisoformat(str(period["start"])),
            datetime.fromisoformat(str(period["end"])),
        )
        for period in research["robustness_development_periods"]
    ]
    experiment = run_robust_experiment(
        frames,
        research,
        development,
        ResearchPeriod(
            "validation",
            datetime.fromisoformat(str(research["robustness_validation_start"])),
            datetime.fromisoformat(str(research["robustness_validation_end"])),
        ),
        ResearchPeriod(
            "final_holdout",
            datetime.fromisoformat(str(research["final_holdout_start"])),
            data_end,
        ),
        CostModel(**research["costs"]),
    )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(robust_experiment_as_dict(experiment), indent=2), encoding="utf-8")
    print(f"Wrote robust portfolio study: {output}")
    print(json.dumps(experiment.validation, indent=2))
    print(json.dumps(experiment.final_holdout, indent=2))
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
    study = commands.add_parser(
        "portfolio-study",
        help="Run predeclared train-only ETF relative-strength selection and untouched holdout",
    )
    study.add_argument("--config", default="config/relative_strength_research.yaml")
    study.add_argument(
        "--output", default="reports/backtests/predeclared-etf-relative-strength.json"
    )
    study.set_defaults(handler=_command_portfolio_study)
    robust_study = commands.add_parser(
        "robust-portfolio-study",
        help="Select across development folds before later validation and final holdout",
    )
    robust_study.add_argument("--config", default="config/relative_strength_research.yaml")
    robust_study.add_argument(
        "--output", default="reports/backtests/robust-etf-relative-strength.json"
    )
    robust_study.set_defaults(handler=_command_robust_portfolio_study)
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
