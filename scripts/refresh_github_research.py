"""Refresh the evidence table used to choose the platform's open-source stack.

Requires an authenticated GitHub CLI (`gh auth login`). It reads public metadata only and does not
clone, execute, or vendor any candidate repository.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CANDIDATES = [
    "OpenBB-finance/OpenBB",
    "microsoft/qlib",
    "polakowo/vectorbt",
    "mementum/backtrader",
    "kernc/backtesting.py",
    "quantopian/zipline",
    "nautechsystems/nautilus_trader",
    "QuantConnect/Lean",
    "AI4Finance-Foundation/FinRL",
    "AI4Finance-Foundation/FinGPT",
    "AI4Finance-Foundation/FinRobot",
    "TauricResearch/TradingAgents",
    "virattt/ai-financial-agent",
    "The-Swarm-Corporation/AutoHedge",
    "dgunning/edgartools",
    "JerBouma/FinanceToolkit",
    "ranaroussi/quantstats",
    "PyPortfolio/PyPortfolioOpt",
    "dcajasn/Riskfolio-Lib",
    "quantopian/alphalens",
    "pmorissette/bt",
    "skfolio/skfolio",
    "hudson-and-thames/mlfinlab",
    "edtechre/pybroker",
    "stefan-jansen/machine-learning-for-trading",
    "hugo2046/QuantsPlaybook",
    "TA-Lib/ta-lib-python",
    "bukosabino/ta",
    "ranaroussi/yfinance",
    "rsheftel/pandas_market_calendars",
    "alpacahq/alpaca-py",
    "man-group/ArcticDB",
    "UFund-Me/Qbot",
    "bbfamily/abu",
    "je-suis-tm/quant-trading",
    "zvtvz/zvt",
    "letianzj/QuantResearch",
    "VivekPa/AIAlpha",
    "asavinov/intelligent-trading-bot",
    "Lumiwealth/lumibot",
    "coding-kitties/investing-algorithm-framework",
    "OpenByteInc/QuantDinger",
    "FinanceData/FinanceDataReader",
    "akfamily/akshare",
    "OpenSourceRisk/Engine",
    "freqtrade/freqtrade",
    "hummingbot/hummingbot",
    "OpenBB-finance/OpenBBTerminal",
    "tradingstrategy-ai/trade-executor",
    "microsoft/LightGBM",
    "dmlc/xgboost",
    "catboost/catboost",
    "scikit-learn/scikit-learn",
    "sktime/sktime",
    "unit8co/darts",
    "google/tf-quant-finance",
    "lballabio/QuantLib",
    "LLMQuant/quant-mind",
    "jugaad-py/jugaad-data",
    "TheAlgorithms/Python",
]

DEEP_REVIEWS: dict[str, tuple[str, str, str, str, int, int]] = {
    "OpenBB-finance/OpenBB": (
        "modular financial-data and research platform",
        "provider abstractions and broad market-data coverage",
        "large platform; avoid importing its application layer",
        "architecture influence only",
        22,
        1,
    ),
    "microsoft/qlib": (
        "ML-oriented quantitative research platform",
        "dataset, model, and workflow concepts",
        "substantial data-format and orchestration commitment",
        "research and model-registry influence",
        23,
        1,
    ),
    "polakowo/vectorbt": (
        "vectorized strategy research and parameter exploration",
        "fast wide parameter sweeps",
        "vectorization can obscure execution-time semantics",
        "optional research adapter after event-engine validation",
        21,
        3,
    ),
    "QuantConnect/Lean": (
        "full algorithmic trading and simulation engine",
        "mature event-driven execution model",
        "C#-centered infrastructure exceeds starter scope",
        "execution-assumption reference only",
        18,
        0,
    ),
    "AI4Finance-Foundation/FinRL": (
        "reinforcement-learning finance research",
        "formal environments and evaluation framing",
        "RL is high variance and unsuitable as an unvalidated default",
        "future challenger-research input",
        15,
        0,
    ),
    "AI4Finance-Foundation/FinGPT": (
        "financial language-model research",
        "finance-specific language evaluation ideas",
        "model weights and licensing/data scope require separate review",
        "agent-evaluation influence only",
        12,
        0,
    ),
    "AI4Finance-Foundation/FinRobot": (
        "financial-agent workflow framework",
        "tool-oriented analyst roles",
        "LLM output must remain non-decisioning here",
        "agent boundaries influence only",
        14,
        0,
    ),
    "TauricResearch/TradingAgents": (
        "multi-agent investment research and debate",
        "bull/bear/risk committee decomposition",
        "agent consensus is not predictive evidence",
        "agent taxonomy influence only",
        16,
        0,
    ),
    "dgunning/edgartools": (
        "typed SEC EDGAR filings toolkit",
        "filing, XBRL, insider, and 13F coverage",
        "filing availability timestamps still need pipeline validation",
        "future isolated SEC adapter candidate",
        24,
        2,
    ),
    "JerBouma/FinanceToolkit": (
        "transparent financial statement analysis toolkit",
        "fundamental-ratio coverage",
        "provider/data provenance needs checking per endpoint",
        "future fundamental adapter candidate",
        21,
        2,
    ),
    "ranaroussi/quantstats": (
        "portfolio and strategy performance analytics",
        "recognizable reporting metric conventions",
        "metric definitions must be independently regression-tested",
        "metric naming and comparison influence",
        22,
        2,
    ),
    "PyPortfolio/PyPortfolioOpt": (
        "portfolio optimization library",
        "well-known constrained allocation methods",
        "estimation error can dominate optimized weights",
        "future portfolio optimizer candidate",
        22,
        2,
    ),
    "dcajasn/Riskfolio-Lib": (
        "risk-aware portfolio optimization",
        "broad risk-measure support",
        "advanced optimization adds dependency and model risk",
        "future constrained-allocation candidate",
        19,
        1,
    ),
    "quantopian/alphalens": (
        "factor-return analysis",
        "factor diagnostics and turnover framing",
        "project activity and dependency fit need review",
        "factor-evaluation influence only",
        18,
        0,
    ),
    "skfolio/skfolio": (
        "scikit-learn-oriented portfolio optimization",
        "estimator-style integration",
        "portfolio layer belongs after signal validation",
        "future portfolio candidate",
        20,
        2,
    ),
    "edtechre/pybroker": (
        "Python algorithmic backtesting framework",
        "ML-aware strategy interfaces",
        "must compare timestamp and cost semantics carefully",
        "backtest API comparison input",
        17,
        0,
    ),
    "stefan-jansen/machine-learning-for-trading": (
        "ML-for-markets educational and research material",
        "broad reproducible research examples",
        "book companion, not a production dependency",
        "methodology influence only",
        18,
        0,
    ),
    "TA-Lib/ta-lib-python": (
        "technical-indicator Python bindings",
        "battle-tested indicator breadth",
        "compiled dependency and indicator proliferation risk",
        "optional indicator adapter candidate",
        20,
        3,
    ),
    "ranaroussi/yfinance": (
        "unofficial market-data convenience client",
        "low-friction daily OHLCV retrieval",
        "not an authoritative point-in-time or licensed institutional feed",
        "starter provider only",
        22,
        4,
    ),
    "rsheftel/pandas_market_calendars": (
        "exchange trading calendars",
        "NYSE session and holiday correctness",
        "calendar package does not validate data availability",
        "incorporated runtime dependency",
        24,
        4,
    ),
    "alpacahq/alpaca-py": (
        "official Alpaca Python SDK",
        "paper-account and market-data integration path",
        "broker integration is out of scope for this research-only phase",
        "future paper-broker adapter candidate",
        20,
        3,
    ),
    "man-group/ArcticDB": (
        "versioned dataframe database",
        "research-data versioning ideas",
        "operational complexity is excessive for the local starter",
        "future storage architecture reference",
        16,
        0,
    ),
    "mementum/backtrader": (
        "event-driven Python backtester",
        "familiar strategy interface",
        "GPL-3.0 is incompatible with this MIT integration plan",
        "excluded; methodology comparison only",
        12,
        0,
    ),
    "kernc/backtesting.py": (
        "compact strategy backtesting library",
        "approachable strategy prototyping",
        "AGPL-3.0 prevents direct dependency in this plan",
        "excluded; methodology comparison only",
        10,
        0,
    ),
    "freqtrade/freqtrade": (
        "full crypto trading bot",
        "operational-risk and configuration patterns",
        "crypto focus and GPL-3.0 are wrong for US-equity research",
        "excluded; operations reference only",
        4,
        0,
    ),
    "Lumiwealth/lumibot": (
        "multi-asset backtesting and broker framework",
        "paper-trading lifecycle concepts",
        "GPL-3.0 and broker scope conflict with current phase",
        "excluded; lifecycle influence only",
        10,
        0,
    ),
    "OpenBB-finance/OpenBBTerminal": (
        "legacy terminal application",
        "historical product breadth",
        "superseded application; not selected as a dependency",
        "excluded; historical comparison only",
        2,
        0,
    ),
    "microsoft/LightGBM": (
        "gradient-boosted tree implementation",
        "strong tabular challenger-model baseline",
        "may overfit temporal data without strict walk-forward control",
        "future optional challenger dependency",
        20,
        3,
    ),
    "dmlc/xgboost": (
        "gradient-boosted tree implementation",
        "well-established nonlinear baseline",
        "complexity is not proof of trading edge",
        "future optional challenger dependency",
        19,
        3,
    ),
    "scikit-learn/scikit-learn": (
        "general machine-learning toolkit",
        "reproducible simple-model baselines and calibration",
        "not finance-specific; data splitting remains this project's responsibility",
        "incorporated runtime dependency",
        24,
        4,
    ),
}


def gh_json(endpoint: str) -> Any:
    completed = subprocess.run(
        ["gh", "api", "-H", "X-GitHub-Api-Version: 2022-11-28", endpoint],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(completed.stdout)


def gh_include(endpoint: str) -> str:
    completed = subprocess.run(
        ["gh", "api", "--include", "-H", "X-GitHub-Api-Version: 2022-11-28", endpoint],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def contributor_count(repository: str) -> str:
    try:
        response = gh_include(f"repos/{repository}/contributors?per_page=1&anon=true")
    except subprocess.CalledProcessError:
        return "Unavailable"
    last_page = re.search(r"[?&]page=(\d+)[^>]*>; rel=\"last\"", response)
    if last_page:
        return last_page.group(1)
    return "1" if "[" in response else "0"


def release_and_ci(repository: str) -> tuple[str, str]:
    release = "None"
    try:
        releases = gh_json(f"repos/{repository}/releases?per_page=1")
        if releases:
            release = str(releases[0].get("published_at") or releases[0].get("created_at"))[:10]
    except subprocess.CalledProcessError:
        release = "Unavailable"
    try:
        workflows = gh_json(f"repos/{repository}/contents/.github/workflows")
        ci = "Yes" if isinstance(workflows, list) and workflows else "No"
    except subprocess.CalledProcessError:
        ci = "No"
    return release, ci


def score(repository: dict[str, Any], ci: str, usefulness: int, integration: int) -> float:
    stars = int(repository["stargazers_count"])
    popularity = min(20.0, math.log10(stars + 1) / math.log10(50_001) * 20)
    pushed = datetime.fromisoformat(str(repository["pushed_at"]).replace("Z", "+00:00"))
    age_days = (datetime.now(UTC) - pushed).days
    activity = 15 if age_days <= 30 else 12 if age_days <= 90 else 8 if age_days <= 365 else 3
    maintenance = 10 if not repository["archived"] else 0
    documentation = min(10.0, int(repository["size"]) / 20_000 * 10)
    testing = 10 if ci == "Yes" else 2
    license_name = (repository.get("license") or {}).get("spdx_id", "NOASSERTION")
    license_points = (
        10 if license_name in {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"} else 0
    )
    usefulness_points = min(usefulness, 20)
    return round(
        popularity
        + activity
        + maintenance
        + documentation
        + testing
        + license_points
        + usefulness_points
        + integration,
        1,
    )


def markdown_escape(value: str) -> str:
    return value.replace("|", "/").replace("\n", " ").strip()


def build_report() -> str:
    observations: list[dict[str, Any]] = []
    for name in CANDIDATES:
        try:
            repository = gh_json(f"repos/{name}")
        except subprocess.CalledProcessError:
            observations.append({"name": name, "unavailable": True})
            continue
        release, ci = release_and_ci(name)
        purpose, strength, weakness, conclusion, usefulness, integration = DEEP_REVIEWS.get(
            name,
            (
                markdown_escape(
                    str(repository.get("description") or "Repository discovery candidate")
                ),
                "Public implementation and documentation were screened at metadata level.",
                "Not a selected component; no code was executed or incorporated.",
                "Discovery-only candidate.",
                5,
                0,
            ),
        )
        observations.append(
            {
                "name": name,
                "repository": repository,
                "contributors": contributor_count(name),
                "release": release,
                "ci": ci,
                "purpose": purpose,
                "strength": strength,
                "weakness": weakness,
                "conclusion": conclusion,
                "score": score(repository, ci, usefulness, integration),
                "deep": name in DEEP_REVIEWS,
            }
        )
    available = [entry for entry in observations if not entry.get("unavailable")]
    available.sort(key=lambda entry: float(entry["score"]), reverse=True)
    discovered_at = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# GitHub repository research",
        "",
        f"Live metadata snapshot: `{discovered_at}`. Generated with `scripts/refresh_github_research.py`.",
        "This discovery process reads public GitHub metadata only; no candidate code was executed or copied.",
        "",
        "## Method and score",
        "",
        "Score / 100 = popularity 20 + activity 15 + maintainability 10 + documentation 10 + "
        "CI/testing 10 + license compatibility 10 + US-equity swing usefulness 20 + integration ease 5.",
        "Popularity uses capped log stars; activity uses days since GitHub `pushed_at`; documentation "
        "uses repository size as a conservative metadata proxy; CI is detected from `.github/workflows`. "
        "`Latest activity` is a push-time proxy, not a claim that a commit is semantically meaningful. "
        "The 30 deep reviews add manual purpose, strength, weakness, swing suitability, and conclusion.",
        "",
        "Licenses marked GPL/AGPL receive no compatibility points and are not incorporated. `NOASSERTION` "
        "means GitHub did not identify an SPDX license, so it is not treated as reusable code.",
        "",
        f"## Candidate pool ({len(available)} available of {len(CANDIDATES)} discovered)",
        "",
        "| Rank | Repository | Stars | Forks | Contributors | Language | License | Latest activity | Latest release | Open issues | Docs proxy | CI | Score |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for rank, entry in enumerate(available, start=1):
        repository = entry["repository"]
        license_info = repository.get("license") or {}
        docs = "Large" if int(repository["size"]) >= 10_000 else "Basic"
        lines.append(
            "| {rank} | [{name}]({url}) | {stars} | {forks} | {contributors} | {language} | "
            "{license} | {pushed} | {release} | {issues} | {docs} | {ci} | {score:.1f} |".format(
                rank=rank,
                name=repository["full_name"],
                url=repository["html_url"],
                stars=repository["stargazers_count"],
                forks=repository["forks_count"],
                contributors=entry["contributors"],
                language=repository.get("language") or "—",
                license=license_info.get("spdx_id", "NOASSERTION"),
                pushed=str(repository["pushed_at"])[:10],
                release=entry["release"],
                issues=repository["open_issues_count"],
                docs=docs,
                ci=entry["ci"],
                score=float(entry["score"]),
            )
        )
    lines.extend(["", "## Deep evaluations (30)", ""])
    deep = [entry for entry in available if entry["deep"]]
    for entry in deep:
        repository = entry["repository"]
        lines.extend(
            [
                f"### {repository['full_name']} — {float(entry['score']):.1f}/100",
                "",
                f"- **Purpose:** {entry['purpose']}",
                f"- **Strongest feature:** {entry['strength']}",
                f"- **Weakness / overlap:** {entry['weakness']}",
                f"- **US-equity swing-trading decision:** {entry['conclusion']}",
                f"- **Live metadata:** {repository['stargazers_count']} stars, "
                f"{repository['forks_count']} forks, {entry['contributors']} contributors, "
                f"{repository.get('language') or 'no primary language'}, "
                f"license {(repository.get('license') or {}).get('spdx_id', 'NOASSERTION')}, "
                f"latest activity {str(repository['pushed_at'])[:10]}, release {entry['release']}, "
                f"CI {entry['ci']}, open issues {repository['open_issues_count']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Selected coherent stack",
            "",
            "The initial implementation uses original code plus pandas, NumPy, scikit-learn, Pydantic, "
            "PyYAML, requests, yfinance, and pandas-market-calendars. The selected projects influenced "
            "architecture, not copied source. The only incorporated package decisions at this stage are "
            "scikit-learn (baseline modelling) and pandas-market-calendars (NYSE timing); yfinance is a "
            "clearly labeled convenience provider. EDGAR, fundamental, portfolio, and vectorized research "
            "adapters remain separate future work until their exact versions, licenses, data terms, and tests "
            "are reviewed.",
            "",
            "## Exclusions and next review",
            "",
            "Crypto-first projects were not selected for the US-equity core. GPL/AGPL candidates are not "
            "integrated. Before adding any candidate, inspect the pinned release, dependency tree, license, "
            "data terms, API behavior, and point-in-time implications; then update `THIRD_PARTY_NOTICES.md`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "docs" / "github-repository-research.md"
    target.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
