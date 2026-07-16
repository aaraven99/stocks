"""Operational database, schema, and artifact healthcheck for CI."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from storage.db import Database

REQUIRED_TABLES={'raw_prices','cleaned_prices','feature_definitions','features','predictions','montecarlo_metrics','rankings','allocations','trade_plans','regimes','paper_trades','positions','paper_positions_daily','equity_curve','models','calibration_metrics','drift_metrics','experiments','champion_challenger','agent_outputs','debate_rounds','agents_transcripts','stress_outputs','journal_logs','portfolio_analytics','feature_stability','leakage_detection','probability_checks','kill_switch_history','research_lab','signal_discovery','reproducibility_manifests','model_risk_decisions','pipeline_runs','labels','analog_memory','training_runs','universe_health','mc_engine_validation','mc_validation_metrics','backtest_runs','backtest_trades','data_quarantine'}


def main():
 db=Database();names={row['name'] for row in db.rows("SELECT name FROM sqlite_master WHERE type='table'")};missing=sorted(REQUIRED_TABLES-names);foreign_key_errors=db.rows('PRAGMA foreign_key_check');running=db.rows("SELECT run_id FROM pipeline_runs WHERE status='running'");latest=db.rows('SELECT run_id,status,errors_json FROM pipeline_runs ORDER BY started_at DESC LIMIT 1');artifacts=Path(db.path).parent
 checks={'integrity_ok':db.integrity()=='ok','required_tables_present':not missing,'foreign_keys_ok':not foreign_key_errors,'no_stale_running_pipeline':not running,'database_nonempty':Path(db.path).exists() and Path(db.path).stat().st_size>0,'daily_artifacts_present':all((artifacts/name).exists() for name in ('daily_report.html','daily_report.md','manifest.json')) if latest else True}
 result={'ok':all(checks.values()),'checks':checks,'missing_tables':missing,'running_pipeline_ids':[row['run_id'] for row in running],'latest_pipeline':latest[0] if latest else None,'database':str(db.path),'table_count':len(names)};print(json.dumps(result,indent=2));return 0 if result['ok'] else 2


if __name__=='__main__':raise SystemExit(main())
