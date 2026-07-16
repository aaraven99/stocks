"""Cross-table semantic validation. A finding is a non-zero CI exit."""
import json,math,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from storage.db import Database


def main():
 db=Database();findings=[]
 def finite(value):
  try:return math.isfinite(float(value))
  except (TypeError,ValueError):return False
 for row in db.rows('SELECT ticker,date,probability,target_before_stop,uncertainty FROM predictions'):
  for field in ('probability','target_before_stop'):
   if not finite(row[field]) or not 0<=float(row[field])<=1:findings.append(f"prediction_{field}_out_of_range:{row['ticker']}:{row['date']}")
  if not finite(row['uncertainty']) or float(row['uncertainty'])<0:findings.append(f"prediction_uncertainty_invalid:{row['ticker']}:{row['date']}")
 required_mc={'expected_return','var_95','cvar_95','p_stop_hit_first','p_target_before_stop','p_plus_3_5d','p_plus_5_10d','p_plus_10_20d'}
 for row in db.rows('SELECT ticker,date,metrics_json FROM montecarlo_metrics'):
  try:metrics=json.loads(row['metrics_json'])
  except (TypeError,json.JSONDecodeError):findings.append(f"mc_json_invalid:{row['ticker']}:{row['date']}");continue
  missing=required_mc-set(metrics)
  if missing:findings.append(f"mc_fields_missing:{row['ticker']}:{sorted(missing)}")
  for key in ('p_stop_hit_first','p_target_before_stop','p_plus_3_5d','p_plus_5_10d','p_plus_10_20d'):
   if key in metrics and (not finite(metrics[key]) or not 0<=float(metrics[key])<=1):findings.append(f'mc_probability_invalid:{row["ticker"]}:{key}')
 if db.rows('SELECT 1 FROM rankings WHERE score<0 OR score>1 LIMIT 1'):findings.append('ranking_score_out_of_range')
 if db.rows('SELECT 1 FROM trade_plans WHERE stop>=entry OR target<=entry OR shares<0 LIMIT 1'):findings.append('trade_plan_geometry_invalid')
 if db.rows('SELECT 1 FROM allocations WHERE weight<0 OR weight>0.1000001 LIMIT 1'):findings.append('allocation_name_cap_breached')
 if db.rows('SELECT 1 FROM calibration_metrics WHERE brier<0 OR brier>1 OR logloss<0 LIMIT 1'):findings.append('calibration_metric_invalid')
 test_mode=os.getenv('TEST_MODE','').lower() in ('1','true','yes') or 'test_mode' in str(db.path).lower()
 if db.rows("SELECT 1 FROM raw_prices WHERE source LIKE '%synthetic%' OR source LIKE '%fallback%' LIMIT 1") and not test_mode:findings.append('ambiguous_or_synthetic_data_in_production_database')
 orphan=db.rows("SELECT DISTINCT j.key feature FROM features f,json_each(f.values_json) j LEFT JOIN feature_definitions d ON d.name=j.key WHERE d.name IS NULL LIMIT 20")
 if orphan:findings.append('features_without_definitions:'+','.join(row['feature'] for row in orphan))
 latest_leakage=db.rows('SELECT passed,findings_json FROM leakage_detection ORDER BY date DESC LIMIT 1')
 if latest_leakage and not latest_leakage[0]['passed']:findings.append('latest_leakage_check_failed')
 result={'ok':not findings,'findings':findings,'database':str(db.path),'integrity':db.integrity()};db.log('validation','semantic validation completed',result);print(json.dumps(result,indent=2));return 0 if result['ok'] else 2


if __name__=='__main__':raise SystemExit(main())
