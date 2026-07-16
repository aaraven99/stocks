"""Run and persist an honest expanding-window backtest from matured real-data labels."""
import argparse,json,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from backtesting.backtester import walk_forward
from data_sources.ticker_universe_source import FALLBACK
from storage.artifacts import write_json,write_text
from storage.db import Database
import yaml


def _now():return datetime.now(timezone.utc).isoformat()


def _load_samples(db):
 placeholders=','.join('?' for _ in FALLBACK)
 rows=db.rows(f"SELECT f.ticker,f.date,f.values_json,l.realized_return,l.positive,(SELECT cp.date FROM cleaned_prices cp WHERE cp.ticker=l.ticker AND cp.date>l.asof_date ORDER BY cp.date LIMIT 1 OFFSET 4) AS exit_date FROM features f JOIN labels l ON l.ticker=f.ticker AND l.asof_date=f.date AND l.horizon_days=5 WHERE f.source_refs LIKE '%backfill%' AND ABS(l.realized_return)<=0.50 AND f.ticker IN ({placeholders}) ORDER BY f.date,f.ticker",FALLBACK)
 return [{'ticker':row['ticker'],'date':row['date'],'exit_date':row['exit_date'],'features':json.loads(row['values_json']),'realized_return':float(row['realized_return']),'outcome':int(row['positive'])} for row in rows if row['exit_date']]


def main(min_win_rate=None,max_days_per_trade=None,enforce=False,model_family=None,probability_floor=None,embargo_dates=None):
 try: configured=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/config.yaml').read_text()).get('backtest',{})
 except Exception: configured={}
 min_win_rate=float(configured.get('min_win_rate',.65) if min_win_rate is None else min_win_rate);max_days_per_trade=float(configured.get('max_days_per_trade',2.0) if max_days_per_trade is None else max_days_per_trade);model_family=str(configured.get('model_family','xgboost') if model_family is None else model_family);probability_floor=float(configured.get('probability_floor',.78) if probability_floor is None else probability_floor);embargo_dates=int(configured.get('embargo_dates',5) if embargo_dates is None else embargo_dates)
 db=Database();run_id='backtest_'+uuid.uuid4().hex[:12];started=_now();config={'min_train':int(configured.get('min_train',60)),'embargo_dates':int(embargo_dates),'max_trades_per_date':int(configured.get('max_trades_per_date',2)),'probability_floor':float(probability_floor),'model_family':model_family,'cost_bps':float(configured.get('cost_bps',30.0)),'notional_fraction_per_trade':.10,'data':'matured_provider_backfill_only_abs_return_lte_50pct'}
 samples=_load_samples(db)
 if len(samples)<80:
  result={'metrics':{'trades':0,'trade_win_rate':0.0,'business_days_per_trade':float('inf')},'trades':[],'diagnostics':[],'error':f'insufficient_valid_samples:{len(samples)}'}
 else:result=walk_forward(samples,**{key:config[key] for key in ('min_train','embargo_dates','max_trades_per_date','probability_floor','cost_bps','model_family')})
 metrics=result['metrics'];acceptance={'win_rate_target':float(min_win_rate),'trade_frequency_target_business_days':float(max_days_per_trade),'win_rate_passed':metrics.get('trade_win_rate',0)>=float(min_win_rate),'frequency_passed':metrics.get('business_days_per_trade',float('inf'))<=float(max_days_per_trade),'out_of_sample':True,'passed':False};acceptance['passed']=acceptance['win_rate_passed'] and acceptance['frequency_passed']
 status='passed' if acceptance['passed'] else 'failed_acceptance'
 db.upsert('backtest_runs',{'run_id':run_id,'started_at':started,'finished_at':_now(),'status':status,'config_json':config,'metrics_json':metrics,'acceptance_json':acceptance},['run_id'])
 for trade in result['trades']:db.upsert('backtest_trades',{'run_id':run_id,**trade},['run_id','ticker','signal_date'])
 payload={'run_id':run_id,'status':status,'sample_count':len(samples),'config':config,'metrics':metrics,'acceptance':acceptance,'diagnostics':result.get('diagnostics',[])};write_json('backtest_report.json',payload);write_text('backtest_report.md','# Leakage-Controlled Backtest\n\n```json\n'+json.dumps(payload,indent=2,default=str)+'\n```\n');print(json.dumps(payload,indent=2,default=str))
 return 0 if acceptance['passed'] or not enforce else 3


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--min-win-rate',type=float);parser.add_argument('--max-days-per-trade',type=float);parser.add_argument('--model-family',choices=['xgboost','logistic']);parser.add_argument('--probability-floor',type=float);parser.add_argument('--embargo-dates',type=int);parser.add_argument('--enforce-acceptance',action='store_true');args=parser.parse_args();raise SystemExit(main(args.min_win_rate,args.max_days_per_trade,args.enforce_acceptance,args.model_family,args.probability_floor,args.embargo_dates))
