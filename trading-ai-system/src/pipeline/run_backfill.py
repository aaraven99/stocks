"""Backfill real historical OHLCV, causal features, labels, and MC validation rows."""
import argparse,json,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.data_quality import score_ohlcv
from core.time_utils import now_local
from data_sources.ticker_universe_source import load_file_fallback
from data_sources.yfinance_source import fetch_ohlcv_batch
from features.feature_engineering import make_features,feature_definitions
from models.inference import predict
from models.analog_memory import store as store_analog
from montecarlo.montecarlo_engine import MonteCarloEngine
from storage.db import Database
from storage.feature_store import FeatureStore
from storage.artifacts import write_json


def _date(value): return value.date().isoformat() if hasattr(value,'date') else str(value)[:10]


def _store_history(db,ticker,frame,quality):
 for idx,row in frame.iterrows():
  day=_date(idx)
  db.upsert('raw_prices',{'ticker':ticker,'date':day,'open':float(row.Open),'high':float(row.High),'low':float(row.Low),'close':float(row.Close),'volume':float(row.Volume),'source':'yfinance_backfill','ingested_at':now_local().isoformat()},['ticker','date','source'])
  db.upsert('cleaned_prices',{'ticker':ticker,'date':day,'open':float(row.Open),'high':float(row.High),'low':float(row.Low),'close':float(row.Close),'volume':float(row.Volume),'data_confidence':quality['data_confidence'],'quality_json':quality},['ticker','date'])


def backfill(max_tickers=10,period='5y',step=5,max_samples=80,mc_sims=1000):
 db=Database();store=FeatureStore(db);store.register(feature_definitions());run_id='backfill_'+uuid.uuid4().hex[:12]
 universe,_=load_file_fallback('top500');tickers=universe[:max_tickers]
 frames,failures=fetch_ohlcv_batch(tickers,period=period,test_mode=False,batch_size=50)
 processed=0;labels=0;mc_validated=0
 for ticker,frame in frames.items():
  quality=score_ohlcv(frame);_store_history(db,ticker,frame,quality)
  starts=list(range(200,max(200,len(frame)-5),step))[-max_samples:]
  for end in starts:
   asof=_date(frame.index[end]);window=frame.iloc[:end+1];features=make_features(window,{'macro':{},'sentiment':{},'options':{},'event':{},'breadth':{}});store.put(ticker,asof,features,sources='yfinance_backfill');prediction=predict(features);returns=window.Close.pct_change().dropna();mc=MonteCarloEngine(mc_sims,42+end).run(returns,'sideways_chop',float(window.Close.iloc[-1]));db.upsert('predictions',{'ticker':ticker,'date':asof,'model_id':prediction['model_id'],'probability':prediction['probability'],'expected_return':prediction['expected_return'],'expected_drawdown':prediction['expected_drawdown'],'target_before_stop':prediction['target_before_stop'],'uncertainty':prediction['uncertainty'],'explanation_json':{'explanation':prediction['explanation'],'confidence':prediction.get('confidence'),'conformal_interval':prediction.get('conformal_interval')}},['ticker','date','model_id']);db.upsert('montecarlo_metrics',{'ticker':ticker,'date':asof,'engine_weights_json':mc['weights'],'metrics_json':mc['metrics'],'distribution_json':mc['distribution'][-5000:]},['ticker','date'])
   future=frame.iloc[end+1:end+6].Close.to_numpy();entry=float(window.Close.iloc[-1]);path=future/entry-1;positive=int(path[-1]>0);label={'realized_return':float(path[-1]),'positive':positive,'max_drawdown':float(path.min()),'target_before_stop':int(path.max()>=.03 and (path.min()>-.06 or np.argmax(path>=.03)<np.argmax(path<=-.06)))};db.upsert('labels',{'ticker':ticker,'asof_date':asof,'horizon_days':5,**label,'created_at':now_local().isoformat()},['ticker','asof_date','horizon_days']);store_analog(db,ticker,asof,features,label)
   for engine,probability in mc['metrics']['engine_probabilities'].items():db.upsert('mc_engine_validation',{'ticker':ticker,'asof_date':asof,'engine':engine,'predicted_probability':probability,'outcome':positive,'brier':(probability-positive)**2},['ticker','asof_date','engine'])
   processed+=1;labels+=1;mc_validated+=len(mc['metrics']['engine_probabilities'])
 summary={'run_id':run_id,'period':period,'requested_tickers':len(tickers),'loaded_tickers':len(frames),'failed_tickers':failures,'feature_samples':processed,'labels_created':labels,'mc_validation_rows':mc_validated,'created_at':now_local().isoformat()};db.upsert('historical_backfills',{'run_id':run_id,'started_at':summary['created_at'],'finished_at':now_local().isoformat(),'period':period,'tickers_requested':len(tickers),'tickers_loaded':len(frames),'samples_created':processed,'labels_created':labels,'mc_validation_rows':mc_validated,'failures_json':failures},['run_id']);write_json('backfill_'+run_id+'.json',summary);print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--max-tickers',type=int,default=10);parser.add_argument('--period',default='5y');parser.add_argument('--step',type=int,default=5);parser.add_argument('--max-samples',type=int,default=80);parser.add_argument('--mc-sims',type=int,default=1000);args=parser.parse_args();backfill(args.max_tickers,args.period,args.step,args.max_samples,args.mc_sims)
