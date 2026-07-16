"""Backfill real historical OHLCV, causal features, labels, and MC validation rows."""
import argparse,json,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from core.data_quality import score_ohlcv
from core.time_utils import now_local
from data_sources.ticker_universe_source import FALLBACK
from data_sources.yfinance_source import fetch_ohlcv_batch
from features.feature_engineering import make_features,feature_definitions
from models.inference import predict
from montecarlo.montecarlo_engine import MonteCarloEngine
from storage.db import Database
from storage.feature_store import FeatureStore
from storage.artifacts import write_json


def _date(value): return value.date().isoformat() if hasattr(value,'date') else str(value)[:10]


def _cached_frames(db,tickers):
 """Load previously verified provider bars when the network provider is unavailable."""
 placeholders=','.join('?' for _ in tickers)
 rows=db.rows(f"SELECT ticker,date,open,high,low,close,volume FROM raw_prices WHERE source LIKE '%backfill' AND ticker IN ({placeholders}) ORDER BY ticker,date",list(tickers)) if tickers else []
 grouped={}
 for row in rows:
  grouped.setdefault(row['ticker'],[]).append(row)
 frames={}
 for ticker,values in grouped.items():
  frame=pd.DataFrame(values).set_index('date')
  frame.index=pd.to_datetime(frame.index)
  frame=frame.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
  if len(frame)>=60: frames[ticker]=frame[['Open','High','Low','Close','Volume']].astype(float)
 return frames


def _store_history(db,ticker,frame,quality):
 raw=[];clean=[];ingested=now_local().isoformat();quality_json=json.dumps(quality);source=str(frame.attrs.get('source','yfinance'))
 source=source if source.endswith('backfill') else source+'_backfill'
 for idx,row in frame.iterrows():
  day=_date(idx)
  values=(ticker,day,float(row.Open),float(row.High),float(row.Low),float(row.Close),float(row.Volume));raw.append((*values,source,ingested));clean.append((*values,quality['data_confidence'],quality_json))
 with db.connect() as connection:
  connection.executemany('INSERT INTO raw_prices(ticker,date,open,high,low,close,volume,source,ingested_at) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(ticker,date,source) DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,volume=excluded.volume,ingested_at=excluded.ingested_at',raw)
  connection.executemany('INSERT INTO cleaned_prices(ticker,date,open,high,low,close,volume,data_confidence,quality_json) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(ticker,date) DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,volume=excluded.volume,data_confidence=excluded.data_confidence,quality_json=excluded.quality_json',clean)


def backfill(max_tickers=10,period='5y',step=5,max_samples=80,mc_sims=1000,skip_mc=False,cached_only=False):
 db=Database();store=FeatureStore(db);store.register(feature_definitions());run_id='backfill_'+uuid.uuid4().hex[:12]
 # Use a stable liquid, cross-sector research panel. The alphabetical CSV head is
 # not a market-cap ranking and is unsuitable for model validation.
 tickers=FALLBACK[:max_tickers]
 frames,failures=({}, {}) if cached_only else fetch_ohlcv_batch(tickers,period=period,test_mode=False,batch_size=50)
 cached=_cached_frames(db,[ticker for ticker in tickers if ticker not in frames])
 for ticker,frame in cached.items(): frames[ticker]=frame; failures.pop(ticker,None)
 processed=0;labels=0;mc_validated=0
 for ticker,frame in frames.items():
  quality=score_ohlcv(frame);_store_history(db,ticker,frame,quality)
  starts=list(range(200,max(200,len(frame)-5),step))[-max_samples:]
  feature_rows=[];prediction_rows=[];label_rows=[];analog_rows=[];mc_rows=[];engine_rows=[]
  for end in starts:
   asof=_date(frame.index[end]);window=frame.iloc[:end+1];features=make_features(window,{'macro':{},'sentiment':{},'options':{},'event':{},'breadth':{}})
   computed_at=now_local().isoformat();source_ref=str(frame.attrs.get('source','yfinance'))+'_backfill';feature_rows.append({'ticker':ticker,'date':asof,'feature_version':'v1','values_json':features,'computed_at':computed_at,'source_refs':source_ref})
   prediction=predict(features);prediction_rows.append({'ticker':ticker,'date':asof,'model_id':prediction['model_id'],'probability':prediction['probability'],'expected_return':prediction['expected_return'],'expected_drawdown':prediction['expected_drawdown'],'target_before_stop':prediction['target_before_stop'],'uncertainty':prediction['uncertainty'],'explanation_json':{'explanation':prediction['explanation'],'confidence':prediction.get('confidence'),'conformal_interval':prediction.get('conformal_interval')}})
   future=frame.iloc[end+1:end+6];entry=float(window.Close.iloc[-1]);close_path=future.Close.to_numpy()/entry-1;low_path=future.Low.to_numpy()/entry-1;high_path=future.High.to_numpy()/entry-1;target_hits=np.flatnonzero(high_path>=.03);stop_hits=np.flatnonzero(low_path<=-.06);target_first=bool(len(target_hits) and (not len(stop_hits) or target_hits[0]<stop_hits[0]));positive=int(close_path[-1]>0);label={'realized_return':float(close_path[-1]),'positive':positive,'max_drawdown':float(low_path.min()),'target_before_stop':int(target_first)};label_rows.append({'ticker':ticker,'asof_date':asof,'horizon_days':5,**label,'created_at':computed_at});analog_rows.append({'ticker':ticker,'date':asof,'feature_vector_json':features,'outcome_json':label})
   if not skip_mc:
    returns=window.Close.pct_change().dropna();mc=MonteCarloEngine(mc_sims,42+end).run(returns,'sideways_chop',entry);mc_rows.append({'ticker':ticker,'date':asof,'engine_weights_json':mc['weights'],'metrics_json':mc['metrics'],'distribution_json':mc['distribution'][-5000:]})
    for engine,probability in mc['metrics']['engine_probabilities'].items():engine_rows.append({'ticker':ticker,'asof_date':asof,'engine':engine,'predicted_probability':probability,'outcome':positive,'brier':(probability-positive)**2});mc_validated+=1
   processed+=1;labels+=1
  db.upsert_many('features',feature_rows,['ticker','date','feature_version']);db.upsert_many('predictions',prediction_rows,['ticker','date','model_id']);db.upsert_many('labels',label_rows,['ticker','asof_date','horizon_days']);db.upsert_many('analog_memory',analog_rows,['ticker','date'])
  if not skip_mc:db.upsert_many('montecarlo_metrics',mc_rows,['ticker','date']);db.upsert_many('mc_engine_validation',engine_rows,['ticker','asof_date','engine'])
 summary={'run_id':run_id,'period':period,'requested_tickers':len(tickers),'loaded_tickers':len(frames),'failed_tickers':failures,'feature_samples':processed,'labels_created':labels,'mc_validation_rows':mc_validated,'created_at':now_local().isoformat()};db.upsert('historical_backfills',{'run_id':run_id,'started_at':summary['created_at'],'finished_at':now_local().isoformat(),'period':period,'tickers_requested':len(tickers),'tickers_loaded':len(frames),'samples_created':processed,'labels_created':labels,'mc_validation_rows':mc_validated,'failures_json':failures},['run_id']);write_json('backfill_'+run_id+'.json',summary);print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--max-tickers',type=int,default=10);parser.add_argument('--period',default='5y');parser.add_argument('--step',type=int,default=5);parser.add_argument('--max-samples',type=int,default=80);parser.add_argument('--mc-sims',type=int,default=1000);parser.add_argument('--skip-mc',action='store_true');parser.add_argument('--cached-only',action='store_true',help='reuse verified persisted provider bars without network access');args=parser.parse_args();backfill(args.max_tickers,args.period,args.step,args.max_samples,args.mc_sims,args.skip_mc,args.cached_only)
