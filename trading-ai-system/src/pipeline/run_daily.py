"""Daily causally ordered research run. All exceptions are recorded and scoped to a stage."""
import os,sys,time,json,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml,numpy as np
from core.utils import set_seed
from core.time_utils import trading_date,now_local
from core.logger import get_logger
from core.data_quality import score_ohlcv
from core.kill_switch import assess_kill_switch
from core.model_risk_management import mrm_decision
from core.reproducibility_manifest import build_manifest
from core.probability_consistency import validate_probability
from core.leakage_detection import check_feature_frame
from core.feature_stability import stability
from core.research_lab import discover_signal
from core.experiment_manager import ExperimentManager
from data_sources.ticker_universe_source import load_universe,load_file_fallback
from data_sources.yfinance_source import fetch_ohlcv,fetch_ohlcv_batch
from data_sources.cross_asset_source import cross_asset_snapshot
from data_sources.fred_source import macro_snapshot
from data_sources.news_sentiment_source import sentiment_snapshot
from data_sources.options_flow_source import options_flow_proxy
from data_sources.earnings_calendar_source import event_risk
from data_sources.market_breadth_source import breadth_snapshot
from data_sources.provider_compliance import provider_compliance
from features.feature_engineering import make_features,feature_definitions
from regime.regime_detection import detect
from regime.regime_forecast import forecast
from regime.regime_playbook import playbook
from montecarlo.montecarlo_engine import MonteCarloEngine
from montecarlo.montecarlo_validation import validate as validate_mc
from models.inference import predict
from models.drift_detection import detect as detect_drift
from models.calibration import brier,calibration_curve
from models.analog_memory import store as store_analog,nearest as nearest_analogs,outcome_prior
from models.strategy_archetypes import evaluate as evaluate_archetypes
from models.online_learning import update_weight
from portfolio.ranking_engine import rank
from portfolio.allocation_engine import allocate
from portfolio.trade_plan import make_plan
from portfolio.bet_clustering import cluster_metrics
from portfolio.risk_of_ruin import probability as ruin_probability
from portfolio.factor_exposure import aggregate as aggregate_factor_exposure
from models.factor_model import exposures as factor_exposures
from stress_testing.scenario_engine import run as stress_run
from stress_testing.stress_metrics import summarize
from storage.db import Database
from storage.feature_store import FeatureStore
from storage.artifacts import write_json,write_text,ensure_artifacts
from paper_trading.paper_engine import PaperEngine
from paper_trading.paper_metrics import metrics as paper_metrics
from paper_trading.alpaca_broker import AlpacaPaperBroker
from agents.agent_manager import run_debate
from agents.llm_client import narrative
from reporting.html_builder import build as html_build
from reporting.markdown_report import build as md_build
from reporting.email_sender import send
from reporting.charts import equity_chart
from pipeline.stages import select_fast_universe,fast_filter,select_deep_tickers

ROOT=Path(__file__).resolve().parents[1]
def config(): return yaml.safe_load((ROOT/'config/config.yaml').read_text())
def sectors():
 import pandas as pd
 path=ROOT/'data/sector_map.csv'
 return dict(zip(pd.read_csv(path).ticker,pd.read_csv(path).sector)) if path.exists() else {}
def _fast_score(df):
 close=df.Close; ret5=float(close.pct_change(5).iloc[-1]);ret20=float(close.pct_change(20).iloc[-1]);vol=float(close.pct_change().tail(20).std())
 volume=float(df.Volume.iloc[-1]/max(df.Volume.tail(20).mean(),1));return ret5*.35+ret20*.50-vol*.25+min(volume,3)*.01
def _persist_prices(db,ticker,df,date,quality):
 for idx,row in df.tail(3).iterrows():
  db.upsert('raw_prices',{'ticker':ticker,'date':str(idx.date()),'open':float(row.Open),'high':float(row.High),'low':float(row.Low),'close':float(row.Close),'volume':float(row.Volume),'source':'yfinance_or_synthetic_test','ingested_at':now_local().isoformat()},['ticker','date','source'])
 row=df.iloc[-1];db.upsert('cleaned_prices',{'ticker':ticker,'date':date,'open':float(row.Open),'high':float(row.High),'low':float(row.Low),'close':float(row.Close),'volume':float(row.Volume),'data_confidence':quality['data_confidence'],'quality_json':quality},['ticker','date'])
def _mature_labels(db,date):
 rows=db.rows("SELECT p.ticker,p.date,p.probability,c.close entry_close FROM predictions p JOIN cleaned_prices c ON c.ticker=p.ticker AND c.date=p.date WHERE p.date < ?",(date,))
 for x in rows:
  later=db.rows('SELECT close FROM cleaned_prices WHERE ticker=? AND date>? ORDER BY date LIMIT 5',(x['ticker'],x['date']))
  if len(later)==5:
   path=np.array([row['close']/x['entry_close']-1 for row in later]);ret=path[-1];label={'realized_return':float(ret),'positive':int(ret>0),'max_drawdown':float(path.min()),'target_before_stop':int(path.max()>=.03 and (path.min()>-.06 or np.argmax(path>=.03)<np.argmax(path<=-.06)))};db.upsert('labels',{'ticker':x['ticker'],'asof_date':x['date'],'horizon_days':5,**label,'created_at':now_local().isoformat()},['ticker','asof_date','horizon_days']);memory=db.rows('SELECT feature_vector_json FROM analog_memory WHERE ticker=? AND date=?',(x['ticker'],x['date']))
   if memory: db.upsert('analog_memory',{'ticker':x['ticker'],'date':x['date'],'feature_vector_json':memory[0]['feature_vector_json'],'outcome_json':label},['ticker','date'])
   mc_rows=db.rows('SELECT metrics_json FROM montecarlo_metrics WHERE ticker=? AND date=?',(x['ticker'],x['date']))
   if mc_rows:
    for engine,predicted in json.loads(mc_rows[0]['metrics_json']).get('engine_probabilities',{}).items():db.upsert('mc_engine_validation',{'ticker':x['ticker'],'asof_date':x['date'],'engine':engine,'predicted_probability':predicted,'outcome':label['positive'],'brier':(predicted-label['positive'])**2},['ticker','asof_date','engine'])
def main(test_mode=False,send_email=False):
 cfg=config();cfg['system']['test_mode']=bool(test_mode or cfg['system'].get('test_mode'));os.environ['TEST_MODE']='1' if cfg['system']['test_mode'] else '0';set_seed(cfg['system']['seed']);db=Database();log=get_logger();date=trading_date();run_id=f'daily_{date}_{uuid.uuid4().hex[:8]}';timings={};errors=[];sector_map=sectors();broker=AlpacaPaperBroker(db);providers=provider_compliance()
 db.upsert('pipeline_runs',{'run_id':run_id,'date':date,'status':'running','started_at':now_local().isoformat(),'finished_at':None,'stage_timings_json':{},'errors_json':[]},['run_id'])
 def stage(name,fn,default=None):
  start=time.perf_counter()
  try:return fn()
  except Exception as exc:
   errors.append({'stage':name,'error':str(exc)});log.exception(name)
   if cfg['system'].get('fail_closed',True) and name in {'universe','batch_ingestion','failover_ingestion','broker_reconcile','regime'}: raise
   return default
  finally:timings[name]=round(time.perf_counter()-start,3)
 broker_state=stage('broker_reconcile',lambda:broker.reconcile(date),{'enabled':False}) if not cfg['system']['test_mode'] else {'enabled':False,'reason':'test_mode'};capital=float(broker_state.get('equity',os.getenv('STARTING_CAPITAL','100000')))
 universe,source=stage('universe',lambda:load_universe('full'),([], 'unavailable'))
 universe=select_fast_universe(universe,12 if cfg['system']['test_mode'] else cfg['universe']['fast_limit'])
 frames,failures=stage('batch_ingestion',lambda:fetch_ohlcv_batch(universe,test_mode=cfg['system']['test_mode']),({},{}))
 failure_rate=len(failures)/max(1,len(universe));failover='none'
 if failure_rate>.20:
  fallback_mode='backup' if failure_rate<=.50 else 'top500';fallback,fb_source=load_file_fallback(fallback_mode);fallback=fallback[:12] if cfg['system']['test_mode'] else fallback[:(2000 if fallback_mode=='backup' else 500)];frames,failures=stage('failover_ingestion',lambda:fetch_ohlcv_batch(fallback,test_mode=cfg['system']['test_mode']),({},{}));universe=fallback;source=f'{source}->{fb_source}';failover=fallback_mode
 if not cfg['system']['test_mode'] and len(frames)==0: raise RuntimeError('No usable market data after universe failover')
 db.upsert('universe_health',{'date':date,'requested':len(universe),'loaded':len(frames),'deep_analyzed':0,'failed':len(failures),'source':source,'failover_mode':failover,'details_json':{'failure_rate':failure_rate,'sample_failures':dict(list(failures.items())[:25])}},['date'])
 quality={ticker:score_ohlcv(df) for ticker,df in frames.items()}
 for ticker,df in frames.items():_persist_prices(db,ticker,df,date,quality[ticker])
 fast=fast_filter(frames,_fast_score)
 fast=[{**row,'quality':quality[row['ticker']]} for row in fast]
 deep_limit=min(cfg['universe']['deep_limit'],len(fast));deep_tickers=select_deep_tickers(fast,deep_limit)
 db.execute('UPDATE universe_health SET deep_analyzed=? WHERE date=?',(len(deep_tickers),date))
 cross=stage('cross_asset',lambda:cross_asset_snapshot(cfg['system']['test_mode']),{}) or {};breadth=breadth_snapshot(frames);spy=frames['SPY'] if 'SPY' in frames else stage('spy_proxy',lambda:fetch_ohlcv('SPY',test_mode=cfg['system']['test_mode']));vix=stage('vix_proxy',lambda:fetch_ohlcv('^VIX',test_mode=cfg['system']['test_mode']),spy); 
 if spy is None: raise RuntimeError('No valid market proxy available; entering safe failure state')
 regime=detect(spy,vix);regime.update(forecast(regime['regime'],regime['volatility'],regime['hmm'],regime['cluster']));db.upsert('regimes',{'date':date,'regime':regime['regime'],'confidence':regime['confidence'],'flip_bearish_5d':regime['flip_bearish_5d'],'vol_spike_5d':regime['vol_spike_5d'],'details_json':regime},['date']);play=playbook(regime['regime'])
 store=FeatureStore(db);store.register(feature_definitions());results=[];sims=min(cfg['montecarlo']['simulations'],4000) if cfg['system']['test_mode'] else cfg['montecarlo']['simulations']
 macro=stage('macro',lambda:macro_snapshot(cfg['system']['test_mode']),{}) or {};engine_history=db.rows('SELECT engine,AVG(brier) brier FROM mc_engine_validation GROUP BY engine');reliability={row['engine']:max(.05,1-row['brier']) for row in engine_history}
 for ticker in deep_tickers:
  df=frames[ticker];context={'macro':macro,'sentiment':sentiment_snapshot(ticker) if providers['news_sentiment']['decision_use'] else {},'options':options_flow_proxy(ticker) if providers['options_flow']['decision_use'] else {},'event':event_risk(ticker) if providers['earnings_calendar']['decision_use'] else {},'cross_asset':cross,'breadth':breadth};features=make_features(df,context);archetypes=evaluate_archetypes(features,regime['regime']);store.put(ticker,date,features);prediction=predict(features);mc=MonteCarloEngine(sims,cfg['system']['seed']+abs(hash(ticker))%10000).run(df.Close.pct_change().dropna(),regime['regime'],float(df.Close.iloc[-1]),reliability=reliability);stress=summarize(stress_run(ticker));checks=validate_probability(prediction);mc_check=validate_mc(mc['metrics'])
  prediction_governance={'explanation':prediction['explanation'],'confidence':prediction.get('confidence'),'conformal_interval':prediction.get('conformal_interval'),'analog_prior_pending':True}
  db.upsert('montecarlo_metrics',{'ticker':ticker,'date':date,'engine_weights_json':mc['weights'],'metrics_json':mc['metrics'],'distribution_json':mc['distribution'][-5000:]},['ticker','date']);db.upsert('predictions',{'ticker':ticker,'date':date,'model_id':prediction['model_id'],'probability':prediction['probability'],'expected_return':prediction['expected_return'],'expected_drawdown':prediction['expected_drawdown'],'target_before_stop':prediction['target_before_stop'],'uncertainty':prediction['uncertainty'],'explanation_json':prediction_governance},['ticker','date','model_id']);db.upsert('probability_checks',{'date':date,'ticker':ticker,'passed':int(checks['passed'] and mc_check['passed']),'findings_json':checks['findings']+mc_check['findings']},['date','ticker'])
  for scenario in stress_run(ticker):db.upsert('stress_outputs',{'ticker':ticker,'date':date,'scenario':scenario['scenario'],'pnl_pct':scenario['pnl_pct'],'penalty':scenario['penalty'],'details_json':scenario},['ticker','date','scenario']);db.upsert('scenarios',{'date':date,'scenario':scenario['scenario'],'details_json':scenario},['date','scenario'])
  if context['event'].get('event_risk'):db.upsert('event_risk_flags',{'ticker':ticker,'date':date,'flag':'earnings_provider','severity':context['event']['event_risk'],'details_json':context['event']},['ticker','date','flag'])
  analogs=nearest_analogs(db,features,date);analog_prior=outcome_prior(analogs);store_analog(db,ticker,date,features);prediction['analog_prior']=analog_prior;prediction_governance['analog_prior']=analog_prior
  for archetype,score in archetypes.items(): db.upsert('strategy_archetype_scores',{'ticker':ticker,'date':date,'archetype':archetype,'score':score,'details_json':{'regime':regime['regime']}},['ticker','date','archetype'])
  db.upsert('predictions',{'ticker':ticker,'date':date,'model_id':prediction['model_id'],'probability':prediction['probability'],'expected_return':prediction['expected_return'],'expected_drawdown':prediction['expected_drawdown'],'target_before_stop':prediction['target_before_stop'],'uncertainty':prediction['uncertainty'],'explanation_json':prediction_governance},['ticker','date','model_id']);results.append({'ticker':ticker,'price':float(df.Close.iloc[-1]),'sector':sector_map.get(ticker,'Unknown'),'returns':df.Close.pct_change().dropna().tail(60),'features':features,'quality':quality[ticker],'prediction':prediction,'mc':mc['metrics'],'stress_penalty':stress['penalty'],'analog_prior':analog_prior,'archetypes':archetypes})
 ranked=rank(results,regime['regime'],play)
 for row in ranked:db.upsert('rankings',{'ticker':row['ticker'],'date':date,'score':row['score'],'action':row['action'],'position_size':0,'regime':regime['regime'],'stress_penalty':row['stress_penalty'],'confidence':row['confidence'],'details_json':row},['ticker','date'])
 labels=_mature_labels(db,date);label_rows=db.rows('SELECT positive FROM labels WHERE asof_date < ?',(date,));pred_rows=db.rows('SELECT probability FROM predictions WHERE date < ?',(date,));observed=[r['positive'] for r in label_rows[-len(pred_rows):]];probs=[r['probability'] for r in pred_rows[-len(observed):]];cal_brier=brier(observed,probs) if observed and len(observed)==len(probs) else .25;cal_curve=calibration_curve(observed,probs) if observed and len(observed)==len(probs) else []
 drift=detect_drift([r['fast_score'] for r in fast],[_fast_score(frames[t]) for t in deep_tickers]) if deep_tickers else {'psi':1,'ks':1,'flag':'material'};drift['flag']='material' if drift['psi']>.45 else drift['flag'];db.upsert('drift_metrics',{'date':date,'model_id':'quant_ensemble_v1','psi':drift['psi'],'ks':drift['ks'],'flag':drift['flag'],'details_json':drift},['date','model_id']);db.upsert('calibration_metrics',{'date':date,'model_id':'quant_ensemble_v1','brier':cal_brier,'logloss':cal_brier,'curve_json':cal_curve},['date','model_id'])
 previous=db.rows('SELECT values_json FROM features WHERE date < ? ORDER BY date DESC LIMIT 500',(date,));current={key:[row['features'].get(key,0.) for row in results] for key in (results[0]['features'] if results else {})}
 if previous:
  prior=[json.loads(row['values_json']) for row in previous]
  for feature,values in current.items():
   ref=[item.get(feature,0.) for item in prior];stat=stability(ref,values);db.upsert('feature_stability',{'date':date,'feature':feature,'psi':stat['psi'],'stable':int(stat['stable']),'details_json':stat},['date','feature'])
 curve=db.rows('SELECT * FROM equity_curve ORDER BY date');dd=curve[-1]['drawdown'] if curve else 0;kill=assess_kill_switch(dd,drift['flag']=='material',db.integrity()=='ok',cfg['risk']['max_drawdown_kill']);mrm=mrm_decision(kill,cal_brier,drift['psi']);closed_before=db.rows("SELECT pnl,entry,shares FROM paper_trades WHERE status='closed' ORDER BY id DESC LIMIT 20");reward=float(np.mean([row['pnl']/max(row['entry']*row['shares'],1) for row in closed_before])) if closed_before else 0.;feedback_multiplier=update_weight(1.,reward);mrm['risk_multiplier']*=feedback_multiplier;mrm['paper_feedback_reward']=reward;mrm['paper_feedback_multiplier']=feedback_multiplier;db.upsert('kill_switch_history',{'date':date,'active':int(kill['active']),'reason':kill['reason'],'metrics_json':{'drawdown':dd,'drift':drift}},['date']);db.upsert('model_risk_decisions',{'date':date,'decision':mrm['decision'],'reason':mrm['reason'],'details_json':mrm},['date'])
 candidates=[x for x in ranked if x['action']=='LONG' and x['quality']['data_confidence']>=cfg['risk']['min_data_confidence']];alloc=allocate(candidates,capital,mrm['risk_multiplier']) if not kill['active'] else []
 for row in alloc:db.upsert('allocations',{'ticker':row['ticker'],'date':date,'weight':row['weight'],'sector':row['sector'],'risk_budget':row['weight'],'reason':'kelly_cvar_correlation_sector_optimizer'},['ticker','date'])
 plans=[dict(make_plan(x['ticker'],x['price'],x['weight'],capital),weight=x['weight']) for x in alloc]
 for p in plans:db.upsert('trade_plans',{'ticker':p['ticker'],'date':date,'entry':p['entry'],'stop':p['stop'],'target':p['target'],'shares':p['shares'],'action':p['action'],'transaction_cost':p['transaction_cost']},['ticker','date'])
 paper=PaperEngine(db,capital);paper.mark_and_close({x['ticker']:x['price'] for x in results},date);paper.open(plans,date);paper.snapshot(date);submissions=[stage('alpaca_order_'+plan['ticker'],lambda p=plan:broker.submit_long_bracket(date,p),{'submitted':False}) for plan in plans] if broker_state.get('enabled') else [];positions=db.rows('SELECT * FROM positions');trades=db.rows("SELECT * FROM paper_trades WHERE status='closed'");market_returns=spy.Close.pct_change().dropna().tail(60);factors={row['ticker']:factor_exposures(row['returns'].tail(60),market_returns) for row in alloc};analytics={'paper':paper_metrics(trades,db.rows('SELECT * FROM equity_curve ORDER BY date')),'clustering':cluster_metrics(positions),'risk_of_ruin':ruin_probability(.5,1.2,.01),'factor_exposure':aggregate_factor_exposure(alloc,factors),'broker':{'state':broker_state,'submissions':submissions}};db.upsert('portfolio_analytics',{'date':date,'metrics_json':analytics,'factor_json':analytics['factor_exposure'],'alpha_beta_json':analytics['factor_exposure'],'cost_json':{'planned_cost':sum(x['transaction_cost'] for x in plans)},'cluster_json':analytics['clustering'],'ruin_json':{'probability':analytics['risk_of_ruin']}},['date'])
 debate,transcript=run_debate(db,date,{'regime':regime['regime'],'risk':regime['flip_bearish_5d'],'top_candidates':[x['ticker'] for x in alloc]});llm=narrative(f"Narrate only these quantitative facts: regime={regime['regime']}; bear_flip={regime['flip_bearish_5d']:.3f}; candidates={[x['ticker'] for x in alloc]}; no investment advice.");db.upsert('agent_outputs',{'date':date,'agent':'LLMNarrative','payload_json':llm},['date','agent']);decision='CASH MODE / NO TRADE DAY' if kill['active'] or not plans or debate['action']=='CASH' else 'RANKED WATCHLIST'
 experiments=ExperimentManager(db)
 for signal in discover_signal(spy):
  experiment_id=f"signal_{signal['signal_name']}_v1";accepted=int(experiments.evaluate(experiment_id,signal['signal_name'],{'formula':signal['formula']},{'score':signal['score']}));db.upsert('signal_discovery',{'date':date,'signal_name':signal['signal_name'],'formula':signal['formula'],'score':signal['score'],'details_json':signal},['date','signal_name']);db.upsert('research_lab',{'date':date,'experiment_id':experiment_id,'signal_name':signal['signal_name'],'metrics_json':signal,'accepted':accepted},['date','experiment_id'])
 leakage=check_feature_frame(__import__('pandas').DataFrame([x['features'] for x in results]));db.upsert('leakage_detection',{'date':date,'passed':int(leakage['passed']),'findings_json':leakage['findings']},['date'])
 report={'date':date,'regime':regime['regime'],'regime_forecast':regime,'decision':decision,'picks':alloc[:12],'debate':debate,'narrative':llm,'kill_switch':kill,'universe_health':{'requested':len(universe),'loaded':len(frames),'deep_analyzed':len(deep_tickers),'failure_rate':failure_rate,'source':source,'failover':failover},'analytics':analytics,'calibration':{'brier':cal_brier,'curve':cal_curve,'drift':drift},'stress':db.rows('SELECT scenario,AVG(pnl_pct) avg_pnl,AVG(penalty) avg_penalty FROM stress_outputs WHERE date=? GROUP BY scenario',(date,)),'equity_chart_artifact':'equity_curve_chart.html','provider_compliance':providers};ensure_artifacts();equity_chart(db.rows('SELECT * FROM equity_curve ORDER BY date')).write_html(ensure_artifacts()/'equity_curve_chart.html',include_plotlyjs='cdn');html=html_build(report);write_text('daily_report.html',html);write_text('daily_report.md',md_build(report));manifest=build_manifest(cfg,{'run_id':run_id,'universe_source':source,'loaded':len(frames),'deep_tickers':deep_tickers,'timings':timings});write_json('manifest.json',manifest);db.upsert('reproducibility_manifests',{'run_id':manifest['run_id'],'date':date,'manifest_json':manifest,'created_at':manifest['created_at']},['run_id'])
 if send_email:stage('email',lambda:send(html,'Systematic Swing Research '+date))
 status='success_with_degradation' if errors else 'success';db.upsert('pipeline_runs',{'run_id':run_id,'date':date,'status':status,'started_at':now_local().isoformat(),'finished_at':now_local().isoformat(),'stage_timings_json':timings,'errors_json':errors},['run_id']);log_dir=ROOT.parent/'research_logs';log_dir.mkdir(exist_ok=True);(log_dir/f'{run_id}.json').write_text(json.dumps({'run_id':run_id,'status':status,'stage_timings':timings,'errors':errors},indent=2));return report
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--test-mode',action='store_true');parser.add_argument('--send-email',action='store_true');args=parser.parse_args();main(args.test_mode,args.send_email)
