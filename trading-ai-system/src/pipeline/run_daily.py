"""Daily causally ordered research run. All exceptions are recorded and scoped to a stage."""
import os,sys,time,json,uuid,hashlib
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
from portfolio.alpha_beta_decomposition import decompose as decompose_alpha_beta
from models.factor_model import exposures as factor_exposures
from stress_testing.scenario_engine import run as stress_run
from stress_testing.stress_metrics import summarize
from storage.db import Database
from storage.feature_store import FeatureStore
from storage.artifacts import write_json,write_text,ensure_artifacts
from paper_trading.paper_engine import PaperEngine
from paper_trading.paper_metrics import metrics as paper_metrics
from paper_trading.attribution import trade_attribution
from paper_trading.portfolio_attribution import summarize as summarize_attribution
from paper_trading.equity_curve import drawdown as recompute_drawdown
from paper_trading.alpaca_broker import AlpacaPaperBroker
from agents.agent_manager import run_debate
from agents.llm_client import narrative
from reporting.html_builder import build as html_build
from reporting.markdown_report import build as md_build
from reporting.email_sender import send
from reporting.charts import equity_chart,equity_chart_png
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
 source=str(df.attrs.get('source','synthetic_test' if os.getenv('TEST_MODE')=='1' else 'yfinance'))
 for idx,row in df.tail(3).iterrows():
  db.upsert('raw_prices',{'ticker':ticker,'date':str(idx.date()),'open':float(row.Open),'high':float(row.High),'low':float(row.Low),'close':float(row.Close),'volume':float(row.Volume),'source':source,'ingested_at':now_local().isoformat()},['ticker','date','source'])
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
 cfg=config();cfg['system']['test_mode']=bool(test_mode or cfg['system'].get('test_mode'));requested_mode=os.getenv('UNIVERSE_MODE','').strip().lower();cfg['universe']['mode']=requested_mode if requested_mode in {'full','backup','backup_2000','top','top500'} else cfg['universe'].get('mode','full');requested_sims=os.getenv('MONTE_CARLO_SIMS','').strip();cfg['montecarlo']['simulations']=int(requested_sims) if requested_sims.isdigit() and int(requested_sims)>=1000 else int(cfg['montecarlo']['simulations']);os.environ['TEST_MODE']='1' if cfg['system']['test_mode'] else '0';set_seed(cfg['system']['seed']);db=Database();log=get_logger();date=trading_date();run_id=f'daily_{date}_{uuid.uuid4().hex[:8]}';timings={};errors=[];sector_map=sectors();broker=AlpacaPaperBroker(db);providers=provider_compliance();started_at=now_local().isoformat()
 db.execute("UPDATE pipeline_runs SET status='abandoned',finished_at=? WHERE status='running' AND julianday(started_at)<julianday('now','-6 hours')",(started_at,));db.upsert('pipeline_runs',{'run_id':run_id,'date':date,'status':'running','started_at':started_at,'finished_at':None,'stage_timings_json':{},'errors_json':[]},['run_id']);db.log('pipeline_start','daily pipeline started',{'run_id':run_id,'test_mode':cfg['system']['test_mode']})
 def stage(name,fn,default=None):
  start=time.perf_counter()
  try:return fn()
  except Exception as exc:
   errors.append({'stage':name,'error':str(exc)});db.log('stage_error',name,{'run_id':run_id,'error':str(exc)});log.exception(name)
   db.quarantine('pipeline_stage',f'{run_id}:{name}',str(exc),{'run_id':run_id,'stage':name})
   if cfg['system'].get('fail_closed',True) and name in {'universe','batch_ingestion','failover_ingestion','broker_reconcile','regime'}:
    db.upsert('pipeline_runs',{'run_id':run_id,'date':date,'status':'failed','started_at':started_at,'finished_at':now_local().isoformat(),'stage_timings_json':timings,'errors_json':errors},['run_id']);raise
   return default
  finally:
   timings[name]=round(time.perf_counter()-start,3);db.log('stage_timing',name,{'run_id':run_id,'seconds':timings[name]})
 broker_state=stage('broker_reconcile',lambda:broker.reconcile(date),{'enabled':False}) if not cfg['system']['test_mode'] else {'enabled':False,'reason':'test_mode'};capital=float(broker_state.get('equity',os.getenv('STARTING_CAPITAL','100000')))
 universe,source=stage('universe',lambda:load_universe(cfg['universe']['mode']),([], 'unavailable'))
 universe=select_fast_universe(universe,12 if cfg['system']['test_mode'] else cfg['universe']['fast_limit'])
 initial_requested=len(universe);initial_source=source
 frames,failures=stage('batch_ingestion',lambda:fetch_ohlcv_batch(universe,test_mode=cfg['system']['test_mode']),({},{}))
 initial_failure_rate=len(failures)/max(1,len(universe));failure_rate=initial_failure_rate;failover='none'
 for ticker,reason in failures.items(): db.quarantine('raw_prices',f'{ticker}:{date}','provider_ingestion_failure',{'error':reason,'run_id':run_id})
 if initial_failure_rate>.20:
  modes=['backup','top500'] if initial_failure_rate<=.50 else ['top500']
  for fallback_mode in modes:
   fallback,fb_source=load_file_fallback(fallback_mode);fallback=fallback[:12] if cfg['system']['test_mode'] else fallback[:(2000 if fallback_mode=='backup' else 500)]
   candidate_frames,candidate_failures=stage('failover_ingestion',lambda:fetch_ohlcv_batch(fallback,test_mode=cfg['system']['test_mode']),({},{}))
   candidate_rate=len(candidate_failures)/max(1,len(fallback))
   for ticker,reason in candidate_failures.items(): db.quarantine('raw_prices',f'{ticker}:{date}',f'failover_{fallback_mode}_ingestion_failure',{'error':reason,'run_id':run_id})
   frames,failures,universe,source,failover,failure_rate=candidate_frames,candidate_failures,fallback,f'{source}->{fb_source}',fallback_mode,candidate_rate
   if candidate_rate<=.50 or fallback_mode=='top500':break
 if not cfg['system']['test_mode'] and len(frames)==0: raise RuntimeError('No usable market data after universe failover')
 db.upsert('universe_health',{'date':date,'requested':initial_requested,'loaded':len(frames),'deep_analyzed':0,'failed':len(failures),'source':source,'failover_mode':failover,'details_json':{'initial_source':initial_source,'initial_failure_rate':initial_failure_rate,'final_failure_rate':failure_rate,'active_universe_size':len(universe),'sample_failures':dict(list(failures.items())[:25])}},['date'])
 quality={ticker:score_ohlcv(df) for ticker,df in frames.items()}
 for ticker,metrics in quality.items():
  if metrics.get('invalid_ohlc_score',0)>0: db.quarantine('cleaned_prices',f'{ticker}:{date}','invalid_ohlcv',metrics)
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
  df=frames[ticker];context={'macro':macro,'sentiment':sentiment_snapshot(ticker) if providers['news_sentiment']['decision_use'] else {},'options':options_flow_proxy(ticker) if providers['options_flow']['decision_use'] else {},'event':event_risk(ticker) if providers['earnings_calendar']['decision_use'] else {},'cross_asset':cross,'breadth':breadth};features=make_features(df,context);store.register(feature_definitions());archetypes=evaluate_archetypes(features,regime['regime']);store.put(ticker,date,features,sources='synthetic_test' if cfg['system']['test_mode'] else 'yfinance');prediction=predict(features);ticker_seed=int(hashlib.sha256(ticker.encode()).hexdigest()[:8],16);mc=MonteCarloEngine(sims,cfg['system']['seed']+ticker_seed%10000).run(df.Close.pct_change().dropna(),regime['regime'],float(df.Close.iloc[-1]),reliability=reliability);scenario_rows=stress_run(ticker,features=features);stress=summarize(scenario_rows);checks=validate_probability(prediction);mc_check=validate_mc(mc['metrics'])
  prediction_governance={'explanation':prediction['explanation'],'confidence':prediction.get('confidence'),'conformal_interval':prediction.get('conformal_interval'),'analog_prior_pending':True}
  db.upsert('montecarlo_metrics',{'ticker':ticker,'date':date,'engine_weights_json':mc['weights'],'metrics_json':mc['metrics'],'distribution_json':mc['distribution'][-5000:]},['ticker','date']);db.upsert('predictions',{'ticker':ticker,'date':date,'model_id':prediction['model_id'],'probability':prediction['probability'],'expected_return':prediction['expected_return'],'expected_drawdown':prediction['expected_drawdown'],'target_before_stop':prediction['target_before_stop'],'uncertainty':prediction['uncertainty'],'explanation_json':prediction_governance},['ticker','date','model_id']);db.upsert('probability_checks',{'date':date,'ticker':ticker,'passed':int(checks['passed'] and mc_check['passed']),'findings_json':checks['findings']+mc_check['findings']},['date','ticker'])
  for scenario in scenario_rows:db.upsert('stress_outputs',{'ticker':ticker,'date':date,'scenario':scenario['scenario'],'pnl_pct':scenario['pnl_pct'],'penalty':scenario['penalty'],'details_json':scenario},['ticker','date','scenario']);db.upsert('scenarios',{'date':date,'scenario':scenario['scenario'],'details_json':scenario},['date','scenario'])
  if context['event'].get('event_risk'):db.upsert('event_risk_flags',{'ticker':ticker,'date':date,'flag':'earnings_provider','severity':context['event']['event_risk'],'details_json':context['event']},['ticker','date','flag'])
  analogs=nearest_analogs(db,features,date);analog_prior=outcome_prior(analogs);store_analog(db,ticker,date,features);prediction['analog_prior']=analog_prior;prediction_governance['analog_prior']=analog_prior
  for archetype,score in archetypes.items(): db.upsert('strategy_archetype_scores',{'ticker':ticker,'date':date,'archetype':archetype,'score':score,'details_json':{'regime':regime['regime']}},['ticker','date','archetype'])
  db.upsert('predictions',{'ticker':ticker,'date':date,'model_id':prediction['model_id'],'probability':prediction['probability'],'expected_return':prediction['expected_return'],'expected_drawdown':prediction['expected_drawdown'],'target_before_stop':prediction['target_before_stop'],'uncertainty':prediction['uncertainty'],'explanation_json':prediction_governance},['ticker','date','model_id']);results.append({'ticker':ticker,'price':float(df.Close.iloc[-1]),'sector':sector_map.get(ticker,'Unknown'),'returns':df.Close.pct_change().dropna().tail(60),'features':features,'quality':quality[ticker],'prediction':prediction,'mc':mc['metrics'],'stress_penalty':stress['penalty'],'analog_prior':analog_prior,'archetypes':archetypes})
 ranked=rank(results,regime['regime'],play)
 for row in ranked:db.upsert('rankings',{'ticker':row['ticker'],'date':date,'score':row['score'],'action':row['action'],'position_size':0,'regime':regime['regime'],'stress_penalty':row['stress_penalty'],'confidence':row['confidence'],'details_json':row},['ticker','date'])
 _mature_labels(db,date);paired=db.rows('SELECT l.positive,p.probability FROM labels l JOIN predictions p ON p.ticker=l.ticker AND p.date=l.asof_date WHERE l.horizon_days=5 AND l.asof_date<? ORDER BY l.asof_date,l.ticker',(date,));observed=[row['positive'] for row in paired];probs=[row['probability'] for row in paired];cal_brier=brier(observed,probs) if observed else .25;cal_curve=calibration_curve(observed,probs) if observed else []
 mc_validation_rows=[]
 for engine in sorted({row['engine'] for row in db.rows('SELECT DISTINCT engine FROM mc_engine_validation')}):
  values=db.rows('SELECT predicted_probability,outcome FROM mc_engine_validation WHERE engine=?',(engine,));truth=np.asarray([row['outcome'] for row in values],dtype=float);forecast_values=np.clip(np.asarray([row['predicted_probability'] for row in values],dtype=float),1e-8,1-1e-8)
  if len(truth):
   metric={'engine':engine,'n':len(truth),'brier':float(np.mean((forecast_values-truth)**2)),'logloss':float(-np.mean(truth*np.log(forecast_values)+(1-truth)*np.log(1-forecast_values))),'calibration':calibration_curve(truth,forecast_values)};mc_validation_rows.append(metric);db.upsert('mc_validation_metrics',{'date':date,'engine':engine,'n':metric['n'],'brier':metric['brier'],'logloss':metric['logloss'],'calibration_json':metric['calibration']},['date','engine'])
 historical_probabilities=[row['probability'] for row in db.rows('SELECT probability FROM predictions WHERE date<? ORDER BY date DESC LIMIT 1000',(date,))];current_probabilities=[row['prediction']['probability'] for row in results];drift=detect_drift(historical_probabilities,current_probabilities) if len(historical_probabilities)>=20 and current_probabilities else {'psi':0.0,'ks':0.0,'flag':'insufficient_reference','reference_rows':len(historical_probabilities)};drift['flag']='material' if drift.get('psi',0)>.45 else drift['flag'];db.upsert('drift_metrics',{'date':date,'model_id':'quant_ensemble_v1','psi':drift.get('psi',0),'ks':drift.get('ks',0),'flag':drift['flag'],'details_json':drift},['date','model_id']);db.upsert('calibration_metrics',{'date':date,'model_id':'quant_ensemble_v1','brier':cal_brier,'logloss':cal_brier,'curve_json':cal_curve},['date','model_id'])
 previous=db.rows('SELECT values_json FROM features WHERE date < ? ORDER BY date DESC LIMIT 500',(date,));current={key:[row['features'].get(key,0.) for row in results] for key in (results[0]['features'] if results else {})}
 if previous:
  prior=[json.loads(row['values_json']) for row in previous]
  for feature,values in current.items():
   ref=[item.get(feature,0.) for item in prior];stat=stability(ref,values);db.upsert('feature_stability',{'date':date,'feature':feature,'psi':stat['psi'],'stable':int(stat['stable']),'details_json':stat},['date','feature'])
 curve=db.rows('SELECT * FROM equity_curve ORDER BY date');dd=curve[-1]['drawdown'] if curve else 0;kill=assess_kill_switch(dd,drift['flag']=='material',db.integrity()=='ok',cfg['risk']['max_drawdown_kill']);mrm=mrm_decision(kill,cal_brier,drift['psi']);closed_before=db.rows("SELECT pnl,entry,shares FROM paper_trades WHERE status='closed' ORDER BY id DESC LIMIT 20");reward=float(np.mean([row['pnl']/max(row['entry']*row['shares'],1) for row in closed_before])) if closed_before else 0.;feedback_multiplier=update_weight(1.,reward);mrm['risk_multiplier']*=feedback_multiplier;mrm['paper_feedback_reward']=reward;mrm['paper_feedback_multiplier']=feedback_multiplier;db.upsert('kill_switch_history',{'date':date,'active':int(kill['active']),'reason':kill['reason'],'metrics_json':{'drawdown':dd,'drift':drift}},['date']);db.upsert('model_risk_decisions',{'date':date,'decision':mrm['decision'],'reason':mrm['reason'],'details_json':mrm},['date'])
 candidates=[x for x in ranked if x['action']=='LONG' and x['quality']['data_confidence']>=cfg['risk']['min_data_confidence']];alloc=allocate(candidates,capital,mrm['risk_multiplier']) if not kill['active'] else []
 for row in alloc:db.upsert('allocations',{'ticker':row['ticker'],'date':date,'weight':row['weight'],'sector':row['sector'],'risk_budget':row['weight'],'reason':'kelly_cvar_correlation_sector_optimizer'},['ticker','date'])
 for row in alloc:db.execute('UPDATE rankings SET position_size=? WHERE ticker=? AND date=?',(row['weight'],row['ticker'],date))
 plans=[dict(make_plan(x['ticker'],x['price'],x['weight'],capital,atr_pct=x['features'].get('atr_pct')),weight=x['weight'],dollar_volume=x['features'].get('dollar_volume',0)) for x in alloc]
 for p in plans:db.upsert('trade_plans',{'ticker':p['ticker'],'date':date,'entry':p['entry'],'stop':p['stop'],'target':p['target'],'shares':p['shares'],'action':p['action'],'transaction_cost':p['transaction_cost']},['ticker','date'])
 paper=PaperEngine(db,capital);paper.mark_and_close({x['ticker']:{'open':float(frames[x['ticker']].Open.iloc[-1]),'high':float(frames[x['ticker']].High.iloc[-1]),'low':float(frames[x['ticker']].Low.iloc[-1]),'close':x['price'],'volume':float(frames[x['ticker']].Volume.iloc[-1])} for x in results},date);paper.open(plans,date);paper.snapshot(date);submissions=[stage('alpaca_order_'+plan['ticker'],lambda p=plan:broker.submit_long_bracket(date,p),{'submitted':False}) for plan in plans] if broker_state.get('enabled') else [];positions=db.rows('SELECT * FROM positions');trades=db.rows("SELECT * FROM paper_trades WHERE status='closed'");curve_rows=db.rows('SELECT * FROM equity_curve ORDER BY date');market_returns=spy.Close.pct_change().dropna().tail(60);factors={row['ticker']:factor_exposures(row['returns'].tail(60),market_returns) for row in alloc};factor_summary=aggregate_factor_exposure(alloc,factors);portfolio_return=float(curve_rows[-1]['daily_pnl']/max(curve_rows[-1]['equity'],1)) if curve_rows else 0.;alpha_beta=decompose_alpha_beta(portfolio_return,float(market_returns.iloc[-1]) if len(market_returns) else 0.,factor_summary.get('beta',0));analytics={'paper':paper_metrics(trades,curve_rows),'clustering':cluster_metrics(alloc),'risk_of_ruin':ruin_probability(.5,1.2,.01),'factor_exposure':factor_summary,'alpha_beta':alpha_beta,'attribution':summarize_attribution(trades),'trade_attribution':[trade_attribution(item) for item in trades[-20:]],'recomputed_drawdown':recompute_drawdown([row['equity'] for row in curve_rows]),'broker':{'state':broker_state,'submissions':submissions}};db.upsert('portfolio_analytics',{'date':date,'metrics_json':analytics,'factor_json':analytics['factor_exposure'],'alpha_beta_json':alpha_beta,'cost_json':{'planned_cost':sum(x['transaction_cost'] for x in plans)},'cluster_json':analytics['clustering'],'ruin_json':{'probability':analytics['risk_of_ruin']}},['date'])
 candidate_evidence=[{'ticker':x['ticker'],'probability':x['prediction']['probability'],'mc_expected_return':x['mc'].get('expected_return',0.0),'stress_penalty':x.get('stress_penalty',0.0),'data_confidence':x['quality'].get('data_confidence',0.0),'confidence':x['prediction'].get('confidence',0.5),'sector':x.get('sector','Unknown')} for x in ranked[:12]]
 debate,transcript=run_debate(db,date,{'regime':regime['regime'],'risk':regime['flip_bearish_5d'],'top_candidates':[x['ticker'] for x in alloc],'candidate_evidence':candidate_evidence,'calibration_brier':cal_brier,'drift_flag':drift.get('flag'),'kill_switch_active':kill['active'],'mc_validation':mc_validation_rows});llm=narrative(f"Narrate only these quantitative facts: regime={regime['regime']}; bear_flip={regime['flip_bearish_5d']:.3f}; candidates={[x['ticker'] for x in alloc]}; no investment advice.");db.upsert('agent_outputs',{'date':date,'agent':'LLMNarrative','payload_json':llm},['date','agent']);decision='CASH MODE / NO TRADE DAY' if kill['active'] or not plans or debate['action']=='CASH' else 'RANKED WATCHLIST'
 experiments=ExperimentManager(db)
 for signal in discover_signal(spy):
  experiment_id=f"signal_{signal['signal_name']}_v1";accepted=int(experiments.evaluate(experiment_id,signal['signal_name'],{'formula':signal['formula']},{'score':signal['score']}));db.upsert('signal_discovery',{'date':date,'signal_name':signal['signal_name'],'formula':signal['formula'],'score':signal['score'],'details_json':signal},['date','signal_name']);db.upsert('research_lab',{'date':date,'experiment_id':experiment_id,'signal_name':signal['signal_name'],'metrics_json':signal,'accepted':accepted},['date','experiment_id'])
 leakage=check_feature_frame(__import__('pandas').DataFrame([x['features'] for x in results]));db.upsert('leakage_detection',{'date':date,'passed':int(leakage['passed']),'findings_json':leakage['findings']},['date'])
 report={'date':date,'regime':regime['regime'],'regime_forecast':regime,'decision':decision,'picks':alloc[:12],'debate':debate,'narrative':llm,'kill_switch':kill,'universe_health':{'requested':initial_requested,'loaded':len(frames),'deep_analyzed':len(deep_tickers),'failure_rate':failure_rate,'source':source,'failover':failover},'analytics':analytics,'calibration':{'brier':cal_brier,'curve':cal_curve,'drift':drift},'mc_validation':mc_validation_rows,'stress':db.rows('SELECT scenario,AVG(pnl_pct) avg_pnl,AVG(penalty) avg_penalty FROM stress_outputs WHERE date=? GROUP BY scenario',(date,)),'equity_chart_artifact':'equity_curve_chart.html','provider_compliance':providers};ensure_artifacts();equity_rows=db.rows('SELECT * FROM equity_curve ORDER BY date');equity_chart(equity_rows).write_html(ensure_artifacts()/'equity_curve_chart.html',include_plotlyjs='cdn');equity_chart_png(equity_rows,ensure_artifacts()/'equity_curve.png');html=html_build(report);write_text('daily_report.html',html);write_text('daily_report.md',md_build(report));manifest=build_manifest(cfg,{'run_id':run_id,'universe_source':source,'loaded':len(frames),'deep_tickers':deep_tickers,'timings':timings});write_json('manifest.json',manifest);db.upsert('reproducibility_manifests',{'run_id':manifest['run_id'],'date':date,'manifest_json':manifest,'created_at':manifest['created_at']},['run_id'])
 if send_email:stage('email',lambda:send(html,'Systematic Swing Research '+date))
 status='success_with_degradation' if errors else 'success';finished_at=now_local().isoformat();db.upsert('pipeline_runs',{'run_id':run_id,'date':date,'status':status,'started_at':started_at,'finished_at':finished_at,'stage_timings_json':timings,'errors_json':errors},['run_id']);db.log('pipeline_finish','daily pipeline finished',{'run_id':run_id,'status':status,'timings':timings,'errors':errors});log_dir=ROOT.parent/'research_logs';log_dir.mkdir(exist_ok=True);(log_dir/f'{run_id}.json').write_text(json.dumps({'run_id':run_id,'status':status,'stage_timings':timings,'errors':errors},indent=2));return report
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--test-mode',action='store_true');parser.add_argument('--send-email',action='store_true');args=parser.parse_args();main(args.test_mode,args.send_email)
