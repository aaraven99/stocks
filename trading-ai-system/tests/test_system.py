import json,os,sys,unittest,uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storage.db import Database
from montecarlo.montecarlo_engine import MonteCarloEngine
from montecarlo.montecarlo_validation import validate
from paper_trading.paper_engine import PaperEngine
from data_sources.ticker_universe_source import load_file_fallback,load_universe
from pipeline.stages import fast_filter,select_deep_tickers,select_fast_universe,validate_universe_contract
from models.model_registry import ModelRegistry
from reporting.html_builder import build as build_html
from features.feature_engineering import make_features,feature_definitions
from agents.debate_engine import DebateEngine
from agents.technical_agent import TechnicalAgent
from paper_trading.paper_metrics import metrics as paper_metrics
from data_sources.provider_compliance import provider_compliance
from backtesting.backtester import _marked_portfolio_returns
from portfolio.portfolio_optimizer import mean_variance_score
class SystemTests(unittest.TestCase):
 def db_path(self):
  path=Path.cwd()/'artifacts'/'test_dbs'/f'{uuid.uuid4().hex}.sqlite';path.parent.mkdir(parents=True,exist_ok=True);return path
 def test_schema_and_paper_lifecycle(self):
  db=Database(self.db_path());self.assertEqual(db.integrity(),'ok')
  engine=PaperEngine(db,100000);engine.open([{'ticker':'TEST','shares':100,'entry':100,'stop':94,'target':110,'weight':.1,'transaction_cost':12}], '2026-01-02')
  self.assertEqual(len(db.rows('SELECT * FROM positions')),1);engine.mark_and_close({'TEST':111},'2026-01-05')
  trades=db.rows("SELECT * FROM paper_trades WHERE status='closed'");self.assertEqual(len(trades),1);self.assertLess(trades[0]['pnl'],1100)
  engine.snapshot('2026-01-05');self.assertEqual(len(db.rows('SELECT * FROM equity_curve')),1)
 def test_montecarlo_barriers_and_quantiles(self):
  returns=np.random.default_rng(42).normal(.0005,.015,300);out=MonteCarloEngine(sims=1000,seed=42).run(__import__('pandas').Series(returns),'bull_low_vol',100)
  self.assertTrue(validate(out['metrics'])['passed']);self.assertEqual(len(out['distribution']),1000)
  for key in ['p_target_before_stop','drawdown_q05','price_target_q50','p_plus_3_5d','p_plus_10_20d']:self.assertIn(key,out['metrics'])
  self.assertTrue(0<=out['metrics']['p_target_before_stop']<=1)
 def test_backtest_marks_overlapping_positions_with_capped_notional(self):
  trades=[{'signal_date':'2026-01-02','exit_date':'2026-01-09','net_return':.10,'outcome':1}]
  daily=_marked_portfolio_returns(trades,['2026-01-02','2026-01-05','2026-01-06','2026-01-07','2026-01-08','2026-01-09'],.10)
  self.assertEqual(len(daily),6);self.assertEqual(sum(x!=0 for x in daily),5);self.assertEqual(trades[0]['holding_days'],5);self.assertAlmostEqual(sum(daily),.01,places=3)
 def test_portfolio_mean_variance_component_is_risk_sensitive(self):
  self.assertGreater(mean_variance_score(.10,.10),mean_variance_score(.10,.30))
 def test_required_persistence_tables(self):
  db=Database(self.db_path());names={row['name'] for row in db.rows("SELECT name FROM sqlite_master WHERE type='table'")}
  required={'raw_prices','cleaned_prices','features','predictions','montecarlo_metrics','rankings','agent_outputs','paper_trades','equity_curve','experiments','drift_metrics','calibration_metrics','analog_memory','labels','historical_backfills','strategy_archetype_scores'};self.assertTrue(required<=names)
 def test_universe_assets_and_throughput_contract(self):
  full,full_source=load_universe('full',force_file=True);backup,backup_source=load_file_fallback('backup');top,top_source=load_file_fallback('top500')
  self.assertGreaterEqual(len(full),3000);self.assertLessEqual(len(full),8000);self.assertGreaterEqual(len(backup),2000);self.assertGreaterEqual(len(top),500)
  self.assertEqual(full_source,'universe_full.csv');self.assertEqual(backup_source,'universe_backup_2000.csv');self.assertEqual(top_source,'universe_top500.csv')
  frames={ticker:pd.DataFrame({'Close':[1,2,3]}) for ticker in select_fast_universe(full,3000)}
  rows=fast_filter(frames,lambda frame: frame.Close.iloc[-1]);deep=select_deep_tickers(rows,250);contract=validate_universe_contract(3000,len(rows),len(deep))
  self.assertTrue(contract['passed'],contract)
 def test_model_challenger_rejection_and_rollback(self):
  db=Database(self.db_path());registry=ModelRegistry(db)
  registry.register_champion('champion-v1','rf',{'brier':.20},'v1.joblib')
  rejected=registry.promote_if_better('challenger-bad','rf',{'brier':.30},'bad.joblib')
  self.assertEqual(rejected['decision'],'reject');self.assertEqual(registry.champion()['model_id'],'champion-v1')
  registry.register('champion-v2','rf','retired',{'brier':.18},'v2.joblib')
  rolled=registry.rollback_to_latest_retired('regression')
  self.assertEqual(rolled['decision'],'rollback');self.assertEqual(registry.champion()['model_id'],'champion-v2')
  registry.register_champion('champion-bad','rf',{'brier':.45},'bad.joblib')
  registry.register('champion-good','rf','retired',{'brier':.18},'good.joblib')
  auto=registry.rollback_if_breached(.30)
  self.assertEqual(auto['decision'],'rollback');self.assertEqual(registry.champion()['model_id'],'champion-good')
 def test_feature_categories_agents_and_paper_metrics(self):
  idx=pd.bdate_range('2025-01-01',periods=280);base=np.linspace(100,130,len(idx));df=pd.DataFrame({'Open':base*.99,'High':base*1.01,'Low':base*.98,'Close':base,'Volume':np.linspace(1e6,2e6,len(idx))},index=idx)
  features=make_features(df,{'macro':{'rates_proxy_return':.01},'sentiment':{'sentiment':.2,'news_count':3},'options':{'put_call_proxy':.8},'event':{'event_risk':0},'breadth':{'pct_above_50d':.6}})
  categories={item['category'] for item in feature_definitions().values()}
  self.assertGreaterEqual(len(features),20);self.assertIn('trend_weekly_return_5w',features);self.assertGreaterEqual(len(categories),15)
  db=Database(self.db_path());verdict,transcript=DebateEngine(db,[TechnicalAgent(db)]).run('2026-01-02',{'regime':'bull_low_vol','risk':.2})
  self.assertTrue(transcript[0]['evidence']);self.assertIn('confidence',transcript[0]);self.assertIn('action',verdict)
  self.assertEqual(paper_metrics([{'pnl':10}], [{'drawdown':-.12}])['max_drawdown'],-.12)
 def test_unavailable_alternative_data_is_not_proxy(self):
  with patch.dict(os.environ,{'FINNHUB_API_KEY':'','POLYGON_API_KEY':'','FRED_API_KEY':''},clear=False):
   statuses=provider_compliance()
  for name in ('news_sentiment','options_flow','earnings_calendar'):
   self.assertNotIn('proxy',statuses[name]['status']);self.assertFalse(statuses[name]['decision_use'])
 def test_configured_alternative_data_is_decision_usable(self):
  with patch.dict(os.environ,{'FINNHUB_API_KEY':'test','POLYGON_API_KEY':'test','FRED_API_KEY':'test'},clear=False):
   statuses=provider_compliance()
  for name in ('macro','news_sentiment','options_flow','earnings_calendar'):
   self.assertEqual(statuses[name]['status'],'provider-backed');self.assertTrue(statuses[name]['decision_use'])
 def test_report_required_sections(self):
  report={'date':'2026-01-02','regime':'BULL','regime_forecast':{},'decision':'RANKED WATCHLIST','picks':[],'debate':{},'narrative':{'text':'Narrative'},'universe_health':{},'analytics':{'paper':{},'factor_exposure':{}},'calibration':{'drift':{}},'stress':[],'provider_compliance':{'options_flow':{'status':'proxy-non-compliant','provider':'test','decision_use':False}}}
  html=build_html(report)
  for section in ['Market Dashboard & Regime Forecast','Monte Carlo Summary','ML Confidence, Calibration & Drift','Stress Test Results','Paper Trading & Equity Curve','Agent Debate Summary','Attribution Summary','Alternative Data Governance','proxy-non-compliant','Research watchlist only','equity_curve_chart.html','Paper trading equity curve']:
   self.assertIn(section,html)
if __name__=='__main__':unittest.main()
