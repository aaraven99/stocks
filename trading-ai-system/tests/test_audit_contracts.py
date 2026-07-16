import os,sys,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from data_sources.yfinance_source import synthetic_ohlcv,fetch_ohlcv_batch,fetch_yahoo_chart
import data_sources.ticker_universe_source as universe_source
from data_sources.ticker_universe_source import load_universe,_ranked_rows
from features.feature_engineering import feature_definitions,make_features
from models.training import purged_splits
from paper_trading.paper_engine import PaperEngine
from storage.db import Database
from reporting.email_sender import send as send_email
from agents.agent_manager import run_debate
from backtesting.backtester import walk_forward
from montecarlo.montecarlo_engine import MonteCarloEngine


class AuditContractTests(unittest.TestCase):
 def test_synthetic_data_is_deterministic(self):
  self.assertTrue(synthetic_ohlcv('TEST').equals(synthetic_ohlcv('TEST')))
 def test_test_mode_provider_provenance_is_explicit(self):
   frames,failures=fetch_ohlcv_batch(['TEST'],test_mode=True);self.assertFalse(failures);self.assertEqual(frames['TEST'].attrs.get('source'),'synthetic_test')

 @patch('data_sources.yfinance_source.requests.get')
 def test_yahoo_chart_parser_is_provider_tagged(self,get):
  bars=80;get.return_value.json.return_value={'chart':{'result':[{'timestamp':list(range(1700000000,1700000000+bars*86400,86400)),'indicators':{'quote':[{'open':[100+i*.1 for i in range(bars)],'high':[101+i*.1 for i in range(bars)],'low':[99+i*.1 for i in range(bars)],'close':[100+i*.1 for i in range(bars)],'volume':[1000000]*bars}]}}]}}
  get.return_value.raise_for_status.return_value=None;frame=fetch_yahoo_chart('TEST','2y');self.assertEqual(len(frame),bars);self.assertEqual(frame.attrs.get('source'),'yahoo_chart');get.assert_called_once()

 def test_every_computed_feature_has_definition(self):
  features=make_features(synthetic_ohlcv('FEATURES'),{'macro':{},'sentiment':{},'options':{},'event':{},'breadth':{}});definitions=feature_definitions()
  self.assertEqual(set(features),set(definitions));self.assertGreaterEqual(len({item['category'] for item in definitions.values()}),15)

 def test_date_grouped_purge_and_embargo(self):
  dates=np.repeat(pd.bdate_range('2025-01-01',periods=80).astype(str),3)
  folds=list(purged_splits(len(dates),dates=dates,embargo=5));self.assertTrue(folds)
  for train,test in folds:self.assertLessEqual(pd.Timestamp(dates[train].max()),pd.Timestamp(dates[test].min())-pd.offsets.BDay(5))

 def test_test_mode_database_path_and_cash_accounting(self):
  path=ROOT/'artifacts/test_dbs'/f'audit_{uuid.uuid4().hex}.sqlite'
  with patch.dict(os.environ,{'TEST_MODE':'1','TRADING_DB_PATH':str(path)}):
   db=Database();self.assertEqual(db.path,path);engine=PaperEngine(db,100000);engine.open([{'ticker':'TEST','shares':10,'entry':100,'stop':94,'target':110,'weight':.01}], '2026-01-02');self.assertLess(db.rows('SELECT cash FROM paper_account')[0]['cash'],99000.01);engine.mark_and_close({'TEST':{'open':100,'high':103,'low':93,'close':96,'volume':1000000}},'2026-01-05');engine.snapshot('2026-01-05');self.assertEqual(len(db.rows('SELECT * FROM execution_events')),2);self.assertEqual(len(db.rows('SELECT * FROM positions')),0);self.assertEqual(len(db.rows('SELECT * FROM paper_trades')),1)

 def test_batched_upsert_is_atomic_and_json_safe(self):
  path=ROOT/'artifacts/test_dbs'/f'audit_batch_{uuid.uuid4().hex}.sqlite'
  with patch.dict(os.environ,{'TRADING_DB_PATH':str(path)}):
   db=Database();db.upsert_many('feature_definitions',[{'name':'batch_a','category':'test','description':'a','provenance':'unit','version_hash':'a','created_at':'2026-01-01'},{'name':'batch_b','category':'test','description':'b','provenance':'unit','version_hash':'b','created_at':'2026-01-01'}],['name']);self.assertEqual(len(db.rows("SELECT * FROM feature_definitions WHERE name LIKE 'batch_%'")),2)

 def test_rejected_records_are_quarantined(self):
  path=ROOT/'artifacts/test_dbs'/f'quarantine_{uuid.uuid4().hex}.sqlite'
  with patch.dict(os.environ,{'TRADING_DB_PATH':str(path)}):
   db=Database();db.quarantine('raw_prices','TEST:2026-01-05','provider_ingestion_failure',{'error':'timeout'})
   row=db.rows('SELECT * FROM data_quarantine')[0]
  self.assertEqual(row['source_table'],'raw_prices');self.assertIn('timeout',row['payload_json'])

 def test_ranked_universe_fallbacks_and_parser(self):
  full,source=load_universe('full',force_file=True);backup,_=load_universe('backup_2000',force_file=True);top,_=load_universe('top500',force_file=True);self.assertGreaterEqual(len(full),3000);self.assertEqual(len(backup),2000);self.assertEqual(len(top),500);self.assertEqual(full[:3],top[:3]);self.assertEqual(_ranked_rows({'data':{'table':{'rows':[{'symbol':'ABC','name':'ABC Corp','marketCap':'1,000'},{'symbol':'ABCW','name':'ABC Warrant','marketCap':'9,000'}]}}},'nasdaq')[0]['symbol'],'ABC')

 def test_partial_dynamic_universe_cannot_be_labeled_full(self):
  with patch.object(universe_source,'_dynamic_universe',return_value=['A']*500):
   values,source=universe_source.load_universe('full')
  self.assertGreaterEqual(len(values),3000)
  self.assertNotIn('dynamic_',source)

 @patch('reporting.email_sender.smtplib.SMTP_SSL')
 def test_gmail_mime_delivery_contract(self,smtp_cls):
  smtp=smtp_cls.return_value.__enter__.return_value
  with patch.dict(os.environ,{'EMAIL_SCHEDULED_RUN':'true','EMAIL_TO':'a@example.com,b@example.com','GMAIL_USER':'sender@example.com','GMAIL_APP_PASSWORD':'test-password','EMAIL_FROM':'sender@example.com'},clear=False):
   result=send_email('<html><body>audit</body></html>','Audit')
  self.assertTrue(result['sent']);smtp.login.assert_called_once_with('sender@example.com','test-password');self.assertEqual(result['recipients'],2);self.assertIn('multipart/alternative',smtp.sendmail.call_args.args[2])

 def test_debate_agents_emit_evidence_based_json(self):
  path=ROOT/'artifacts/test_dbs'/f'agent_{uuid.uuid4().hex}.sqlite'
  with patch.dict(os.environ,{'TRADING_DB_PATH':str(path)}):
   db=Database();verdict,transcript=run_debate(db,'2026-01-05',{'regime':'bull_low_vol','risk':.10,'candidate_evidence':[{'ticker':'TEST','probability':.72,'mc_expected_return':.04,'stress_penalty':.01,'data_confidence':.95}],'drift_flag':'stable','kill_switch_active':False})
  self.assertEqual(verdict['rounds'],7);self.assertGreater(verdict['evidence_count'],0);self.assertEqual(len(transcript),26*7);self.assertTrue(all(x['schema_version']=='agent_assessment_v2' and x['evidence'] for x in transcript));self.assertGreater(len({x['focus'] for x in transcript}),10)

 def test_root_workflow_dst_state_and_delivery_contract(self):
  text=(ROOT.parent/'.github/workflows/daily_run.yml').read_text();self.assertIn("cron: '0 7 * * 1-5'",text);self.assertIn("cron: '0 8 * * 1-5'",text);self.assertIn('actions/checkout@v6',text);self.assertIn('actions/cache/restore@v5',text);self.assertIn('quant-state-',text);self.assertIn('replace(hour=7, minute=0',text);self.assertIn('timedelta(minutes=5)',text);self.assertIn('artifacts/trading_system.sqlite',text);self.assertIn('artifacts/equity_curve.png',text);self.assertIn('if-no-files-found: error',text)

 def test_weekly_workflow_requires_governance_artifacts(self):
  text=(ROOT.parent/'.github/workflows/weekly_train.yml').read_text();
  for artifact in ('champion_model.joblib','calibration_objects.json','model_registry_snapshot.json','backtest_report.md','monthly_research_review.md'):
   self.assertIn(artifact,text)

 def test_monte_carlo_has_probability_weighted_target(self):
  from data_sources.yfinance_source import synthetic_ohlcv
  returns=synthetic_ohlcv('MC_CONTRACT').Close.pct_change().dropna()
  result=MonteCarloEngine(1000,42).run(returns,'bull_low_vol',100.0)
  self.assertIn('probability_weighted_price_target',result['metrics'])
  self.assertGreater(result['metrics']['probability_weighted_price_target'],0)

 def test_backtest_governance_defaults_are_configured(self):
  cfg=(ROOT/'src/config/config.yaml').read_text();self.assertIn('model_family: xgboost',cfg);self.assertIn('probability_floor: 0.78',cfg);self.assertIn('embargo_dates: 5',cfg);self.assertEqual(walk_forward.__defaults__[1],5)


if __name__=='__main__':unittest.main()
