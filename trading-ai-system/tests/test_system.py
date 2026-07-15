import json,sys,unittest,uuid
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storage.db import Database
from montecarlo.montecarlo_engine import MonteCarloEngine
from montecarlo.montecarlo_validation import validate
from paper_trading.paper_engine import PaperEngine
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
 def test_required_persistence_tables(self):
  db=Database(self.db_path());names={row['name'] for row in db.rows("SELECT name FROM sqlite_master WHERE type='table'")}
  required={'raw_prices','cleaned_prices','features','predictions','montecarlo_metrics','rankings','agent_outputs','paper_trades','equity_curve','experiments','drift_metrics','calibration_metrics','analog_memory','labels'};self.assertTrue(required<=names)
if __name__=='__main__':unittest.main()
