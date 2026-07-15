"""Opt-in provider throughput test for a real 3,000-symbol Stage 1 run."""
import os,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_sources.ticker_universe_source import load_universe
from data_sources.yfinance_source import fetch_ohlcv_batch

@unittest.skipUnless(os.getenv('RUN_LIVE_THROUGHPUT')=='1','set RUN_LIVE_THROUGHPUT=1 to use live market data')
class LiveThroughputTests(unittest.TestCase):
 def test_three_thousand_symbol_stage_contract(self):
  universe,source=load_universe('full');universe=universe[:3000];frames,failures=fetch_ohlcv_batch(universe,batch_size=100)
  self.assertGreaterEqual(len(universe),3000);self.assertLess(len(failures)/len(universe),.50,{'source':source,'failures':len(failures)});self.assertGreaterEqual(min(250,len(frames)),200)

if __name__=='__main__':unittest.main()
