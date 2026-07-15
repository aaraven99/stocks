from data_sources.yfinance_source import fetch_ohlcv
from core.constants import BENCHMARKS
def cross_asset_snapshot(test_mode=False):
 out={}
 for ticker in BENCHMARKS:
  try:
   x=fetch_ohlcv(ticker,test_mode=test_mode).Close; out[ticker]={'return_5d':float(x.pct_change(5).iloc[-1]),'vol_20d':float(x.pct_change().tail(20).std()*252**.5),'available':True}
  except Exception: out[ticker]={'return_5d':0.,'vol_20d':0.,'available':False}
 return out
