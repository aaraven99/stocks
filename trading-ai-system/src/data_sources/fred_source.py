import numpy as np
from data_sources.yfinance_source import fetch_ohlcv
def macro_snapshot(test_mode=False):
 try:
  tlt=fetch_ohlcv('TLT',test_mode=test_mode)['Close'].pct_change().tail(20).mean(); return {'rates_proxy_return':float(tlt),'inflation_proxy':0.03,'credit_proxy':0.0}
 except Exception:return {'rates_proxy_return':0.0,'inflation_proxy':0.03,'credit_proxy':0.0}
