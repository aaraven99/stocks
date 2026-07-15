import pandas as pd, numpy as np
from datetime import datetime
from core.validation import validate_ohlcv
def synthetic_ohlcv(ticker, periods=280):
 rng=np.random.default_rng(abs(hash(ticker))%2**32); r=rng.normal(.0004,.02,periods); close=100*np.exp(np.cumsum(r)); idx=pd.bdate_range(end=pd.Timestamp.today().normalize(),periods=periods); return pd.DataFrame({'Open':close*(1+rng.normal(0,.003,periods)),'High':close*(1+abs(rng.normal(0,.008,periods))),'Low':close*(1-abs(rng.normal(0,.008,periods))),'Close':close,'Volume':rng.integers(1e5,5e6,periods)},index=idx)
def fetch_ohlcv(ticker,period='2y',test_mode=False):
 if test_mode:return synthetic_ohlcv(ticker)
 try:
  import yfinance as yf
  df=yf.download(ticker,period=period,auto_adjust=True,progress=False,threads=False)
  if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
  df=df.rename(columns=str.title)[['Open','High','Low','Close','Volume']].dropna(); ok,issues=validate_ohlcv(df)
  if ok and len(df)>60:return df
  raise RuntimeError('invalid provider response: '+','.join(issues))
 except Exception as exc: raise RuntimeError(f'OHLCV unavailable for {ticker}: {exc}') from exc

def fetch_ohlcv_batch(tickers, period='2y', test_mode=False, batch_size=100):
 """Fetch a broad universe in provider batches; failures are explicit, never silently dropped."""
 tickers=list(dict.fromkeys(tickers)); frames={}; failures={}
 if test_mode:
  return {ticker:synthetic_ohlcv(ticker) for ticker in tickers}, failures
 try:
  import yfinance as yf
  for start in range(0,len(tickers),batch_size):
   batch=tickers[start:start+batch_size]
   try:
    raw=yf.download(batch,period=period,auto_adjust=True,progress=False,threads=True,group_by='ticker')
    for ticker in batch:
     try:
      df=raw[ticker].copy() if isinstance(raw.columns,pd.MultiIndex) else raw.copy()
      df=df.rename(columns=str.title)[['Open','High','Low','Close','Volume']].dropna()
      valid,issues=validate_ohlcv(df)
      if valid and len(df)>=60: frames[ticker]=df
      else: failures[ticker]=','.join(issues or ['insufficient_history'])
     except Exception as exc: failures[ticker]=str(exc)
   except Exception as exc:
    failures.update({ticker:str(exc) for ticker in batch})
 except Exception as exc:
  failures.update({ticker:str(exc) for ticker in tickers})
 return frames,failures
