import hashlib
from functools import partial
import pandas as pd, numpy as np, requests
from datetime import datetime
from core.validation import validate_ohlcv
def _tag(df,source):
 df.attrs['source']=source
 return df
def synthetic_ohlcv(ticker, periods=280):
 seed=int(hashlib.sha256(str(ticker).encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed); r=rng.normal(.0004,.02,periods); close=100*np.exp(np.cumsum(r)); idx=pd.bdate_range(end=pd.Timestamp.today().normalize(),periods=periods); return pd.DataFrame({'Open':close*(1+rng.normal(0,.003,periods)),'High':close*(1+abs(rng.normal(0,.008,periods))),'Low':close*(1-abs(rng.normal(0,.008,periods))),'Close':close,'Volume':rng.integers(1e5,5e6,periods)},index=idx)

def fetch_yahoo_chart(ticker,period='2y'):
 """Direct Yahoo chart fallback when the yfinance wrapper/curl path fails."""
 years=int(str(period).rstrip('y') or 2) if str(period).endswith('y') else 2
 end=int(pd.Timestamp.now(tz='UTC').timestamp());start=end-(years*365*24*3600)
 url=f'https://query1.finance.yahoo.com/v8/finance/chart/{str(ticker).upper()}'
 response=requests.get(url,params={'period1':start,'period2':end,'interval':'1d','events':'div,splits'},headers={'User-Agent':'quant-research/1.0'},timeout=10);response.raise_for_status();payload=response.json();result=((payload.get('chart') or {}).get('result') or [None])[0]
 if not result:raise RuntimeError('Yahoo chart returned no result')
 timestamps=result.get('timestamp') or [];quote=((result.get('indicators') or {}).get('quote') or [{}])[0]
 frame=pd.DataFrame({key:quote.get(key,[]) for key in ('open','high','low','close','volume')},index=pd.to_datetime(timestamps,unit='s',utc=True).tz_convert(None).normalize());frame=frame.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}).replace([np.inf,-np.inf],np.nan).dropna();valid,issues=validate_ohlcv(frame)
 if not valid or len(frame)<60:raise RuntimeError('invalid Yahoo chart: '+','.join(issues or ['insufficient_history']))
 return _tag(frame,'yahoo_chart')
def fetch_ohlcv(ticker,period='2y',test_mode=False):
 if test_mode:return _tag(synthetic_ohlcv(ticker),'synthetic_test')
 try:
  import yfinance as yf
  df=yf.download(ticker,period=period,auto_adjust=True,progress=False,threads=False)
  if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
  df=df.rename(columns=str.title)[['Open','High','Low','Close','Volume']].dropna(); ok,issues=validate_ohlcv(df)
  if ok and len(df)>60:return _tag(df,'yfinance')
  raise RuntimeError('invalid provider response: '+','.join(issues))
 except Exception as exc:
  try:
   return fetch_yahoo_chart(ticker,period=period)
  except Exception as yahoo_exc:
   try:
    from data_sources.stooq_source import fetch_stooq
    return fetch_stooq(ticker,test_mode=False)
   except Exception as fallback_exc:raise RuntimeError(f'OHLCV unavailable for {ticker}; yfinance={exc}; yahoo={yahoo_exc}; stooq={fallback_exc}') from fallback_exc

def fetch_ohlcv_batch(tickers, period='2y', test_mode=False, batch_size=100):
 """Fetch a broad universe in provider batches; failures are explicit, never silently dropped."""
 tickers=list(dict.fromkeys(tickers)); frames={}; failures={}
 if test_mode:
  return {ticker:_tag(synthetic_ohlcv(ticker),'synthetic_test') for ticker in tickers}, failures
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
      if valid and len(df)>=60: frames[ticker]=_tag(df,'yfinance')
      else: failures[ticker]=','.join(issues or ['insufficient_history'])
     except Exception as exc: failures[ticker]=str(exc)
   except Exception as exc:
    failures.update({ticker:str(exc) for ticker in batch})
 except Exception as exc:
  failures.update({ticker:str(exc) for ticker in tickers})
 # Independent provider fallback before Stooq; failures remain explicit.
 if failures:
  from core.parallel import map_resilient
  yahoo_fetch=partial(fetch_yahoo_chart,period=period)
  for ticker,frame,error in map_resilient(yahoo_fetch,list(failures),workers=min(16,len(failures))):
   if frame is not None:frames[ticker]=frame;failures.pop(ticker,None)
 if failures and len(failures)/max(len(tickers),1)<=.20:
  from core.parallel import map_resilient
  from data_sources.stooq_source import fetch_stooq
  for ticker,frame,error in map_resilient(fetch_stooq,list(failures),workers=min(8,len(failures))):
   if frame is not None:frames[ticker]=frame;failures.pop(ticker,None)
 return frames,failures
