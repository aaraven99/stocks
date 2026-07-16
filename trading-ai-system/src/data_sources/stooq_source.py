"""Independent Stooq daily OHLCV fallback (not a yfinance alias)."""
from io import StringIO
import pandas as pd,requests
from core.validation import validate_ohlcv


def fetch_stooq(ticker,test_mode=False):
 if test_mode:
  from data_sources.yfinance_source import synthetic_ohlcv
  frame=synthetic_ohlcv(ticker);frame.attrs['source']='synthetic_test';return frame
 symbol=str(ticker).lower().replace('-','.')
 if not symbol.startswith('^') and not symbol.endswith('.us'):symbol+='.us'
 response=requests.get('https://stooq.com/q/d/l/',params={'s':symbol,'i':'d'},timeout=20,headers={'User-Agent':'quant-research/1.0'});response.raise_for_status();frame=pd.read_csv(StringIO(response.text),parse_dates=['Date']).set_index('Date').rename(columns=str.title)
 frame=frame[['Open','High','Low','Close','Volume']].sort_index().dropna();valid,issues=validate_ohlcv(frame)
 if not valid or len(frame)<60:raise RuntimeError('invalid Stooq response: '+','.join(issues or ['insufficient_history']))
 frame.attrs['source']='stooq';return frame
