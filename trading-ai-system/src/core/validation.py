import pandas as pd
def validate_ohlcv(df):
 errors=[]
 if df is None or df.empty: return False,['empty']
 if not pd.DatetimeIndex(pd.to_datetime(df.index)).is_monotonic_increasing: errors.append('non_monotonic_dates')
 for c in ['Open','High','Low','Close']:
  if c not in df or (df[c]<=0).any(): errors.append('invalid_'+c.lower())
 if 'Volume' in df and (df.Volume<0).any(): errors.append('negative_volume')
 if {'High','Low'}.issubset(df) and (df.High<df.Low).any(): errors.append('high_below_low')
 return not errors, errors
