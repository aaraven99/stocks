import os
import requests

def _latest(series,key):
 rows=[x for x in series if x.get('value') not in (None,'.')]
 return float(rows[-1]['value']) if rows else 0.

def macro_snapshot(test_mode=False):
 if test_mode:return {'available':False,'source':'test_mode'}
 key=os.getenv('FRED_API_KEY')
 if not key:return {'available':False,'source':'unavailable_provider_credentials'}
 try:
  values={}
  for name in ('DFF','CPIAUCSL','BAMLH0A0HYM2'):
   payload=requests.get('https://api.stlouisfed.org/fred/series/observations',params={'series_id':name,'api_key':key,'file_type':'json','sort_order':'asc'},timeout=20).json();values[name]=_latest(payload.get('observations',[]),name)
  return {'available':True,'rates_return':values['DFF'],'inflation':values['CPIAUCSL'],'credit_spread':values['BAMLH0A0HYM2'],'source':'fred_series_api'}
 except Exception as exc:return {'available':False,'source':'provider_error','error':str(exc)[:200]}
