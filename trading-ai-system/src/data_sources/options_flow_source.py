import os
import requests

def options_flow_snapshot(ticker):
 key=os.getenv('POLYGON_API_KEY')
 if not key:return {'available':False,'source':'unavailable_provider_credentials'}
 try:
  payload=requests.get(f'https://api.polygon.io/v3/snapshot/options/{ticker}',params={'limit':250,'apiKey':key},timeout=20).json()
  rows=payload.get('results',[]); calls=sum(float(x.get('day',{}).get('volume') or 0) for x in rows if x.get('details',{}).get('contract_type')=='call'); puts=sum(float(x.get('day',{}).get('volume') or 0) for x in rows if x.get('details',{}).get('contract_type')=='put')
  return {'available':True,'put_call_ratio':puts/max(calls,1.),'unusual_flow':float(max(puts,calls)),'source':'polygon_options_snapshot'}
 except Exception as exc:return {'available':False,'source':'provider_error','error':str(exc)[:200]}

options_flow_proxy=options_flow_snapshot
