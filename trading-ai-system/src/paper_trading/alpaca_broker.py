"""Paper-only Alpaca Trading API adapter with broker-state reconciliation and idempotent bracket submission."""
import os,json,hashlib
from datetime import datetime,timezone
import requests
class AlpacaPaperBroker:
 base_url='https://paper-api.alpaca.markets'
 def __init__(self,db,timeout=20):
  self.db=db;self.key=os.getenv('ALPACA_API_KEY');self.secret=os.getenv('ALPACA_SECRET_KEY');self.timeout=timeout
  self.enabled=bool(self.key and self.secret and os.getenv('ALPACA_PAPER_TRADE','true').lower()=='true')
  if os.getenv('ALPACA_PAPER_TRADE','true').lower()!='true':raise RuntimeError('Live trading is prohibited: ALPACA_PAPER_TRADE must be true')
 def _headers(self):return {'APCA-API-KEY-ID':self.key,'APCA-API-SECRET-KEY':self.secret,'Content-Type':'application/json'}
 def _request(self,method,path,**kwargs):
  response=requests.request(method,self.base_url+path,headers=self._headers(),timeout=self.timeout,**kwargs)
  if response.status_code>=400:raise RuntimeError(f'Alpaca {method} {path}: {response.status_code} {response.text[:500]}')
  return response.json() if response.content else {}
 def account(self):
  if not self.enabled:return None
  account=self._request('GET','/v2/account');self.db.upsert('broker_account_snapshots',{'date':datetime.now(timezone.utc).date().isoformat(),'equity':float(account['equity']),'cash':float(account['cash']),'buying_power':float(account['buying_power']),'details_json':account},['date']);return account
 def positions(self):return self._request('GET','/v2/positions') if self.enabled else []
 def open_orders(self):return self._request('GET','/v2/orders',params={'status':'open','nested':'true','limit':500}) if self.enabled else []
 def get_order(self,broker_order_id):
  if not self.enabled:return None
  return self._request('GET',f'/v2/orders/{broker_order_id}')
 def cancel_order(self,broker_order_id):
  if not self.enabled:return {'canceled':False,'reason':'alpaca_credentials_not_configured'}
  self._request('DELETE',f'/v2/orders/{broker_order_id}')
  order=self.get_order(broker_order_id)
  if order:
   self._persist_order(order,datetime.now(timezone.utc).date().isoformat())
  return {'canceled':True,'order':order}
 def _persist_order(self,order,date):
  self.db.upsert('broker_orders',{'client_order_id':order.get('client_order_id') or order['id'],'broker_order_id':order['id'],'ticker':order['symbol'],'date':date,'status':order['status'],'payload_json':{},'response_json':order,'updated_at':datetime.now(timezone.utc).isoformat()},['client_order_id'])
 def client_order_id(self,date,ticker):return 'qrs-'+hashlib.sha256(f'{date}:{ticker}:long-bracket-v1'.encode()).hexdigest()[:24]
 def submit_long_bracket(self,date,plan):
  if not self.enabled:return {'submitted':False,'reason':'alpaca_credentials_not_configured'}
  client_id=self.client_order_id(date,plan['ticker']);existing=self.db.rows('SELECT * FROM broker_orders WHERE client_order_id=?',(client_id,))
  if existing:return {'submitted':False,'reason':'duplicate_client_order_id','client_order_id':client_id}
  if plan['action']!='LONG' or plan['shares']<=0:raise ValueError('Broker accepts long-only positive-share plans')
  if not (plan['target']>plan['entry']>plan['stop']>0):raise ValueError('Invalid long bracket prices')
  payload={'symbol':plan['ticker'],'qty':str(int(plan['shares'])),'side':'buy','type':'market','time_in_force':'day','order_class':'bracket','client_order_id':client_id,'take_profit':{'limit_price':f"{plan['target']:.2f}"},'stop_loss':{'stop_price':f"{plan['stop']:.2f}"}}
  try:
   response=self._request('POST','/v2/orders',json=payload);status=response.get('status','accepted');self.db.upsert('broker_orders',{'client_order_id':client_id,'broker_order_id':response.get('id'),'ticker':plan['ticker'],'date':date,'status':status,'payload_json':payload,'response_json':response,'updated_at':datetime.now(timezone.utc).isoformat()},['client_order_id']);return {'submitted':True,'client_order_id':client_id,'response':response}
  except Exception as exc:
   self.db.upsert('broker_orders',{'client_order_id':client_id,'broker_order_id':None,'ticker':plan['ticker'],'date':date,'status':'rejected_or_error','payload_json':payload,'response_json':{'error':str(exc)},'updated_at':datetime.now(timezone.utc).isoformat()},['client_order_id']);raise
 def reconcile(self,date):
  if not self.enabled:return {'enabled':False,'positions':0,'orders':0}
  account=self.account();positions=self.positions();orders=self.open_orders()
  for order in orders:self._persist_order(order,date)
  return {'enabled':True,'equity':float(account['equity']),'positions':len(positions),'orders':len(orders)}
