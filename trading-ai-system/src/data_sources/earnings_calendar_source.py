import os
from datetime import date,timedelta
import requests

def event_risk(ticker):
 key=os.getenv('FINNHUB_API_KEY')
 if not key:return {'available':False,'source':'unavailable_provider_credentials'}
 try:
  payload=requests.get('https://finnhub.io/api/v1/calendar/earnings',params={'symbol':ticker,'from':date.today().isoformat(),'to':(date.today()+timedelta(days=30)).isoformat(),'token':key},timeout=15).json()
  rows=[x for x in payload.get('earningsCalendar',[]) if x.get('symbol')==ticker]
  days=min([int((date.fromisoformat(x['date'])-date.today()).days) for x in rows if x.get('date')],default=30)
  return {'available':True,'days_to_earnings':days,'event_risk':float(days<=5),'source':'finnhub_earnings_calendar'}
 except Exception as exc:return {'available':False,'source':'provider_error','error':str(exc)[:200]}
