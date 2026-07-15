import os
from datetime import date,timedelta
import requests

def sentiment_snapshot(ticker):
 key=os.getenv('FINNHUB_API_KEY')
 if not key:return {'available':False,'source':'unavailable_provider_credentials'}
 try:
  rows=requests.get('https://finnhub.io/api/v1/company-news',params={'symbol':ticker,'from':(date.today()-timedelta(days=7)).isoformat(),'to':date.today().isoformat(),'token':key},timeout=15).json()
  scores=[float(item.get('sentiment',0)) for item in rows if item.get('sentiment') is not None]
  return {'available':True,'sentiment':sum(scores)/len(scores) if scores else 0.,'news_count':len(rows),'source':'finnhub_company_news'}
 except Exception as exc:return {'available':False,'source':'provider_error','error':str(exc)[:200]}
