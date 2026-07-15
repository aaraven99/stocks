from pathlib import Path
import os
import pandas as pd, requests
from core.cache import DiskCache
FALLBACK=['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','JPM','XOM','UNH','AVGO','COST','LLY','AMD','NFLX','SPY','QQQ','IWM','DIA','XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLU','XLRE']
def _dynamic_universe():
 cache=DiskCache('.cache',ttl=86400); cached=cache.get('us_equity_universe')
 if cached and len(cached)>=500:return cached
 headers={'User-Agent':'Mozilla/5.0 institutional-research-contact@example.com','Accept':'application/json'}; symbols=[]
 for endpoint in ['https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=nasdaq','https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=nyse']:
  try:
   rows=requests.get(endpoint,headers=headers,timeout=20).json()['data']['rows']; symbols += [r.get('symbol','') for r in rows if r.get('symbol','') and '^' not in r.get('symbol','')]
  except Exception: continue
 symbols=list(dict.fromkeys(s.replace('.','-') for s in symbols))
 if len(symbols)>=500: cache.set('us_equity_universe',symbols);return symbols
 return []
def load_universe(mode='top500', force_file=False):
 dynamic=[] if force_file or os.getenv('TEST_MODE','').lower() in ('1','true','yes') else _dynamic_universe()
 if dynamic:
  if mode in ('top500','top'):return dynamic[:500],'dynamic_nasdaq_nyse_top500'
  if mode in ('backup','backup_2000'):return dynamic[:2000],'dynamic_nasdaq_nyse_backup2000'
  return dynamic[:8000],'dynamic_nasdaq_nyse_full'
 root=Path(__file__).resolve().parents[1]/'data'
 names={'full':['universe_full.csv','universe_backup_2000.csv','universe_top500.csv'],'backup':['universe_backup_2000.csv','universe_top500.csv'],'backup_2000':['universe_backup_2000.csv','universe_top500.csv'],'top':['universe_top500.csv'],'top500':['universe_top500.csv']}
 for name in names.get(mode,names['full']):
  p=root/name
  if p.exists():
   try:
    values=pd.read_csv(p,keep_default_na=False).iloc[:,0].astype(str).tolist()
    if values:return values,name
   except Exception: pass
 return FALLBACK,'embedded_resilient_fallback'
def load_file_fallback(mode):
 """Mandatory deterministic failover order: full CSV, backup_2000 CSV, top500 CSV."""
 return load_universe(mode,force_file=True)
