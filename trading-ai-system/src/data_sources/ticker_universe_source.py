"""Ranked US equity/ETF universe retrieval with deterministic CSV failover."""
from pathlib import Path
import os,re
import pandas as pd, requests
from core.cache import DiskCache

FALLBACK=['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','JPM','XOM','UNH','AVGO','COST','LLY','AMD','NFLX','SPY','QQQ','IWM','DIA','XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLU','XLRE']
_SYMBOL_RE=re.compile(r'^[A-Z][A-Z0-9.-]{0,5}$')
_EXCLUDE_RE=re.compile(r'warrant|unit|right|notes|preferred|depositary|debenture',re.I)


def _ranked_rows(payload,exchange):
 rows=((payload or {}).get('data') or {}).get('table',{}).get('rows',[]) or []
 out=[]
 for row in rows:
  symbol=str(row.get('symbol','')).upper().replace('.','-');name=str(row.get('name',''));cap=str(row.get('marketCap','')).replace(',','')
  if not _SYMBOL_RE.fullmatch(symbol) or _EXCLUDE_RE.search(name) or not re.fullmatch(r'\d+(?:\.\d+)?',cap):continue
  out.append({'symbol':symbol,'market_cap':float(cap),'exchange':exchange})
 return out


def _dynamic_universe():
 cache=DiskCache('.cache',ttl=86400);cached=cache.get('us_equity_universe_v2')
 if isinstance(cached,list) and len(cached)>=500:
  return [str(row.get('symbol',row)) if isinstance(row,dict) else str(row) for row in cached]
 headers={'User-Agent':'Mozilla/5.0 quant-research-contact@example.com','Accept':'application/json'};by_symbol={}
 for exchange in ('nasdaq','nyse'):
  try:
   endpoint=f'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange={exchange}'
   payload=requests.get(endpoint,headers=headers,timeout=30).json()
   for row in _ranked_rows(payload,exchange):
    if row['symbol'] not in by_symbol or row['market_cap']>by_symbol[row['symbol']]['market_cap']:by_symbol[row['symbol']]=row
  except (OSError,ValueError,KeyError,TypeError,requests.RequestException):
   continue
 ranked=sorted(by_symbol.values(),key=lambda row:(-row['market_cap'],row['symbol']))
 if len(ranked)>=500:
  cache.set('us_equity_universe_v2',ranked);return [row['symbol'] for row in ranked]
 return []


def _read_csv(path,minimum=1):
 try:
  frame=pd.read_csv(path,keep_default_na=False)
  if frame.empty or str(frame.columns[0]).lower()!='ticker':return []
  values=frame.iloc[:,0].astype(str).str.upper().str.replace('.','-',regex=False).tolist()
  values=[value for value in values if _SYMBOL_RE.fullmatch(value)]
  values=list(dict.fromkeys(values));return values if len(values)>=int(minimum) else []
 except (OSError,ValueError,KeyError,IndexError,pd.errors.ParserError):
  return []


def load_universe(mode='top500',force_file=False):
 dynamic=[] if force_file or os.getenv('TEST_MODE','').lower() in ('1','true','yes') else _dynamic_universe()
 if dynamic:
  # A "full" run is explicitly a 3,000+ symbol attempt.  Do not silently
  # label a partial Nasdaq response as a full-market scan; fall through to
  # the deterministic CSV ladder instead.  Smaller requested modes may use
  # the dynamic response when at least the mode's minimum is available.
  required={'full':3000,'backup':2000,'backup_2000':2000,'top':500,'top500':500}.get(mode,500)
  if len(dynamic) < required:
   dynamic=[]
  else:
   if mode in ('top500','top'):return dynamic[:500],'dynamic_nasdaq_nyse_market_cap_ranked_top500'
   if mode in ('backup','backup_2000'):return dynamic[:2000],'dynamic_nasdaq_nyse_market_cap_ranked_backup2000'
   return dynamic[:8000],'dynamic_nasdaq_nyse_market_cap_ranked_full'
 root=Path(__file__).resolve().parents[1]/'data'
 names={'full':['universe_full.csv','universe_backup_2000.csv','universe_top500.csv'],'backup':['universe_backup_2000.csv','universe_top500.csv'],'backup_2000':['universe_backup_2000.csv','universe_top500.csv'],'top':['universe_top500.csv'],'top500':['universe_top500.csv']}
 minimums={'universe_full.csv':3000,'universe_backup_2000.csv':2000,'universe_top500.csv':500}
 for name in names.get(mode,names['full']):
  values=_read_csv(root/name,minimums.get(name,1))
  if values:return values,name
 return FALLBACK,'embedded_resilient_fallback'


def load_file_fallback(mode='full'):
 """Mandatory deterministic failover order: full CSV, backup_2000 CSV, top500 CSV."""
 return load_universe(mode,force_file=True)
