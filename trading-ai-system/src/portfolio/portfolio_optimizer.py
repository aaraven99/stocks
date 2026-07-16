"""Long-only constrained risk-budgeting optimizer using expected return, CVaR proxy, and correlation penalties."""
import numpy as np
from portfolio.correlation_penalty import penalty
from portfolio.sector_constraints import allowed

def mean_variance_score(expected_return,volatility,risk_aversion=3.0):
 """Small, bounded mean-variance utility used as an optimizer input."""
 mu=float(expected_return);variance=max(float(volatility),0.0)**2
 return float(mu-float(risk_aversion)*variance)

def optimize(candidates,max_weight=.10,max_positions=12,max_sector=.25):
 selected=[];sector_weights={};series=[]
 for row in sorted(candidates,key=lambda x:x['score'],reverse=True):
  if len(selected)>=max_positions:break
  risk=max(abs(row['mc'].get('cvar_95',-.05)),.01);corr=penalty(row.get('returns',[]),series) if row.get('returns') is not None else 0
  mv=mean_variance_score(row['mc'].get('expected_return',0.0),row.get('volatility',row.get('features',{}).get('volatility_20d',.2)))
  adjusted=max(0.0,(row['score']+.10*mv)*(1-corr)/risk)
  if adjusted<=0 or not allowed(sector_weights,row.get('sector','Unknown'),max_weight,max_sector):continue
  item={**row,'optimizer_score':adjusted,'mean_variance_score':mv};selected.append(item);sector=item.get('sector','Unknown');sector_weights[sector]=sector_weights.get(sector,0)+max_weight;series.append(item.get('returns',[]))
 if not selected:return []
 raw=np.array([x['optimizer_score'] for x in selected],dtype=float);raw=raw/raw.sum()
 # A capped long-only projection. Do not renormalize above the per-name cap.
 for _ in range(len(raw)+1):
  capped=raw>=max_weight
  if not capped.any():break
  raw[capped]=max_weight;remaining=max(0.0,1.0-raw[capped].sum());uncapped=~capped
  if not uncapped.any() or raw[uncapped].sum()<=remaining:break
  raw[uncapped]=raw[uncapped]/raw[uncapped].sum()*remaining
 return [{**row,'weight':float(weight)} for row,weight in zip(selected,raw)]
