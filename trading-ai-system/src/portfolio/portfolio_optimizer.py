"""Long-only constrained risk-budgeting optimizer using expected return, CVaR proxy, and correlation penalties."""
import numpy as np
from portfolio.correlation_penalty import penalty
from portfolio.sector_constraints import allowed
def optimize(candidates,max_weight=.10,max_positions=12,max_sector=.25):
 selected=[];sector_weights={};series=[]
 for row in sorted(candidates,key=lambda x:x['score'],reverse=True):
  if len(selected)>=max_positions:break
  risk=max(abs(row['mc'].get('cvar_95',-.05)),.01);corr=penalty(row.get('returns',[]),series) if row.get('returns') is not None else 0
  adjusted=row['score']*(1-corr)/risk
  if adjusted<=0 or not allowed(sector_weights,row.get('sector','Unknown'),max_weight,max_sector):continue
  item={**row,'optimizer_score':adjusted};selected.append(item);sector=item.get('sector','Unknown');sector_weights[sector]=sector_weights.get(sector,0)+max_weight;series.append(item.get('returns',[]))
 if not selected:return []
 raw=np.array([x['optimizer_score'] for x in selected]);raw=raw/raw.sum();raw=np.minimum(raw,max_weight);raw=raw/raw.sum()
 return [{**row,'weight':float(weight)} for row,weight in zip(selected,raw)]
