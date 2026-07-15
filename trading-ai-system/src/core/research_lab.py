import numpy as np
def discover_signal(frame):
 r=frame['Close'].pct_change(); signals={'reversal_5d':-r.rolling(5).sum(),'trend_20d':r.rolling(20).sum(),'vol_adjusted_momentum':r.rolling(10).mean()/r.rolling(10).std()}
 out=[]
 for name,s in signals.items():
  score=float(s.corr(r.shift(-5))) if s.notna().sum()>30 else 0.; out.append({'signal_name':name,'formula':name,'score':score})
 return out
