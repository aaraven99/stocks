import numpy as np


def calculate(returns):
 r=np.asarray(returns,dtype=float);equity=np.cumprod(1+r)
 if not len(r):return {'total_return':0.0,'sharpe':0.0,'sortino':0.0,'max_drawdown':0.0,'win_rate':0.0,'observations':0}
 drawdown=equity/np.maximum.accumulate(equity)-1;downside=r[r<0]
 return {'total_return':float(equity[-1]-1),'sharpe':float(r.mean()/max(r.std(ddof=1) if len(r)>1 else 0,1e-9)*np.sqrt(252)),'sortino':float(r.mean()/max(downside.std(ddof=1) if len(downside)>1 else 0,1e-9)*np.sqrt(252)),'max_drawdown':float(drawdown.min()),'win_rate':float((r>0).mean()),'observations':int(len(r))}
