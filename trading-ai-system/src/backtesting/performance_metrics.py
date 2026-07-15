import numpy as np
def calculate(returns):
 r=np.asarray(returns,dtype=float); equity=np.cumprod(1+r); dd=equity/np.maximum.accumulate(equity)-1; downside=r[r<0]; return {'total_return':float(equity[-1]-1) if len(r) else 0.,'sharpe':float(r.mean()/max(r.std(),1e-9)*np.sqrt(252)) if len(r) else 0.,'sortino':float(r.mean()/max(downside.std() if len(downside) else 1e-9,1e-9)*np.sqrt(252)) if len(r) else 0.,'max_drawdown':float(dd.min()) if len(r) else 0.,'win_rate':float((r>0).mean()) if len(r) else 0.}
