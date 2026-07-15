import numpy as np
def penalty(returns,selected):
 if not selected:return 0.
 corrs=[abs(returns.corr(x)) for x in selected if len(x)==len(returns)]; return float(np.mean(corrs)) if corrs else 0.
