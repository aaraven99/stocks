import numpy as np
def exposures(stock,market):
 b=float(np.cov(stock,market)[0,1]/max(np.var(market),1e-9)); a=float(np.mean(stock)-b*np.mean(market)); return {'alpha':a,'beta':b}
