import numpy as np
def simulate(returns,n,horizon,rng,market_corr=.4):
 mu=returns.mean(); sig=max(returns.std(),1e-5); market=rng.normal(0,sig,(n,horizon)); own=rng.normal(mu,sig,(n,horizon)); r=market_corr*market+(1-market_corr**2)**.5*own; return np.prod(1+r,axis=1)-1
