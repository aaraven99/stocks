import numpy as np
def simulate(returns,n,horizon,rng,bear=False):
 mu=returns.mean()*(.25 if bear else 1); sig=max(returns.std()*(1.6 if bear else 1),1e-5); states=rng.random((n,horizon))<(.22 if bear else .08); r=rng.normal(mu,sig,(n,horizon)); r[states]-=sig; return np.prod(1+r,axis=1)-1
