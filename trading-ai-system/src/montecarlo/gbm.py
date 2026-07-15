import numpy as np
def simulate(returns,n,horizon,rng):
 mu=returns.mean(); sig=max(returns.std(),1e-5); return np.exp(np.cumsum(rng.normal(mu,sig,(n,horizon)),axis=1))[:,-1]-1
