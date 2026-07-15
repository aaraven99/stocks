import numpy as np
def simulate(returns,n,horizon,rng):
 x=np.asarray(returns.dropna()); draws=rng.choice(x,size=(n,horizon),replace=True); return np.prod(1+draws,axis=1)-1
