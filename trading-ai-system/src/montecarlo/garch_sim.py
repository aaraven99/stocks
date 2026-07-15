import numpy as np
def simulate(returns,n,horizon,rng):
 x=np.asarray(returns.dropna()); var=np.var(x); alpha=.08; beta=.88; eps=rng.normal(size=(n,horizon)); out=np.zeros((n,horizon));
 for t in range(horizon):
  var=.000001+alpha*(out[:,t-1]**2 if t else var)+beta*var; out[:,t]=eps[:,t]*np.sqrt(var)
 return np.prod(1+out,axis=1)-1
