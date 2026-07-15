import numpy as np
def combine(samples,weights):
 keys=list(samples); n=min(len(samples[k]) for k in keys); out=np.zeros(n)
 for k in keys: out+=weights[k]*samples[k][:n]
 return out
