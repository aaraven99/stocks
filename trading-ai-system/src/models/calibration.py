import numpy as np
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def calibration_curve(y,p,bins=10):
 y=np.asarray(y);p=np.asarray(p); out=[]
 for lo in np.linspace(0,.9,bins):
  m=(p>=lo)&(p<lo+.1)
  if m.any():out.append({'prediction':float(p[m].mean()),'outcome':float(y[m].mean()),'count':int(m.sum())})
 return out
