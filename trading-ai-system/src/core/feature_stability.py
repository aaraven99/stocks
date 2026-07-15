import numpy as np
def psi(expected,actual,bins=10):
 e=np.asarray(expected); a=np.asarray(actual); edges=np.unique(np.quantile(e,np.linspace(0,1,bins+1))); 
 if len(edges)<3:return 0.0
 ep=np.histogram(e,edges)[0]/max(len(e),1); ap=np.histogram(a,edges)[0]/max(len(a),1); ep=np.clip(ep,1e-6,None); ap=np.clip(ap,1e-6,None); return float(np.sum((ap-ep)*np.log(ap/ep)))
def stability(expected,actual):
 p=psi(expected,actual); return {'psi':p,'stable':p<.2}
