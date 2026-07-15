import numpy as np
def ensemble_uncertainty(probabilities): return float(np.std(probabilities))
def confidence(probability,uncertainty): return float(max(0,min(1,1-uncertainty*3))*abs(probability-.5)*2)

def conformal_interval(calibration_truth, calibration_probability, alpha=.10):
 y=np.asarray(calibration_truth,dtype=float);p=np.asarray(calibration_probability,dtype=float)
 if len(y)==0 or len(y)!=len(p):return {'alpha':alpha,'qhat':1.0,'lower':0.0,'upper':1.0,'method':'unavailable'}
 scores=np.abs(y-p);q=float(np.quantile(scores,min(1,(1-alpha)*(len(scores)+1)/max(len(scores),1))))
 return {'alpha':alpha,'qhat':q,'lower':None,'upper':None,'method':'split_conformal_abs_probability_error'}

def apply_conformal(probability, conformal):
 q=float(conformal.get('qhat',1.0))
 return {'lower':float(max(0,probability-q)),'upper':float(min(1,probability+q)),'qhat':q,'alpha':conformal.get('alpha',.10)}
