import numpy as np
def hmm_proxy(returns):
 """Gaussian HMM when hmmlearn is installed; deterministic fallback is explicitly identified."""
 x=np.asarray(returns.dropna(),dtype=float).reshape(-1,1)
 if len(x)<60:return {'state':1,'probabilities':[0.,1.,0.],'method':'insufficient_history'}
 try:
  from hmmlearn.hmm import GaussianHMM
  model=GaussianHMM(n_components=3,covariance_type='diag',n_iter=200,random_state=42).fit(x);posterior=model.predict_proba(x)[-1];means=model.means_.ravel();ordered=np.argsort(means);state=int(np.where(ordered==np.argmax(posterior))[0][0]);return {'state':state,'probabilities':posterior.tolist(),'means':means.tolist(),'method':'gaussian_hmm'}
 except ImportError:
  q=np.quantile(x.ravel(),[.33,.67]);last=float(x[-1,0]);state=0 if last<q[0] else 2 if last>q[1] else 1;return {'state':state,'probabilities':[float(state==0),float(state==1),float(state==2)],'method':'quantile_fallback_hmmlearn_unavailable'}
