"""Deterministic, multi-engine Monte Carlo with pathwise barrier and drawdown statistics."""
import numpy as np
from montecarlo import gbm,bootstrap,garch_sim,regime_switching_sim,copula_sim
from montecarlo.montecarlo_metrics import metrics
from montecarlo.bayesian_weighting import weights
from montecarlo.mixture_distribution import combine
class MonteCarloEngine:
 def __init__(self,sims=25000,seed=42): self.sims=max(1000,int(sims));self.seed=seed
 def _terminal(self,returns,horizon,rng,bear,reliability=None):
  n=self.sims;samples={'gbm':gbm.simulate(returns,n,horizon,rng),'bootstrap':bootstrap.simulate(returns,n,horizon,rng),'garch':garch_sim.simulate(returns,n,horizon,rng),'regime_switching':regime_switching_sim.simulate(returns,n,horizon,rng,bear),'copula':copula_sim.simulate(returns,n,horizon,rng)}
  base={'gbm':.8,'bootstrap':1.0,'garch':.9,'regime_switching':1.0,'copula':.8};scores={key:base[key]*float((reliability or {}).get(key,1.0)) for key in base};w=weights(scores);return combine(samples,w),w,{key:float((value>0).mean()) for key,value in samples.items()}
 def _paths(self,returns,horizon,rng):
  history=np.asarray(returns.dropna(),dtype=float);draws=rng.choice(history,size=(self.sims,horizon),replace=True);return np.cumprod(1+draws,axis=1)-1
 def run(self,returns,regime='sideways_chop',spot=None,stop=.06,target=.10,reliability=None):
  rng=np.random.default_rng(self.seed);bear=regime=='bear_high_vol';distributions={};weights_out=None
  for horizon in (5,10,20):
   dist,w,engine_probs=self._terminal(returns,horizon,rng,bear,reliability);distributions[horizon]=dist;weights_out=w
  paths=self._paths(returns,20,rng);min_path=paths.min(axis=1);max_path=paths.max(axis=1);stop_first=(min_path<=-stop)&((max_path<target)|(np.argmax(paths>=target,axis=1)>=np.argmax(paths<=-stop,axis=1)))
  equity=1+paths;drawdowns=(equity/np.maximum.accumulate(equity,axis=1)-1).min(axis=1)
  m=metrics(distributions[10],stop);m.update({'p_plus_3_5d':float((distributions[5]>=.03).mean()),'p_plus_5_10d':float((distributions[10]>=.05).mean()),'p_plus_10_20d':float((distributions[20]>=.10).mean()),'p_target_before_stop':float(((max_path>=target)&~stop_first).mean()),'p_stop_hit_first':float(stop_first.mean()),'drawdown_q05':float(np.quantile(drawdowns,.05)),'drawdown_q50':float(np.quantile(drawdowns,.50)),'drawdown_q95':float(np.quantile(drawdowns,.95)),'price_target_q50':float((spot or 1)*(1+np.quantile(distributions[10],.50))),'price_target_q90':float((spot or 1)*(1+np.quantile(distributions[10],.90))),'probability_weighted_price_target':float((spot or 1)*(1+np.mean(distributions[10])) )})
  m['engine_probabilities']=engine_probs
  return {'weights':weights_out,'metrics':m,'distribution':distributions[10].tolist(),'horizon_distributions':{str(k):v.tolist() for k,v in distributions.items()},'drawdown_distribution':drawdowns.tolist()}
