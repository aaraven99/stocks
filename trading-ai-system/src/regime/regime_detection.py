import numpy as np
from regime.hmm_regime import hmm_proxy
from regime.clustering_regime import cluster_proxy
def rules_regime(spy, vix):
 r=float(spy.Close.pct_change(20).iloc[-1]); vol=float(spy.Close.pct_change().tail(20).std()*252**.5); v=float(vix.Close.pct_change().tail(20).std()*252**.5) if vix is not None else vol
 if r<-.03 or vol>.35:return 'bear_high_vol'
 if r>.02 and vol<.22:return 'bull_low_vol'
 if r>.02:return 'bull_high_vol'
 return 'sideways_chop'
def detect(spy,vix):
 returns=spy.Close.pct_change().dropna(); regime=rules_regime(spy,vix); hmm=hmm_proxy(returns); cluster=cluster_proxy(returns)
 agreement=int((hmm['state']==2 and regime.startswith('bull')) or (hmm['state']==0 and regime=='bear_high_vol') or (hmm['state']==1 and regime=='sideways_chop'))
 return {'regime':regime,'confidence':.55+.20*agreement,'return_20d':float(spy.Close.pct_change(20).iloc[-1]),'volatility':float(spy.Close.pct_change().tail(20).std()*252**.5),'hmm':hmm,'cluster':cluster,'method':'rules+hmm_proxy+clustering_proxy'}
