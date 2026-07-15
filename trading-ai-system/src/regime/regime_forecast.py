def forecast(regime,volatility,hmm=None,cluster=None):
 bear={'bull_low_vol':.12,'bull_high_vol':.27,'sideways_chop':.35,'bear_high_vol':.68}.get(regime,.35)
 if hmm and hmm.get('state')==0: bear=min(.95,bear+.10)
 if cluster and cluster.get('volatility',0)>.02: bear=min(.95,bear+.05)
 return {'flip_bearish_5d':bear,'vol_spike_5d':min(.9,.15+volatility),'method':'rules_hmm_cluster_ensemble'}
