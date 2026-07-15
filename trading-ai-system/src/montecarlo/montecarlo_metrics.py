import numpy as np
def metrics(dist,stop=.06):
 q=np.quantile(dist,[.01,.05,.1,.5,.9,.95,.99]); return {'p_plus_3':float((dist>=.03).mean()),'p_plus_5':float((dist>=.05).mean()),'p_plus_10':float((dist>=.10).mean()),'p_stop_hit_first':float((dist<=-stop).mean()),'expected_return':float(dist.mean()),'var_95':float(q[1]),'cvar_95':float(dist[dist<=q[1]].mean() if (dist<=q[1]).any() else q[1]),'q01':float(q[0]),'q10':float(q[2]),'median':float(q[3]),'q90':float(q[4]),'q99':float(q[6]),'drawdown_proxy':float(q[1]),'r_expectancy':float(dist.mean()/max(abs(q[1]),.001))}
