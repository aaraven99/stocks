import numpy as np
def breadth_snapshot(frames):
 vals=[float((x.Close>x.Close.rolling(50).mean()).iloc[-1]) for x in frames.values() if len(x)>=50]; return {'pct_above_50d':float(np.mean(vals)) if vals else .5,'advance_decline_proxy':float(np.mean(vals)-.5) if vals else 0.}
