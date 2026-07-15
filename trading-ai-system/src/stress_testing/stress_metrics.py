import numpy as np
def summarize(outputs): return {'worst_case':float(min((x['pnl_pct'] for x in outputs),default=0)),'mean_shock':float(np.mean([x['pnl_pct'] for x in outputs])) if outputs else 0.,'penalty':float(np.mean([x['penalty'] for x in outputs])) if outputs else 0.}
