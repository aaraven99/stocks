from portfolio.transaction_cost_model import estimate
from portfolio.playbook_execution import execution_plan
from portfolio.trailing_stop_optimizer import stop as optimized_stop
def make_plan(ticker,price,weight,capital,action='LONG',stop_pct=.06,target_pct=.12,atr_pct=None):
 stop_pct=optimized_stop(float(atr_pct),stop_pct) if atr_pct is not None else stop_pct
 shares=int(capital*weight/max(price,1)); p=execution_plan(action,price,stop_pct,target_pct); return {**p,'ticker':ticker,'shares':shares,'transaction_cost':estimate(price,shares)}
