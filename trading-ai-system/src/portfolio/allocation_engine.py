from portfolio.portfolio_optimizer import optimize
from portfolio.kelly import fractional_kelly
from portfolio.risk_budgeting import inverse_vol_weights
from portfolio.capacity_model import capacity
def allocate(candidates,capital,risk_multiplier=1):
 items=optimize(candidates)
 risk_parity=inverse_vol_weights(items)
 out=[]
 for item in items:
  kelly=fractional_kelly(item['prediction']['probability'],max(item['mc']['expected_return'],.03),max(abs(item['mc']['cvar_95']),.03))
  blended=.5*item['weight']+.5*risk_parity.get(item['ticker'],0);capacity_dollars=capacity(item.get('features',{}).get('dollar_volume',0));capacity_weight=capacity_dollars/max(capital,1)
  weight=min(blended,kelly,.10,capacity_weight)*risk_multiplier
  out.append({**item,'weight':weight,'dollars':capital*weight,'kelly_fraction':kelly,'risk_parity_weight':risk_parity.get(item['ticker'],0),'capacity_dollars':capacity_dollars})
 total=sum(x['weight'] for x in out)
 return [{**x,'weight':x['weight']/total*min(total,1)} for x in out] if total else []
