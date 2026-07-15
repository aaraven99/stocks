from portfolio.portfolio_optimizer import optimize
from portfolio.kelly import fractional_kelly
def allocate(candidates,capital,risk_multiplier=1):
 items=optimize(candidates)
 out=[]
 for item in items:
  kelly=fractional_kelly(item['prediction']['probability'],max(item['mc']['expected_return'],.03),max(abs(item['mc']['cvar_95']),.03))
  weight=min(item['weight'],kelly,.10)*risk_multiplier
  out.append({**item,'weight':weight,'dollars':capital*weight,'kelly_fraction':kelly})
 total=sum(x['weight'] for x in out)
 return [{**x,'weight':x['weight']/total*min(total,1)} for x in out] if total else []
