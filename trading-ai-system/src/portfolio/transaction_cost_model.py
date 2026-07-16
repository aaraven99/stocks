"""Transparent transaction-cost estimate for planning and capacity checks."""

def estimate(price,shares,spread_bps=8,impact_bps=4,commission_per_share=0.0,minimum_commission=0.0):
 notional=abs(float(price)*float(shares))
 market_cost=notional*(float(spread_bps)+float(impact_bps))/10000.0
 commission=max(float(minimum_commission),abs(float(shares))*float(commission_per_share)) if shares else 0.0
 return market_cost+commission
