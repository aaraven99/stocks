def estimate(price,shares,spread_bps=8,impact_bps=4): return abs(price*shares)*(spread_bps+impact_bps)/10000
