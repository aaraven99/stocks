import math
def probability(win_rate, payoff, risk_fraction):
 edge=win_rate*payoff-(1-win_rate); return float(min(1,max(0,math.exp(-max(edge,0)*max(1,1/risk_fraction)))))
