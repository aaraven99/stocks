def score(prediction,mc): return .5*prediction['probability']+.3*max(0,mc['expected_return']*10)+.2*(1-mc['p_stop_hit_first'])
def regime_weighted_score(prediction,mc,regime,archetypes=None):
 base=score(prediction,mc); multiplier={'bull_low_vol':1.0,'bull_high_vol':.8,'sideways_chop':.6,'bear_high_vol':.3}.get(regime,.5)
 archetype=max((archetypes or {}).values(),default=.5)
 return float(max(0.,min(1.,base*multiplier*.75+archetype*.25)))
