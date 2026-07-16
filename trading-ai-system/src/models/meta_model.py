def score(prediction,mc): return .5*prediction['probability']+.3*max(0,mc['expected_return']*10)+.2*(1-mc['p_stop_hit_first'])
def regime_weighted_score(prediction,mc,regime,archetypes=None):
 base=score(prediction,mc);penalty={'bull_low_vol':0.0,'bull_high_vol':.04,'sideways_chop':.08,'bear_high_vol':.18}.get(regime,.10)
 archetype=max((archetypes or {}).values(),default=.5)
 return float(max(0.,min(1.,base*.80+archetype*.20-penalty)))
