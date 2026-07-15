def score(prediction,mc): return .5*prediction['probability']+.3*max(0,mc['expected_return']*10)+.2*(1-mc['p_stop_hit_first'])
