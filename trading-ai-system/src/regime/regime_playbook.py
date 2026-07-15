def playbook(regime):
 return {'bull_low_vol':{'risk_multiplier':1.,'min_score':.52},'bull_high_vol':{'risk_multiplier':.7,'min_score':.58},'sideways_chop':{'risk_multiplier':.45,'min_score':.62},'bear_high_vol':{'risk_multiplier':.15,'min_score':.70}}[regime]
