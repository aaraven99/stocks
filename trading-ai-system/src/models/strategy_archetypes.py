"""Transparent strategy archetype scores persisted beside quantitative decisions."""
def evaluate(features,regime):
 trend=.5+.8*float(features.get('trend_sma50_gap',0))+.4*float(features.get('trend_weekly_return_5w',0))
 mean_reversion=.5-.7*float(features.get('momentum_5d',0))-.3*float(features.get('rsi14',.5)-.5)
 breakout=.5+.8*float(features.get('pattern_breakout',0))+.3*float(features.get('volume_ratio_20d',1)-1)
 if regime.startswith('bear'): trend*=.6;breakout*=.6
 return {'trend_following':max(0.,min(1.,trend)),'mean_reversion':max(0.,min(1.,mean_reversion)),'breakout':max(0.,min(1.,breakout))}
