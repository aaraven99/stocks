from .indicator_registry import register
@register('cross_asset','cross_asset')
def calc(df,c):
 a=c.get('cross_asset',{}); spy=a.get('SPY',{}); vix=a.get('^VIX',{}); return {'spy_return_5d':float(spy.get('return_5d',0)),'vix_volatility':float(vix.get('vol_20d',.2)),'cross_asset_risk':float(max(vix.get('return_5d',0),0))}
