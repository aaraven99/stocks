from .indicator_registry import register
@register('breadth','breadth')
def calc(df,c): return {'market_breadth':float(c.get('breadth',{}).get('pct_above_50d',.5)),'advance_decline_proxy':float(c.get('breadth',{}).get('advance_decline_proxy',0))}
