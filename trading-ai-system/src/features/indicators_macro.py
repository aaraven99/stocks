from .indicator_registry import register
@register('macro','macro')
def calc(df,c):
 m=c.get('macro',{}); return {'rates_proxy_return':float(m.get('rates_proxy_return',0)),'inflation_proxy':float(m.get('inflation_proxy',.03)),'credit_proxy':float(m.get('credit_proxy',0))}
