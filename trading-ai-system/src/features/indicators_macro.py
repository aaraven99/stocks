from .indicator_registry import register
@register('macro','macro')
def calc(df,c):
 m=c.get('macro',{}); return {'rates_return':float(m.get('rates_return',0)),'inflation_rate':float(m.get('inflation',0)),'credit_spread':float(m.get('credit_spread',0))}
