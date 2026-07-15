from .indicator_registry import register
@register('event_risk','event_risk')
def calc(df,c):
 e=c.get('event',{}); return {'event_risk':float(e.get('event_risk',0)),'days_to_earnings':float(e.get('days_to_earnings',30))}
