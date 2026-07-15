from .indicator_registry import register
@register('theory','market_theories')
def calc(df,c):
 r=df.Close.pct_change(); return {'random_walk_zscore':float(r.tail(5).mean()/max(r.tail(20).std(),1e-9)),'reflexivity_proxy':float(r.tail(5).sum()*df.Volume.tail(5).mean()/max(df.Volume.tail(20).mean(),1))}
