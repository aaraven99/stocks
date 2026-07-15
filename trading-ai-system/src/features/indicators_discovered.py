from .indicator_registry import register
@register('discovered','research_lab')
def calc(df,c):
 r=df.Close.pct_change(); return {'discovered_reversal_5d':float(-r.tail(5).sum()),'discovered_vol_adj_mom':float(r.tail(10).mean()/max(r.tail(10).std(),1e-9))}
