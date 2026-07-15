from .indicator_registry import register
@register('harmonics','harmonics_proxy')
def calc(df,c):
 r=df.Close.pct_change().tail(30); return {'harmonic_retrace_proxy':float(abs(r.tail(5).sum())/max(abs(r.tail(20).sum()),1e-6)),'harmonic_symmetry':float(1-abs(abs(r.tail(10).sum())-abs(r.tail(20).head(10).sum())))}
