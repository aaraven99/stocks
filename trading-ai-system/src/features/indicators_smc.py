from .indicator_registry import register
@register('smc','smc_proxy')
def calc(df,c):
 h=df.High.rolling(20).max().iloc[-1]; l=df.Low.rolling(20).min().iloc[-1]; p=df.Close.iloc[-1]; return {'smc_range_position':float((p-l)/max(h-l,1e-9)),'smc_breakout':float(p>=h),'smc_displacement':float(df.Close.pct_change().tail(3).sum())}
