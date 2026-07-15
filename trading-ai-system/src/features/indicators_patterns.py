from .indicator_registry import register
@register('patterns','classical_patterns')
def calc(df,c):
 x=df.Close.tail(20); slope=(x.iloc[-1]-x.iloc[0])/max(x.iloc[0],1); return {'pattern_trend_slope':float(slope),'pattern_consolidation':float(x.pct_change().std()),'pattern_breakout':float(df.Close.iloc[-1]>df.High.shift(1).rolling(20).max().iloc[-1])}
