from .indicator_registry import register
@register('intraday_proxy','intraday_proxy')
def calc(df,c): return {'overnight_gap':float(df.Open.iloc[-1]/df.Close.iloc[-2]-1),'intraday_range':float((df.High.iloc[-1]-df.Low.iloc[-1])/df.Close.iloc[-1])}
