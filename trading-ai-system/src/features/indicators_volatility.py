from .indicator_registry import register
@register('volatility','volatility')
def calc(df,c):
 r=df.Close.pct_change(); atr=(df.High-df.Low).rolling(14).mean().iloc[-1]/df.Close.iloc[-1]; return {'volatility_20d':float(r.tail(20).std()*(252**.5)),'atr_pct':float(atr),'downside_vol':float(r.tail(20).clip(upper=0).std()*(252**.5))}
