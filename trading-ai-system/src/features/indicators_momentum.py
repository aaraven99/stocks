from .indicator_registry import register
@register('momentum','momentum')
def calc(df,c):
 r=df.Close.pct_change(); up=r.clip(lower=0).rolling(14).mean(); dn=-r.clip(upper=0).rolling(14).mean(); rsi=100-100/(1+up.iloc[-1]/max(dn.iloc[-1],1e-9)); return {'momentum_5d':float(df.Close.pct_change(5).iloc[-1]),'momentum_20d':float(df.Close.pct_change(20).iloc[-1]),'rsi14':float(rsi/100),'macd_proxy':float(df.Close.ewm(span=12).mean().iloc[-1]/df.Close.ewm(span=26).mean().iloc[-1]-1)}
