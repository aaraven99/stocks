from .indicator_registry import register
@register('volume','volume')
def calc(df,c):
 v=df.Volume; return {'volume_ratio_20d':float(v.iloc[-1]/max(v.rolling(20).mean().iloc[-1],1)),'obv_proxy':float((df.Close.pct_change().fillna(0)*v).tail(20).sum()/max(v.tail(20).sum(),1)),'dollar_volume':float(df.Close.iloc[-1]*v.iloc[-1])}
