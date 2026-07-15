from .indicator_registry import register
@register('orderflow_proxy','volume_orderflow')
def calc(df,c):
 rng=(df.High-df.Low).replace(0,1e-9); clv=((df.Close-df.Low)-(df.High-df.Close))/rng; return {'close_location_value':float(clv.iloc[-1]),'money_flow_proxy':float((clv*df.Volume).tail(10).sum()/max(df.Volume.tail(10).sum(),1))}
