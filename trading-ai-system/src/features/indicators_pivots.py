from .indicator_registry import register
@register('pivots','fibonacci_pivots_structure')
def calc(df,c):
 hi=df.High.tail(60).max(); lo=df.Low.tail(60).min(); p=df.Close.iloc[-1]; return {'fib_position':float((p-lo)/max(hi-lo,1e-9)),'pivot_distance':float(p/((df.High.iloc[-2]+df.Low.iloc[-2]+df.Close.iloc[-2])/3)-1),'structure_high':float(p>=df.High.tail(20).max())}
