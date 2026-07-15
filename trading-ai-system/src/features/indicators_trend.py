from .indicator_registry import register
@register('trend','trend')
def calc(df,c):
 close=df.Close; sma20=close.rolling(20).mean().iloc[-1]; sma50=close.rolling(50).mean().iloc[-1]; sma200=close.rolling(200).mean().iloc[-1] if len(close)>=200 else sma50; return {'trend_sma20_gap':float(close.iloc[-1]/sma20-1),'trend_sma50_gap':float(close.iloc[-1]/sma50-1),'trend_sma200_gap':float(close.iloc[-1]/sma200-1),'trend_weekly_proxy':float(close.iloc[-1]/close.iloc[-5]-1)}
