from .indicator_registry import register
@register('trend','trend')
def calc(df,c):
 close=df.Close; sma20=close.rolling(20).mean().iloc[-1]; sma50=close.rolling(50).mean().iloc[-1]; sma200=close.rolling(200).mean().iloc[-1] if len(close)>=200 else sma50; weekly=close.resample('W-FRI').last().dropna(); weekly_return=float(weekly.iloc[-1]/weekly.iloc[-6]-1) if len(weekly)>=6 else float(close.iloc[-1]/close.iloc[max(0,len(close)-26)]-1); weekly_vol=float(weekly.pct_change().tail(12).std()) if len(weekly)>=3 else 0.; return {'trend_sma20_gap':float(close.iloc[-1]/sma20-1),'trend_sma50_gap':float(close.iloc[-1]/sma50-1),'trend_sma200_gap':float(close.iloc[-1]/sma200-1),'trend_weekly_return_5w':weekly_return,'trend_weekly_volatility':weekly_vol}
