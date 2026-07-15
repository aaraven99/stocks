from .indicator_registry import register
@register('candles','candlesticks')
def calc(df,c):
 row=df.iloc[-1]; body=abs(row.Close-row.Open)/max(row.High-row.Low,1e-9); return {'candle_body_ratio':float(body),'candle_bullish':float(row.Close>row.Open),'candle_upper_wick':float((row.High-max(row.Open,row.Close))/max(row.High-row.Low,1e-9))}
