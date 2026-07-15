from .indicator_registry import register
@register('sentiment','sentiment_alternative')
def calc(df,c):
 s=c.get('sentiment',{}); return {'news_sentiment':float(s.get('sentiment',0)),'news_count':float(s.get('news_count',0)),'put_call_ratio':float(c.get('options',{}).get('put_call_ratio',0))}
