import hashlib
def sentiment_snapshot(ticker):
 h=int(hashlib.sha256(ticker.encode()).hexdigest()[:8],16); return {'sentiment':((h%200)-100)/500,'news_count':h%25,'source':'deterministic_proxy'}
