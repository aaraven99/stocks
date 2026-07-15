import hashlib
def event_risk(ticker):
 days=int(hashlib.sha1(ticker.encode()).hexdigest()[:4],16)%30; return {'days_to_earnings_proxy':days,'event_risk':float(days<=5)}
