import hashlib
def options_flow_proxy(ticker):
 h=int(hashlib.md5(ticker.encode()).hexdigest()[:6],16); return {'put_call_proxy':0.6+(h%100)/100,'unusual_flow_proxy':(h%100)/100}
