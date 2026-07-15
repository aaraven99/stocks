REGISTRY={}
def register(name,category):
 def deco(fn): REGISTRY[name]={'fn':fn,'category':category}; return fn
 return deco
def definitions(): return {k:{'category':v['category'],'description':k,'provenance':'OHLCV/cross_asset'} for k,v in REGISTRY.items()}
def calculate_all(df,context=None):
 out={}
 for name,item in REGISTRY.items():
  try: out.update(item['fn'](df,context or {}))
  except Exception: out[name]=0.0
 return out
