REGISTRY={};DERIVED={}
def register(name,category):
 def deco(fn): REGISTRY[name]={'fn':fn,'category':category}; return fn
 return deco
def definitions():
 items=DERIVED or {k:v['category'] for k,v in REGISTRY.items()}
 return {name:{'category':category,'description':name.replace('_',' '),'provenance':'causal OHLCV and explicitly supplied as-of context'} for name,category in items.items()}
def calculate_all(df,context=None):
 out={}
 for name,item in REGISTRY.items():
  try:
   values=item['fn'](df,context or {})
   for feature,value in values.items():out[feature]=value;DERIVED[feature]=item['category']
  except Exception: out[name]=0.0
 return out
