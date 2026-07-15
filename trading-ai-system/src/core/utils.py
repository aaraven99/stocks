import hashlib, json, random
from datetime import date, datetime
import numpy as np
def set_seed(seed=42): random.seed(seed); np.random.seed(seed)
def stable_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def json_default(x):
    if isinstance(x,(datetime,date)): return x.isoformat()
    if hasattr(x,'item'): return x.item()
    return str(x)
def clamp(x,lo=0.,hi=1.): return max(lo,min(hi,float(x)))
