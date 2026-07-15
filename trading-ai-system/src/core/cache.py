import json, time
from pathlib import Path
class DiskCache:
 def __init__(self,path='.cache',ttl=21600): self.path=Path(path); self.path.mkdir(parents=True,exist_ok=True); self.ttl=ttl
 def get(self,key):
  p=self.path/(key+'.json')
  if not p.exists() or time.time()-p.stat().st_mtime>self.ttl:return None
  try:return json.loads(p.read_text())
  except Exception:return None
 def set(self,key,value): (self.path/(key+'.json')).write_text(json.dumps(value,default=str))
