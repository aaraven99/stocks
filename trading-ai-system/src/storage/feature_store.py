from datetime import datetime,timezone
from core.utils import stable_hash
class FeatureStore:
 def __init__(self,db): self.db=db
 def register(self,definitions,version='v1'):
  for name,meta in definitions.items(): self.db.upsert('feature_definitions',{'name':name,'category':meta.get('category','derived'),'description':meta.get('description',name),'provenance':meta.get('provenance','ohlcv'),'version_hash':stable_hash([name,version]),'created_at':datetime.now(timezone.utc).isoformat()},['name'])
 def put(self,ticker,date,values,version='v1',sources='ohlcv'):
  self.db.upsert('features',{'ticker':ticker,'date':date,'feature_version':version,'values_json':values,'computed_at':datetime.now(timezone.utc).isoformat(),'source_refs':sources},['ticker','date','feature_version'])
