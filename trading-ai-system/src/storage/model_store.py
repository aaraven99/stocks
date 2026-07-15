from datetime import datetime,timezone
class ModelStore:
 def __init__(self,db): self.db=db
 def register(self,model_id,kind,status,metrics,path='',feature_version='v1'):
  self.db.upsert('models',{'model_id':model_id,'kind':kind,'status':status,'trained_at':datetime.now(timezone.utc).isoformat(),'metrics_json':metrics,'path':path,'feature_version':feature_version},['model_id'])
 def champion(self):
  r=self.db.rows("SELECT * FROM models WHERE status='champion' ORDER BY trained_at DESC LIMIT 1"); return r[0] if r else None
