from datetime import datetime,timezone
class ExperimentStore:
 def __init__(self,db): self.db=db
 def record(self,ident,name,status,params,metrics): self.db.upsert('experiments',{'experiment_id':ident,'name':name,'status':status,'started_at':datetime.now(timezone.utc).isoformat(),'ended_at':datetime.now(timezone.utc).isoformat(),'params_json':params,'metrics_json':metrics},['experiment_id'])
