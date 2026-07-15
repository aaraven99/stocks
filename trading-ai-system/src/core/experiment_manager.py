from storage.experiment_store import ExperimentStore
class ExperimentManager:
 def __init__(self,db): self.store=ExperimentStore(db); self.db=db
 def evaluate(self,ident,name,params,metrics,threshold=.0):
  accepted=metrics.get('sharpe',metrics.get('score',-1))>threshold; self.store.record(ident,name,'accepted' if accepted else 'rejected',params,metrics); return accepted
 def promote(self,date,champion,challenger,reason): self.db.upsert('champion_challenger',{'date':date,'champion_id':champion,'challenger_id':challenger,'decision':'promote' if challenger else 'retain','reason':reason},['date','challenger_id'])
