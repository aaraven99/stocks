from storage.model_store import ModelStore
class ModelRegistry(ModelStore):
 def register_champion(self,model_id,kind,metrics,path=''):
  for old in self.db.rows("SELECT model_id FROM models WHERE status='champion'"): self.db.execute("UPDATE models SET status='retired' WHERE model_id=?",(old['model_id'],))
  self.register(model_id,kind,'champion',metrics,path)
 def promote_if_better(self,model_id,kind,metrics,path='',max_brier_degradation=.02):
  old=self.champion();old_brier=None
  if old:
   import json
   try: old_brier=float(json.loads(old.get('metrics_json') or '{}').get('brier',1.0))
   except Exception: old_brier=1.0
  new_brier=float(metrics.get('brier',1.0))
  if old and new_brier>old_brier+max_brier_degradation:
   self.register(model_id,kind,'challenger_rejected',metrics,path)
   return {'decision':'reject','champion_id':old['model_id'],'challenger_id':model_id,'reason':f'challenger_brier_{new_brier:.4f}_worse_than_champion_{old_brier:.4f}'}
  for row in self.db.rows("SELECT model_id FROM models WHERE status='champion'"):
   self.db.execute("UPDATE models SET status='retired' WHERE model_id=?",(row['model_id'],))
  self.register(model_id,kind,'champion',metrics,path)
  return {'decision':'promote','champion_id':model_id,'challenger_id':old['model_id'] if old else 'none','reason':'purged_cv_calibration_and_champion_gate_passed'}
 def rollback_to_latest_retired(self,reason):
  rows=self.db.rows("SELECT model_id FROM models WHERE status='retired' ORDER BY trained_at DESC LIMIT 1")
  if not rows:return {'decision':'rollback_unavailable','reason':reason}
  for row in self.db.rows("SELECT model_id FROM models WHERE status='champion'"):
   self.db.execute("UPDATE models SET status='rolled_back' WHERE model_id=?",(row['model_id'],))
  self.db.execute("UPDATE models SET status='champion' WHERE model_id=?",(rows[0]['model_id'],))
  return {'decision':'rollback','champion_id':rows[0]['model_id'],'reason':reason}
 def rollback_if_breached(self,max_brier=.30,reason='champion_brier_threshold_breached'):
  champion=self.champion()
  if not champion:return {'decision':'no_champion'}
  import json
  try:brier=float(json.loads(champion.get('metrics_json') or '{}').get('brier',0))
  except Exception:brier=0
  if brier<=max_brier:return {'decision':'within_limit','champion_id':champion['model_id'],'brier':brier}
  result=self.rollback_to_latest_retired(reason);result['breached_brier']=brier;return result
