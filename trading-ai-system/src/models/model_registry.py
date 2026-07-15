from storage.model_store import ModelStore
class ModelRegistry(ModelStore):
 def register_champion(self,model_id,kind,metrics,path=''):
  for old in self.db.rows("SELECT model_id FROM models WHERE status='champion'"): self.db.execute("UPDATE models SET status='retired' WHERE model_id=?",(old['model_id'],))
  self.register(model_id,kind,'champion',metrics,path)
