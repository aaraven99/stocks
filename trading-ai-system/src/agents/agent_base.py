class Agent:
 name='agent'; stance='neutral'
 def __init__(self,db=None):self.db=db
 def assess(self,context): return {'agent':self.name,'stance':self.stance,'score':.5,'reasoning':'Evidence-based assessment unavailable.'}
 def persist(self,date,payload):
  if self.db:self.db.upsert('agent_outputs',{'date':date,'agent':self.name,'payload_json':payload},['date','agent'])
