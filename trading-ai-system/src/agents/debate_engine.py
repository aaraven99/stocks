class DebateEngine:
 def __init__(self,db,agents):self.db=db;self.agents=agents
 def run(self,date,context):
  rounds=['thesis','rebuttal','cross_exam','red_team_attack','adversarial_stress_verdict','risk_committee_veto','final_verdict']; transcript=[]
  for n,label in enumerate(rounds,1):
   for a in self.agents:
    x=a.assess(dict(context,round=label)); x['round']=label; transcript.append(x);self.db.upsert('debate_rounds',{'date':date,'round_no':n,'agent':a.name,'stance':x['stance'],'content':x['reasoning'],'score':x['score']},['date','round_no','agent']);self.db.upsert('agent_outputs',{'date':date,'agent':a.name,'payload_json':x},['date','agent'])
  verdict={'bull_score':sum(x['score'] for x in transcript if x['stance']=='bull'),'bear_score':sum(x['score'] for x in transcript if x['stance']=='bear'),'rounds':7}; bull_count=max(1,sum(x['stance']=='bull' for x in transcript));bear_count=max(1,sum(x['stance']=='bear' for x in transcript)); verdict['action']='CASH' if context.get('risk',.5)>.55 or verdict['bear_score']/bear_count>verdict['bull_score']/bull_count+.12 else 'WATCHLIST_LONGS';self.db.upsert('agents_transcripts',{'date':date,'transcript_json':transcript,'verdict_json':verdict},['date']);return verdict,transcript
