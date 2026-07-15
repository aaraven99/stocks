from agents.agent_base import Agent
class TechnicalAgent(Agent):
 name='TechnicalAgent'; stance='bull'
 def assess(self,context):
  regime=context.get('regime','sideways_chop'); risk=context.get('risk',.5); score=max(.05,min(.95,.58-risk*.3+(.10 if self.stance=='bull' and regime.startswith('bull') else 0)-(.10 if self.stance=='bear' and regime=='bear_high_vol' else 0))); return {'agent':self.name,'stance':self.stance,'score':score,'reasoning':self.name+' '+self.stance+' assessment uses regime='+regime+', quantified risk='+format(risk,'.2f')+', and retains uncertainty.'}
