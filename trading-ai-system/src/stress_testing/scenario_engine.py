from stress_testing.shock_library import SHOCKS
from stress_testing.adversarial_scenarios import generate
def run(ticker,beta=.8,features=None):
 rows=[{'ticker':ticker,'scenario':k,'pnl_pct':v*beta,'penalty':max(0,-v*beta)} for k,v in SHOCKS.items()]
 rows.extend({'ticker':ticker,'scenario':item['scenario'],'pnl_pct':-item['severity'],'penalty':item['severity'],'adversarial':True} for item in generate(features or {}));return rows
