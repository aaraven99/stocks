from stress_testing.shock_library import SHOCKS
def run(ticker,beta=.8): return [{'ticker':ticker,'scenario':k,'pnl_pct':v*beta,'penalty':max(0,-v*beta)} for k,v in SHOCKS.items()]
