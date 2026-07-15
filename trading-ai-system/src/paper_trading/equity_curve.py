def drawdown(equities):
 peak=0; out=[]
 for x in equities: peak=max(peak,x); out.append(x/peak-1 if peak else 0)
 return out
