def windows(dates,train=252,test=21,embargo=5):
 for start in range(0,max(0,len(dates)-train-test),test): yield dates[start:start+train],dates[start+train+embargo:start+train+embargo+test]

def eligible_history(dates,current_index,embargo=1):
 """Return signal dates strictly before the embargo window."""
 return set(list(dates)[:max(0,int(current_index)-int(embargo))])
