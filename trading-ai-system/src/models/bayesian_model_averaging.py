def average(models):
 z=sum(max(m.get('weight',1),0) for m in models) or 1; return sum(m.get('probability',.5)*max(m.get('weight',1),0) for m in models)/z
