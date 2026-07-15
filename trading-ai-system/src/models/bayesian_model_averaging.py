def average(models):
 z=sum(max(m.get('weight',1),0) for m in models) or 1; return sum(m.get('probability',.5)*max(m.get('weight',1),0) for m in models)/z
def update_weights(models,observed):
 scores=[]
 for model in models:
  p=float(model.get('probability',.5));scores.append(max(1e-6,1-(p-int(observed))**2))
 total=sum(scores) or 1.;return [{**model,'weight':score/total} for model,score in zip(models,scores)]
