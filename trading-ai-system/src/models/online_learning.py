def update_weight(weight,reward,rate=.05): return max(.1,min(2.,weight*(1+rate*reward)))
