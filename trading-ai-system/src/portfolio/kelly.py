def fractional_kelly(probability,win=.10,loss=.06,fraction=.25): return max(0.,fraction*(probability*win-(1-probability)*loss)/max(win*loss,1e-9))
