def shrink(value,sector_mean,n): return (value*n+sector_mean*20)/(n+20)
def shrink_probabilities(probabilities,prior=.5):
 n=len(probabilities);return [(float(p)*n+prior*20)/(n+20) for p in probabilities]
