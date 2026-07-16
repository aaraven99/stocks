def weights(validation_scores,prior=.5,effective_observations=20):
 posterior={key:float(prior)+float(effective_observations)*max(0,min(1,float(score))) for key,score in validation_scores.items()};total=sum(posterior.values()) or 1.;return {key:value/total for key,value in posterior.items()}
