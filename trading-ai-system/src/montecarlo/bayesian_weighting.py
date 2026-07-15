def weights(validation_scores):
 s={k:max(float(v),1e-6) for k,v in validation_scores.items()}; z=sum(s.values()); return {k:v/z for k,v in s.items()}
