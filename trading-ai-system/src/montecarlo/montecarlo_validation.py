def validate(metrics):
 findings=[]
 if metrics['q01']>metrics['q10']:findings.append('quantile_order')
 if not 0<=metrics['p_plus_3']<=1:findings.append('probability_bounds')
 return {'passed':not findings,'findings':findings}
