def validate_probability(row):
 findings=[]
 for key in ['probability','target_before_stop']:
  if key in row and not 0<=float(row[key])<=1: findings.append(key+'_out_of_bounds')
 if row.get('expected_return',0)>0 and row.get('probability',.5)<.05: findings.append('return_probability_tension')
 return {'passed':not findings,'findings':findings}
