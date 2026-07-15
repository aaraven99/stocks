def check_feature_frame(df, target_columns=()):
 findings=[]
 for c in df.columns:
  low=c.lower()
  if any(x in low for x in ['future','forward','target']) and c not in target_columns: findings.append('suspicious_name:'+c)
 if not df.index.is_monotonic_increasing: findings.append('non_monotonic_index')
 return {'passed':not findings,'findings':findings}
