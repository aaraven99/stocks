def assess_kill_switch(drawdown,drift_flag,integrity_ok,limit=.15):
 reasons=[]
 if drawdown<=-limit: reasons.append('drawdown_limit')
 if drift_flag: reasons.append('material_model_drift')
 if not integrity_ok: reasons.append('database_integrity')
 return {'active':bool(reasons),'reason':','.join(reasons) or 'healthy'}
