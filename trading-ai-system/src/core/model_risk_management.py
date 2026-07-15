def mrm_decision(kill,calibration_brier,drift_psi):
 if kill.get('active'): return {'decision':'halt_risk','reason':kill['reason'],'risk_multiplier':0.0}
 if calibration_brier>.30 or drift_psi>.25:return {'decision':'derisk','reason':'calibration_or_drift','risk_multiplier':.5}
 return {'decision':'normal','reason':'within_limits','risk_multiplier':1.0}
