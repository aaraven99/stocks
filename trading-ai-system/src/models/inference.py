"""Inference is model-backed when a governed champion is available; cold start is explicitly labelled."""
from pathlib import Path
import numpy as np
from core.constants import ARTIFACTS
from models.uncertainty import ensemble_uncertainty,confidence
from models.explainability import explain
_CACHE=None
def _bundle():
 global _CACHE
 if _CACHE is not None:return _CACHE
 path=ARTIFACTS/'champion_model.joblib'
 if not path.exists():_CACHE={};return _CACHE
 try:
  import joblib;_CACHE=joblib.load(path);return _CACHE
 except Exception:_CACHE={};return _CACHE
def predict(features):
 bundle=_bundle();model_id='rule_based_cold_start';probs=[];attributions=[]
 if bundle:
  names=bundle['feature_names'];x=np.array([[float(features.get(k,0.0)) for k in names]])
  for model in bundle['models']:
   try:probs.append(float(model.predict_proba(x)[0,1]))
   except Exception:continue
  if probs:
   model_id=bundle['model_id'];importance=np.asarray(bundle.get('importance',np.ones(len(names))))
   attributions=[{'feature':k,'attribution':float(v*x[0,i])} for i,(k,v) in enumerate(zip(names,importance))]
 if not probs:
  z=.8*features.get('momentum_20d',0)+.6*features.get('trend_sma50_gap',0)-.7*features.get('volatility_20d',.2)+.2*features.get('news_sentiment',0)
  probs=[float(1/(1+np.exp(-z)))];attributions=explain(features)
 p=float(np.mean(probs));u=ensemble_uncertainty(probs) if len(probs)>1 else .18
 return {'model_id':model_id,'probability':p,'expected_return':float(features.get('momentum_20d',0)*.35+features.get('trend_sma50_gap',0)*.2),'expected_drawdown':-abs(float(features.get('volatility_20d',.2)))*.08,'target_before_stop':float(max(0,min(1,p*(1-features.get('event_risk',0)*.2)))),'uncertainty':u,'confidence':confidence(p,u),'explanation':sorted(attributions,key=lambda x:abs(x['attribution']),reverse=True)[:10]}
