import numpy as np
def explain(features,model=None,n=10,feature_names=None,background=None):
 if model is not None and feature_names is not None:
  try:
   import shap
   values=shap.TreeExplainer(model).shap_values(background)
   row=values[-1] if getattr(values,'ndim',1)>1 else values
   return [{'feature':name,'attribution':float(value),'method':'shap'} for name,value in sorted(zip(feature_names,row),key=lambda x:abs(float(x[1])),reverse=True)[:n]]
  except Exception:
   importance=getattr(model,'feature_importances_',None)
   if importance is not None:
    return [{'feature':name,'attribution':float(value),'method':'feature_importance_fallback'} for name,value in sorted(zip(feature_names,importance),key=lambda x:abs(float(x[1])),reverse=True)[:n]]
 transformed={key:float(np.sign(value)*np.log1p(abs(float(value)))) for key,value in features.items() if np.isfinite(value)};ranked=sorted(transformed.items(),key=lambda x:abs(x[1]),reverse=True)[:n]
 return [{'feature':k,'attribution':v,'method':'signed_log_magnitude_fallback'} for k,v in ranked]
