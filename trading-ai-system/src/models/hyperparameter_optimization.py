from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def candidate_grid():
 return [
  {'family':'logistic','C':c,'class_weight':'balanced'}
  for c in [.25,1.0,4.0]
 ] + [
  {'family':'random_forest','n_estimators':n,'max_depth':d,'min_samples_leaf':leaf,'class_weight':'balanced'}
  for n in [200,400] for d in [4,7] for leaf in [3,8]
 ] + [
  {'family':'xgboost','n_estimators':n,'max_depth':d,'learning_rate':lr,'subsample':.85,'colsample_bytree':.85}
  for n in [150,300] for d in [2,3] for lr in [.03,.08]
 ]


def build_estimator(params, random_state=42):
 family=params['family']
 if family=='logistic':
  return LogisticRegression(max_iter=1000,C=params['C'],class_weight=params['class_weight'],random_state=random_state)
 if family=='random_forest':
  return RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],class_weight=params['class_weight'],random_state=random_state,n_jobs=1)
 if family=='xgboost':
  try:
   from xgboost import XGBClassifier
  except Exception:
   return None
  return XGBClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],learning_rate=params['learning_rate'],subsample=params['subsample'],colsample_bytree=params['colsample_bytree'],eval_metric='logloss',tree_method='hist',random_state=random_state,n_jobs=1)
 raise ValueError(f'unknown model family: {family}')
