"""Weekly causal retraining from matured labels only; no synthetic labels or future features."""
import json,sys,uuid,shutil
from datetime import datetime,timezone,date
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

from models.calibration import brier,calibration_curve
from models.hyperparameter_optimization import build_estimator,candidate_grid
from models.training import purged_splits
from models.uncertainty import conformal_interval
from storage.artifacts import ensure_artifacts,write_json,write_text
from storage.db import Database
from models.model_registry import ModelRegistry
from models.hierarchical_bayes import shrink_probabilities


def _now():
 return datetime.now(timezone.utc).isoformat()


def _load_training_frame(db):
 rows=db.rows("SELECT f.values_json,l.positive FROM features f JOIN labels l ON l.ticker=f.ticker AND l.asof_date=f.date WHERE l.horizon_days=5 ORDER BY f.date,f.ticker")
 parsed=[json.loads(r['values_json']) for r in rows]
 if not parsed:return rows,[],np.empty((0,0)),np.empty((0,))
 names=sorted(set.intersection(*(set(x) for x in parsed)))[:80]
 X=np.nan_to_num(np.array([[float(x.get(n,0.0)) for n in names] for x in parsed],dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
 y=np.asarray([int(r['positive']) for r in rows],dtype=int)
 return rows,names,X,y


def _score_candidate(params,X,y):
 estimator=build_estimator(params)
 if estimator is None:return None
 folds=[]
 for train,test in purged_splits(len(X),embargo=5):
  if len(set(y[train]))<2 or len(set(y[test]))<2:continue
  model=build_estimator(params)
  if model is None:continue
  model.fit(X[train],y[train])
  p=np.clip(model.predict_proba(X[test])[:,1],1e-5,1-1e-5)
  folds.append({'brier':brier(y[test],p),'logloss':float(log_loss(y[test],p,labels=[0,1])),'train_rows':int(len(train)),'test_rows':int(len(test))})
 if not folds:return None
 return {'params':params,'folds':folds,'brier':float(np.mean([x['brier'] for x in folds])),'logloss':float(np.mean([x['logloss'] for x in folds]))}


def _calibrated_model(params,X,y):
 base=build_estimator(params)
 if base is None:return None
 method='isotonic' if params['family'] in ('random_forest','xgboost') and len(X)>=120 else 'sigmoid'
 cv=3 if min(np.bincount(y))>=3 else 2
 return CalibratedClassifierCV(base,method=method,cv=cv).fit(X,y)


def main():
 db=Database();run_id='train_'+uuid.uuid4().hex[:12]
 db.upsert('training_runs',{'run_id':run_id,'started_at':_now(),'finished_at':None,'rows_used':0,'cv_metrics_json':{},'model_id':None,'status':'running'},['run_id'])
 rows,names,X,y=_load_training_frame(db)
 if len(rows)<60 or len(set(y))<2:
  metrics={'reason':'insufficient_matured_labels_or_single_class','rows':len(rows),'classes':sorted(set(map(int,y))) if len(y) else []}
  db.upsert('training_runs',{'run_id':run_id,'started_at':None,'finished_at':_now(),'rows_used':len(rows),'cv_metrics_json':metrics,'model_id':None,'status':'deferred'},['run_id'])
  print('training deferred: insufficient matured labels')
  return 0

 scored=[x for x in (_score_candidate(params,X,y) for params in candidate_grid()) if x]
 if not scored:
  metrics={'reason':'no_candidate_passed_purged_cv','rows':len(rows)}
  db.upsert('training_runs',{'run_id':run_id,'started_at':None,'finished_at':_now(),'rows_used':len(rows),'cv_metrics_json':metrics,'model_id':None,'status':'failed'},['run_id'])
  print('training failed: no candidate passed purged CV')
  return 2

 scored=sorted(scored,key=lambda x:(x['brier'],x['logloss']))
 winners=scored[:min(3,len(scored))]
 models=[m for m in (_calibrated_model(item['params'],X,y) for item in winners) if m is not None]
 if not models:
  metrics={'reason':'calibration_failed','candidate_count':len(scored)}
  db.upsert('training_runs',{'run_id':run_id,'started_at':None,'finished_at':_now(),'rows_used':len(rows),'cv_metrics_json':metrics,'model_id':None,'status':'failed'},['run_id'])
  print('training failed: calibration failed')
  return 2

 probabilities=np.clip(shrink_probabilities(np.mean([m.predict_proba(X)[:,1] for m in models],axis=0).tolist()),1e-5,1-1e-5)
 importance=np.zeros(len(names))
 for model in models:
  calibrated=getattr(model,'calibrated_classifiers_',[])
  estimator=getattr(calibrated[0],'estimator',None) if calibrated else None
  if hasattr(estimator,'coef_'):importance+=np.abs(estimator.coef_[0])
  elif hasattr(estimator,'feature_importances_'):importance+=np.asarray(estimator.feature_importances_)
 if not importance.any():importance=np.ones(len(names))
 importance=importance/max(float(importance.sum()),1e-9)
 conformal=conformal_interval(y,probabilities)
 metrics={'validation':'purged_walk_forward_embargo','candidate_scores':scored[:10],'selected_candidates':winners,'brier':brier(y,probabilities),'logloss':float(log_loss(y,probabilities,labels=[0,1])),'calibration':calibration_curve(y,probabilities),'calibrators':['sigmoid_or_isotonic_by_candidate'],'uncertainty':{'conformal':conformal},'explainability':'shap_when_available_else_model_importance'}

 model_id='challenger_'+run_id
 artifacts=ensure_artifacts();path=artifacts/(model_id+'.joblib')
 bundle={'model_id':model_id,'feature_names':names,'models':models,'importance':importance,'conformal':conformal,'metrics':metrics}
 joblib.dump(bundle,path)
 registry=ModelRegistry(db)
 existing=registry.champion()
 existing_metrics=json.loads(existing.get('metrics_json') or '{}') if existing else {}
 if existing and (not Path(existing.get('path','')).exists() or int(existing_metrics.get('training_rows',0))<60): db.execute("UPDATE models SET status='retired' WHERE model_id=?",(existing['model_id'],))
 rollback=registry.rollback_if_breached()
 if rollback['decision']=='rollback': db.upsert('champion_challenger',{'date':date.today().isoformat(),'champion_id':rollback['champion_id'],'challenger_id':'automatic_rollback','decision':'rollback','reason':rollback['reason']},['date','challenger_id'])
 decision=registry.promote_if_better(model_id,'calibrated_challenger_ensemble_with_optional_xgboost',metrics,str(path))
 if rollback['decision']=='rollback':
  active=db.rows('SELECT path FROM models WHERE model_id=?',(rollback['champion_id'],))
  if active and Path(active[0]['path']).exists():shutil.copy2(active[0]['path'],artifacts/'champion_model.joblib')
 if decision['decision']=='promote':shutil.copy2(path,artifacts/'champion_model.joblib')
 db.upsert('champion_challenger',{'date':date.today().isoformat(),'champion_id':decision.get('champion_id',model_id),'challenger_id':decision.get('challenger_id',model_id),'decision':decision['decision'],'reason':decision['reason']},['date','challenger_id'])
 write_json('model_registry_snapshot.json',db.rows('SELECT * FROM models'))
 write_json('calibration_objects.json',metrics)
 write_text('monthly_research_review.md','# Monthly Research Review\n\nChampion promotion is gated by matured-label purged walk-forward validation, calibration, conformal uncertainty, and challenger comparison. Rollback support is implemented in the model registry.')
 db.upsert('training_runs',{'run_id':run_id,'started_at':None,'finished_at':_now(),'rows_used':len(rows),'cv_metrics_json':metrics,'model_id':model_id,'status':'success' if decision['decision']=='promote' else 'challenger_rejected'},['run_id'])
 print(model_id+' '+decision['decision'])
 return 0


if __name__=='__main__':
 raise SystemExit(main())
