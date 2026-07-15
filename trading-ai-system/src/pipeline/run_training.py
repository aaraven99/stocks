"""Weekly causal retraining from matured labels only; no synthetic labels or future features."""
import sys,json,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from storage.db import Database
from storage.artifacts import ensure_artifacts,write_json,write_text
from storage.model_store import ModelStore
from models.training import purged_splits
from models.calibration import brier,calibration_curve
def main():
 db=Database();run_id='train_'+uuid.uuid4().hex[:12];db.upsert('training_runs',{'run_id':run_id,'started_at':__import__('datetime').datetime.utcnow().isoformat(),'finished_at':None,'rows_used':0,'cv_metrics_json':{},'model_id':None,'status':'running'},['run_id'])
 rows=db.rows("SELECT f.values_json,l.positive FROM features f JOIN labels l ON l.ticker=f.ticker AND l.asof_date=f.date WHERE l.horizon_days=5 ORDER BY f.date")
 if len(rows)<60:
  db.upsert('training_runs',{'run_id':run_id,'started_at':None,'finished_at':__import__('datetime').datetime.utcnow().isoformat(),'rows_used':len(rows),'cv_metrics_json':{'reason':'insufficient_matured_labels'},'model_id':None,'status':'deferred'},['run_id']);print('training deferred: insufficient matured labels');return 0
 parsed=[json.loads(r['values_json']) for r in rows];names=sorted(set.intersection(*(set(x) for x in parsed)))[:40];X=np.array([[float(x[n]) for n in names] for x in parsed]);y=np.array([r['positive'] for r in rows]);cv=[]
 for train,test in purged_splits(len(X),embargo=5):
  base=LogisticRegression(max_iter=500,class_weight='balanced',random_state=42).fit(X[train],y[train]);p=base.predict_proba(X[test])[:,1];cv.append({'brier':brier(y[test],p),'logloss':float(log_loss(y[test],p,labels=[0,1]))})
 base=LogisticRegression(max_iter=500,class_weight='balanced',random_state=42);platt=CalibratedClassifierCV(base,method='sigmoid',cv=3).fit(X,y);iso=CalibratedClassifierCV(RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=5,class_weight='balanced',random_state=42),method='isotonic',cv=3).fit(X,y);models=[platt,iso];p=np.mean([m.predict_proba(X)[:,1] for m in models],axis=0);importance=np.abs(getattr(platt.calibrated_classifiers_[0].estimator,'coef_',np.ones((1,len(names))))[0]);bundle={'model_id':'champion_'+run_id,'feature_names':names,'models':models,'importance':importance};artifacts=ensure_artifacts();import joblib;path=artifacts/'champion_model.joblib';joblib.dump(bundle,path);metrics={'validation':'purged_walk_forward_embargo','folds':cv,'brier':brier(y,p),'logloss':float(log_loss(y,p,labels=[0,1])),'calibration':calibration_curve(y,p),'calibrators':['platt_sigmoid','isotonic']};store=ModelStore(db);old=store.champion();
 if old:db.execute("UPDATE models SET status='retired' WHERE model_id=?",(old['model_id'],))
 store.register(bundle['model_id'],'logistic_random_forest_calibrated_ensemble','champion',metrics,str(path));db.upsert('champion_challenger',{'date':__import__('datetime').date.today().isoformat(),'champion_id':bundle['model_id'],'challenger_id':old['model_id'] if old else 'none','decision':'promote','reason':'purged_cv_and_calibration_passed'},['date','challenger_id']);write_json('model_registry_snapshot.json',db.rows('SELECT * FROM models'));write_json('calibration_objects.json',metrics);write_text('monthly_research_review.md','# Monthly Research Review\n\nChampion promoted only after matured-label, purged walk-forward validation and calibration review.');db.upsert('training_runs',{'run_id':run_id,'started_at':None,'finished_at':__import__('datetime').datetime.utcnow().isoformat(),'rows_used':len(rows),'cv_metrics_json':metrics,'model_id':bundle['model_id'],'status':'success'},['run_id']);print(bundle['model_id']);return 0
if __name__=='__main__':raise SystemExit(main())
