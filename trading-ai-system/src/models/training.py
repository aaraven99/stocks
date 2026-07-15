import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from models.calibration import brier,calibration_curve
def purged_splits(n,n_splits=4,embargo=5):
 size=n//n_splits
 for i in range(1,n_splits):
  test=np.arange(i*size,min(n,(i+1)*size)); train=np.arange(0,max(0,i*size-embargo));
  if len(train)>20 and len(test):yield train,test
def train(X,y):
 X=np.nan_to_num(np.asarray(X));y=np.asarray(y); model=RandomForestClassifier(n_estimators=120,max_depth=5,random_state=42,class_weight='balanced'); model.fit(X,y); p=model.predict_proba(X)[:,1]; return model,{'brier':brier(y,p),'logloss':float(log_loss(y,p,labels=[0,1])),'calibration':calibration_curve(y,p),'validation':'walk_forward_purged_embargo'}
