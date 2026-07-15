"""Causal nearest-neighbour memory over prior feature snapshots and matured outcomes."""
import json
import numpy as np
def store(db,ticker,date,features,outcome=None):
 db.upsert('analog_memory',{'ticker':ticker,'date':date,'feature_vector_json':features,'outcome_json':outcome or {}},['ticker','date'])
def nearest(db,features,before_date,k=10):
 rows=db.rows('SELECT ticker,date,feature_vector_json,outcome_json FROM analog_memory WHERE date < ?',(before_date,));keys=sorted(features)
 if not rows:return []
 target=np.array([float(features.get(key,0)) for key in keys]);out=[]
 for row in rows:
  vector=json.loads(row['feature_vector_json']);x=np.array([float(vector.get(key,0)) for key in keys]);den=np.linalg.norm(target)*np.linalg.norm(x);similarity=float(target@x/den) if den else 0.;out.append({**row,'similarity':similarity,'outcome':json.loads(row['outcome_json'])})
 return sorted(out,key=lambda row:row['similarity'],reverse=True)[:k]
def outcome_prior(analogs):
 values=[row['outcome'].get('realized_return') for row in analogs if row['outcome'].get('realized_return') is not None]
 return {'count':len(values),'mean_return':float(np.mean(values)) if values else 0.,'win_rate':float(np.mean([v>0 for v in values])) if values else .5}
