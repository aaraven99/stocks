from models.meta_model import regime_weighted_score
from models.ensemble import blend
def rank(rows,regime,playbook):
 out=[]
 for r in rows:
  mc=r['mc']; p=r['prediction']; combined=blend(p,mc);quality=r['quality']['data_confidence']; stress=r.get('stress_penalty',0); base=regime_weighted_score(combined,mc,regime,r.get('archetypes')); score=.86*base+.14*quality-stress; action='LONG' if score>=playbook['min_score'] else 'HOLD'; out.append({**r,'score':float(max(0,min(1,score))),'action':action,'confidence':p['confidence'],'volatility':r['features'].get('volatility_20d',.2),'ensemble_prediction':combined})
 return sorted(out,key=lambda x:x['score'],reverse=True)
