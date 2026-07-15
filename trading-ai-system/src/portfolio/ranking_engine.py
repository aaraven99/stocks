def rank(rows,regime,playbook):
 out=[]
 for r in rows:
  mc=r['mc']; p=r['prediction']; quality=r['quality']['data_confidence']; stress=r.get('stress_penalty',0); score=(.43*p['probability']+.25*max(0,mc['expected_return']*10)+.18*(1-mc['p_stop_hit_first'])+.14*quality-stress)*playbook['risk_multiplier']; action='LONG' if score>=playbook['min_score'] else 'HOLD'; out.append({**r,'score':float(score),'action':action,'confidence':p['confidence'],'volatility':r['features'].get('volatility_20d',.2)})
 return sorted(out,key=lambda x:x['score'],reverse=True)
