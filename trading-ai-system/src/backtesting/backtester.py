"""Leakage-controlled expanding-window backtester for matured five-day labels."""
from collections import defaultdict
from datetime import date

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtesting.performance_metrics import calculate
from backtesting.benchmark import relative
from backtesting.walkforward import eligible_history


def _business_days(start,end):return max(1,int(np.busday_count(str(start),str(end)))+1)


def _estimator(model_family='xgboost'):
 """Build the causal walk-forward estimator.

 XGBoost is the production default because it is part of the declared model
 stack and captures nonlinear interactions between the registered indicators.
 The logistic pipeline is a deterministic fallback when an installation does
 not have the optional compiled XGBoost wheel.
 """
 if str(model_family).lower() in {'xgb','xgboost'}:
  try:
   from xgboost import XGBClassifier
   return XGBClassifier(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=1,
    random_state=42,
   ), 'xgboost'
  except Exception:
   model_family='logistic_fallback'
 return make_pipeline(
  SimpleImputer(strategy='median'),
  StandardScaler(),
  LogisticRegression(max_iter=2000,C=.5,class_weight='balanced',random_state=42),
 ), 'logistic_fallback'

def _marked_portfolio_returns(trades,dates,notional_fraction):
 """Mark fixed-notional five-day positions instead of compounding overlapping labels."""
 date_index={value:i for i,value in enumerate(dates)};daily=[0.0 for _ in dates]
 for trade in trades:
  start=date_index.get(trade['signal_date']);exit_index=date_index.get(trade.get('exit_date'))
  if start is None:continue
  if exit_index is None or exit_index<=start:exit_index=min(len(dates)-1,start+5)
  holding=max(1,exit_index-start);gross=max(-.999,1.0+float(trade['net_return']))
  per_day=gross**(1.0/holding)-1.0
  trade['holding_days']=holding;trade['notional_fraction']=float(notional_fraction)
  for index in range(start+1,exit_index+1):daily[index]+=float(notional_fraction)*per_day
 return [float(value) for value in daily]


def walk_forward(samples,min_train=60,embargo_dates=5,max_trades_per_date=2,probability_floor=.78,cost_bps=30.0,model_family='xgboost'):
 """Fit only on matured earlier observations and score each later date once."""
 samples=sorted(samples,key=lambda row:(row['date'],row['ticker']))
 dates=sorted({row['date'] for row in samples});by_date=defaultdict(list)
 for row in samples:by_date[row['date']].append(row)
 # Build the feature matrix once.  The previous implementation repeatedly
 # scanned every row for every date, which made a daily multi-year panel
 # unnecessarily quadratic while leaving the statistical split unchanged.
 feature_names=sorted(set.intersection(*(set(row['features']) for row in samples))) if samples else []
 matrix=np.asarray([[row['features'].get(name,np.nan) for name in feature_names] for row in samples],dtype=float) if feature_names else np.empty((len(samples),0))
 outcome=np.asarray([row['outcome'] for row in samples],dtype=int)
 row_dates=np.asarray([row['date'] for row in samples])
 date_to_indices=defaultdict(list)
 for index,row_date in enumerate(row_dates):date_to_indices[row_date].append(index)
 trades=[];benchmark_observations=[];diagnostics=[];model_used='not_run'
 for date_index,signal_date in enumerate(dates):
  cutoff_index=date_index-int(embargo_dates)
  if cutoff_index<=0:continue
  eligible_dates=eligible_history(dates,date_index,embargo_dates);train_indices=np.flatnonzero(np.isin(row_dates,list(eligible_dates)));test_indices=np.asarray(date_to_indices[signal_date],dtype=int);train=[samples[index] for index in train_indices];test=[samples[index] for index in test_indices]
  if len(train)<int(min_train) or len({int(row['outcome']) for row in train})<2:continue
  if not feature_names:continue
  X=matrix[train_indices];y=outcome[train_indices];Xt=matrix[test_indices]
  model,model_used=_estimator(model_family)
  model.fit(X,y);probabilities=model.predict_proba(Xt)[:,1]
  ranked=sorted(zip(test,probabilities),key=lambda item:(item[1],item[0]['ticker']),reverse=True)
  selected=[item for item in ranked if item[1]>=float(probability_floor)][:int(max_trades_per_date)]
  cohort=[]
  for row,probability in selected:
   gross=float(row['realized_return']);net=gross-float(cost_bps)/10000.0
   cohort.append(net);trades.append({'ticker':row['ticker'],'signal_date':signal_date,'exit_date':row.get('exit_date',signal_date),'side':'LONG','probability':float(probability),'gross_return':gross,'net_return':net,'cost_bps':float(cost_bps),'outcome':int(net>0)})
  if cohort:benchmark_observations.append(float(np.mean([row['realized_return']-float(cost_bps)/10000.0 for row in test])))
  diagnostics.append({'date':signal_date,'train_rows':len(train),'candidates':len(test),'selected':len(selected),'positive_rate_train':float(y.mean())})
 dates_for_marking=dates;notional_fraction=1.0/max(1,int(max_trades_per_date)*5)
 daily_returns=_marked_portfolio_returns(trades,dates_for_marking,notional_fraction)
 benchmark_returns=[float(value)/5.0 for value in benchmark_observations] if benchmark_observations else [0.0]
 metrics=calculate(daily_returns);benchmark_total=calculate(benchmark_returns)['total_return'];metrics.update(relative(metrics['total_return'],benchmark_total));metrics.update({'benchmark_total_return':benchmark_total,'trades':len(trades),'trade_win_rate':float(np.mean([row['outcome'] for row in trades])) if trades else 0.0,'average_trade_return':float(np.mean([row['net_return'] for row in trades])) if trades else 0.0,'profit_factor':float(sum(max(row['net_return'],0) for row in trades)/max(abs(sum(min(row['net_return'],0) for row in trades)),1e-12)) if trades else 0.0,'test_dates':len({row['signal_date'] for row in trades}),'business_days_per_trade':float(_business_days(trades[0]['signal_date'],trades[-1]['signal_date'])/len(trades)) if trades else float('inf'),'method':f'expanding_walk_forward_matured_labels_embargo_marked_positions_{model_used}','model_family':model_used,'embargo_dates':int(embargo_dates),'probability_floor':float(probability_floor),'cost_bps_round_trip':float(cost_bps),'marking':'geometric_path_from_matured_five_day_label','notional_fraction_per_trade':float(notional_fraction)})
 return {'metrics':metrics,'trades':trades,'diagnostics':diagnostics}
