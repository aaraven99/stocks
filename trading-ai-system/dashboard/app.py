import sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import streamlit as st
import plotly.express as px
from storage.db import Database
from data_sources.yfinance_source import fetch_ohlcv
from reporting.charts import candle_chart,histogram
st.set_page_config(page_title='QRS Terminal',layout='wide',initial_sidebar_state='expanded')
st.markdown('<style>.stApp{background:#071018;color:#dbe7f3}.block-container{padding-top:1rem;max-width:1600px}div[data-testid="stMetric"]{background:#0e1c28;padding:7px;border-left:3px solid #27d17f}.stDataFrame{font-size:11px}</style>',unsafe_allow_html=True)
@st.cache_data(ttl=60)
def q(sql,params=()):
 try:return pd.DataFrame(Database().rows(sql,params))
 except Exception as exc:st.warning(f'Database warning: {exc}');return pd.DataFrame()
def latest():
 x=q('SELECT max(date) date FROM rankings');return x.iloc[0,0] if not x.empty and x.iloc[0,0] else pd.Timestamp.today().date().isoformat()
st.sidebar.title('QRS TERMINAL')
universe=st.sidebar.selectbox('Universe',['full universe','top500','backup'],1)
date=st.sidebar.date_input('As-of date',pd.Timestamp(latest())).isoformat();ticker=st.sidebar.text_input('Ticker search','SPY').upper()
test=st.sidebar.toggle('TEST_MODE status',True);st.sidebar.toggle('Use Monte Carlo weighting',True);st.sidebar.toggle('Use ML model predictions',True);st.sidebar.toggle('Use only top-ranked picks',True)
if st.sidebar.button('Refresh'):st.cache_data.clear();st.rerun()
if st.sidebar.button('Run pipeline safely'):
 with st.spinner('Running safe test pipeline'):
  result=subprocess.run([sys.executable,str(ROOT/'src/pipeline/run_daily.py'),'--test-mode'],capture_output=True,text=True,timeout=300);st.sidebar.code((result.stdout or result.stderr)[-1000:]);st.cache_data.clear()
overview,scanner,deep,portfolio,mc_lab,governance,debate=st.tabs(['Overview','Market Scanner','Single Ticker Deep Dive','Portfolio','Monte Carlo & Forecast Lab','Model Governance / MRM','Agents / Debate'])
with overview:
 st.header('Portfolio Overview');eq=q('SELECT * FROM equity_curve ORDER BY date')
 if eq.empty:st.warning('No data. Run the safe pipeline.')
 else:
  last=eq.iloc[-1];trades=q("SELECT * FROM paper_trades WHERE status='closed'");win=(trades.pnl>0).mean() if not trades.empty else 0
  daily_return=eq.daily_pnl/eq.equity.shift(1).fillna(eq.equity.iloc[0]);downside=daily_return[daily_return<0];sharpe=daily_return.mean()/max(daily_return.std(),1e-9)*252**.5;sortino=daily_return.mean()/max(downside.std(),1e-9)*252**.5 if len(downside)>1 else 0;avg_trade=trades.pnl.mean() if not trades.empty else 0
  values=[('Equity',f'${last.equity:,.0f}'),('Total return',f'{last.equity/eq.equity.iloc[0]-1:.2%}'),('Sharpe',f'{sharpe:.2f}'),('Sortino',f'{sortino:.2f}'),('Max DD',f'{eq.drawdown.min():.2%}'),('Win rate',f'{win:.1%}'),('Avg trade P&L',f'${avg_trade:,.0f}'),('Exposure',f'{last.exposure:.1%}'),('Cash',f'${last.cash:,.0f}')]
  cols=st.columns(9)
  for col,(label,value) in zip(cols,values):col.metric(label,value)
  st.plotly_chart(px.line(eq,x='date',y='equity',title='Equity Curve'),use_container_width=True);st.plotly_chart(px.area(eq,x='date',y='drawdown',title='Drawdown'),use_container_width=True);st.plotly_chart(px.bar(eq,x='date',y='daily_pnl',title='Daily P&L'),use_container_width=True);st.subheader('Latest daily run');st.dataframe(q('SELECT date,status,stage_timings_json,errors_json FROM pipeline_runs ORDER BY started_at DESC LIMIT 1'),hide_index=True,use_container_width=True)
with scanner:
 st.header('Market Scanner');r=q('SELECT r.*,p.probability,p.expected_return,c.close last_close,m.metrics_json,a.sector,a.weight allocation_weight FROM rankings r LEFT JOIN predictions p ON r.ticker=p.ticker AND r.date=p.date LEFT JOIN cleaned_prices c ON r.ticker=c.ticker AND r.date=c.date LEFT JOIN montecarlo_metrics m ON r.ticker=m.ticker AND r.date=m.date LEFT JOIN allocations a ON r.ticker=a.ticker AND r.date=a.date WHERE r.date=? ORDER BY r.score DESC',(date,))
 if r.empty:st.warning('No scanner records for selected date.')
 else:
  details=r.details_json.map(json.loads);metrics=r.metrics_json.map(lambda value:json.loads(value) if value else {});r['sector']=[sector if pd.notna(sector) else row.get('sector','Unknown') for row,sector in zip(details,r.sector)];r['volatility']=[row.get('volatility',0) for row in details];r['volatility_bucket']=pd.cut(r.volatility,[-float('inf'),.20,.35,float('inf')],labels=['low','medium','high']);r['mc_expected_return']=[row.get('expected_return',0) for row in metrics];r['downside_risk_percentile']=[row.get('q10',row.get('var_95',0)) for row in metrics];r['mc_skew']=[row.get('p_plus_5_10d',0)-row.get('p_stop_hit_first',0) for row in metrics];r['recommended_position_size']=r.allocation_weight.fillna(r.position_size);r['final_ranking_score']=r.score;r['recommended_action']=r.action;r['regime_classification']=r.regime
  f1,f2,f3,f4=st.columns(4);sector_filter=f1.selectbox('Sector',['All']+sorted(r.sector.fillna('Unknown').unique().tolist()));vol_filter=f2.selectbox('Volatility bucket',['All','low','medium','high']);threshold=f3.slider('Confidence threshold',0.,1.,0.);skew=f4.slider('Monte Carlo skew threshold',-1.,1.,-1.)
  view=r[(r.confidence>=threshold)&(r.mc_skew>=skew)];view=view if sector_filter=='All' else view[view.sector==sector_filter];view=view if vol_filter=='All' else view[view.volatility_bucket.astype(str)==vol_filter];columns=['ticker','sector','last_close','probability','mc_expected_return','downside_risk_percentile','regime_classification','stress_penalty','final_ranking_score','recommended_action','recommended_position_size'];st.dataframe(view[columns],use_container_width=True,hide_index=True);st.download_button('Export scanner CSV',view.to_csv(index=False),'scanner.csv','text/csv')
with deep:
 st.header(f'{ticker} Deep Dive');prices=fetch_ohlcv(ticker,test_mode=test);st.plotly_chart(candle_chart(prices.tail(120)),use_container_width=True)
 pr=q('SELECT * FROM predictions WHERE ticker=? ORDER BY date DESC LIMIT 1',(ticker,));mc=q('SELECT * FROM montecarlo_metrics WHERE ticker=? ORDER BY date DESC LIMIT 1',(ticker,))
 if not pr.empty:
  p=pr.iloc[0];a,b,c=st.columns(3);a.metric('P(positive)',f'{p.probability:.1%}');b.metric('Expected return',f'{p.expected_return:.2%}');c.metric('Uncertainty',f'{p.uncertainty:.2%}');st.subheader('SHAP / model attribution');explanation=json.loads(p.explanation_json);attributions=explanation.get('explanation',[]) if isinstance(explanation,dict) else explanation
  if attributions and isinstance(attributions,list) and 'feature' in attributions[0]:st.bar_chart(pd.DataFrame(attributions).set_index('feature')['attribution'])
  else:st.warning('No compatible feature-attribution payload is available for this prediction.')
 if not mc.empty:
  metrics=json.loads(mc.iloc[0].metrics_json);st.plotly_chart(histogram(json.loads(mc.iloc[0].distribution_json)),use_container_width=True);st.json({k:metrics[k] for k in ['var_95','cvar_95','p_stop_hit_first','expected_return']})
 st.dataframe(q('SELECT scenario,pnl_pct,penalty FROM stress_outputs WHERE ticker=? ORDER BY date DESC',(ticker,)),hide_index=True)
 debate_rows=q('SELECT date,round_no,agent,stance,content,score FROM debate_rounds WHERE date=(SELECT max(date) FROM debate_rounds) ORDER BY round_no,agent')
 if not debate_rows.empty:
  st.subheader('Latest agent debate context')
  st.dataframe(debate_rows[debate_rows.content.fillna('').str.contains(ticker,case=False,na=False)].tail(50) if debate_rows.content.fillna('').str.contains(ticker,case=False,na=False).any() else debate_rows.tail(20),hide_index=True,use_container_width=True)
 drift=q('SELECT flag,psi,ks FROM drift_metrics ORDER BY date DESC LIMIT 1')
 if not drift.empty:st.caption(f"Model drift: {drift.iloc[0].flag} | PSI {drift.iloc[0].psi:.3f} | KS {drift.iloc[0].ks:.3f}")
with portfolio:
 st.header('Portfolio');pos=q('SELECT * FROM positions');st.dataframe(pos,use_container_width=True,hide_index=True)
 if not pos.empty:st.plotly_chart(px.pie(pos,names='ticker',values='weight',title='Allocation'),use_container_width=True)
 sector=q('SELECT sector,sum(weight) weight FROM allocations WHERE date=? GROUP BY sector',(date,))
 if not sector.empty:st.plotly_chart(px.bar(sector,x='sector',y='weight',title='Sector Exposure'),use_container_width=True)
 if len(pos)>1:
  closes=pd.concat({symbol:fetch_ohlcv(symbol,test_mode=test).Close.pct_change() for symbol in pos.ticker},axis=1).dropna().tail(60);st.plotly_chart(px.imshow(closes.corr(),text_auto='.2f',color_continuous_scale='RdBu_r',title='Position Correlation Heatmap'),use_container_width=True)
 st.subheader('Paper Trading');st.dataframe(q('SELECT * FROM paper_trades ORDER BY id DESC LIMIT 100'),use_container_width=True,hide_index=True)
 analytics_table=q('SELECT factor_json,alpha_beta_json,cost_json,cluster_json,ruin_json FROM portfolio_analytics ORDER BY date DESC LIMIT 1')
 if not analytics_table.empty:st.subheader('Risk decomposition');st.json({key:json.loads(analytics_table.iloc[0][key]) for key in analytics_table.columns})
with mc_lab:
 st.header('Monte Carlo & Forecast Lab');m=q('SELECT ticker,engine_weights_json,metrics_json FROM montecarlo_metrics WHERE date=?',(date,));st.dataframe(m,use_container_width=True,hide_index=True)
 if not m.empty:
  weights=pd.DataFrame([{'ticker':row.ticker,**json.loads(row.engine_weights_json)} for _,row in m.iterrows()]).set_index('ticker');st.plotly_chart(px.bar(weights.mean().reset_index(),x='index',y=0,title='Monte Carlo Engine Weights'),use_container_width=True)
 st.subheader('Forecast validation');st.dataframe(q('SELECT * FROM calibration_metrics ORDER BY date DESC LIMIT 100'),use_container_width=True,hide_index=True)
with governance:
 st.header('Model Governance / MRM');st.dataframe(q('SELECT * FROM models ORDER BY trained_at DESC'),use_container_width=True,hide_index=True);st.dataframe(q('SELECT * FROM champion_challenger ORDER BY date DESC'),use_container_width=True,hide_index=True);dr=q('SELECT * FROM drift_metrics ORDER BY date')
 if not dr.empty:st.plotly_chart(px.line(dr,x='date',y='psi',color='model_id',title='PSI Drift'),use_container_width=True)
 calibration=q('SELECT * FROM calibration_metrics ORDER BY date DESC LIMIT 1')
 if not calibration.empty:
  curve=json.loads(calibration.iloc[0].curve_json)
  if curve:st.plotly_chart(px.line(pd.DataFrame(curve),x='prediction',y='outcome',markers=True,title='Calibration Curve'),use_container_width=True)
 st.dataframe(q('SELECT * FROM kill_switch_history ORDER BY date DESC LIMIT 50'),hide_index=True);st.dataframe(q('SELECT * FROM model_risk_decisions ORDER BY date DESC LIMIT 50'),hide_index=True);st.dataframe(q('SELECT * FROM reproducibility_manifests ORDER BY created_at DESC LIMIT 20'),hide_index=True)
with debate:
 st.header('Agents / Debate');st.dataframe(q('SELECT * FROM debate_rounds WHERE date=? ORDER BY round_no,agent',(date,)),use_container_width=True,hide_index=True);st.dataframe(q('SELECT * FROM agents_transcripts WHERE date=?',(date,)),use_container_width=True,hide_index=True)
st.divider();snapshot=q('SELECT * FROM rankings WHERE date=?',(date,));st.download_button('Export dashboard snapshot HTML',f'<html><body><h1>QRS Terminal {date}</h1>{snapshot.to_html()}</body></html>','dashboard_snapshot.html','text/html')
if snapshot.empty:
 mini_report='# No report\n'
else:
 cols=['ticker','score','action']; available=[c for c in cols if c in snapshot.columns]
 mini_report='| '+' | '.join(available)+' |\n| '+' | '.join('---' for _ in available)+' |\n'
 mini_report+=''.join('| '+' | '.join(str(row[c]) for c in available)+' |\n' for _,row in snapshot[available].iterrows())
st.download_button('Export mini daily report',mini_report,'daily_report.md','text/markdown')
