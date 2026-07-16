import plotly.graph_objects as go
from pathlib import Path
def equity_chart(rows): return go.Figure(go.Scatter(x=[r['date'] for r in rows],y=[r['equity'] for r in rows],name='Equity'))
def drawdown_chart(rows): return go.Figure(go.Scatter(x=[r['date'] for r in rows],y=[r['drawdown'] for r in rows],fill='tozeroy',name='Drawdown'))
def candle_chart(df): return go.Figure(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close))
def histogram(dist): return go.Figure(go.Histogram(x=dist,nbinsx=60,name='Simulated return'))

def equity_chart_png(rows,path):
 """Write an email-safe PNG; Gmail removes iframes and executable Plotly HTML."""
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(10,3.2),dpi=140);fig.patch.set_facecolor('#071018');ax.set_facecolor('#071018')
 if rows:ax.plot([r['date'] for r in rows],[r['equity'] for r in rows],color='#27d17f',linewidth=2)
 else:ax.text(.5,.5,'No paper equity observations yet',ha='center',va='center',color='#dbe7f3',transform=ax.transAxes)
 ax.tick_params(colors='#a9b7c6',labelsize=7);ax.grid(color='#223746',alpha=.5);ax.set_title('Paper Trading Equity Curve',color='#dbe7f3');fig.autofmt_xdate();fig.tight_layout();fig.savefig(path,facecolor=fig.get_facecolor());plt.close(fig);return path
