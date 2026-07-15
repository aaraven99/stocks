import plotly.graph_objects as go
def equity_chart(rows): return go.Figure(go.Scatter(x=[r['date'] for r in rows],y=[r['equity'] for r in rows],name='Equity'))
def drawdown_chart(rows): return go.Figure(go.Scatter(x=[r['date'] for r in rows],y=[r['drawdown'] for r in rows],fill='tozeroy',name='Drawdown'))
def candle_chart(df): return go.Figure(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close))
def histogram(dist): return go.Figure(go.Histogram(x=dist,nbinsx=60,name='Simulated return'))
