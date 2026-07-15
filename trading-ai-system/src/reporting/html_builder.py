from reporting.templates import CSS
from reporting.report_sections import disclaimer


def _pct(value):
 try:return f"{float(value):.1%}"
 except Exception:return "n/a"


def _num(value, digits=2):
 try:
  if value is None:return "n/a"
  return f"{float(value):.{digits}f}"
 except Exception:
  return "n/a"


def build(report):
 picks=report.get('picks',[])
 rows=''.join(
  f"<tr><td>{x['ticker']}</td><td>{x['action']}</td><td>{x['score']:.3f}</td><td>{_pct(x['prediction']['probability'])}</td><td>{_pct(x['mc']['expected_return'])}</td><td>{_pct(x['mc']['var_95'])}</td><td>{_pct(x['mc'].get('p_target_before_stop',0))}</td><td>{_pct(x['weight'])}</td></tr>"
  for x in picks
 ) or '<tr><td colspan="8">CASH MODE / no allocations</td></tr>'
 mc_rows=''.join(
  f"<tr><td>{x['ticker']}</td><td>{_pct(x['mc'].get('p_plus_3_5d',0))}</td><td>{_pct(x['mc'].get('p_plus_5_10d',0))}</td><td>{_pct(x['mc'].get('p_plus_10_20d',0))}</td><td>{_pct(x['mc'].get('p_stop_hit_first',0))}</td><td>{_pct(x['mc'].get('cvar_95',0))}</td><td>{x['mc'].get('price_target_q50',0):.2f}</td></tr>"
  for x in picks
 ) or '<tr><td colspan="7">No Monte Carlo allocation candidates</td></tr>'
 stress_rows=''.join(
  f"<tr><td>{x.get('scenario')}</td><td>{_pct(x.get('avg_pnl',0))}</td><td>{float(x.get('avg_penalty',0)):.3f}</td></tr>"
  for x in report.get('stress',[])
 ) or '<tr><td colspan="3">No stress rows for this run</td></tr>'
 forecast=report.get('regime_forecast',{});health=report.get('universe_health',{});analytics=report.get('analytics',{});paper=analytics.get('paper',{});debate=report.get('debate',{});cal=report.get('calibration',{})
 narrative=report.get('narrative',{}).get('text','No narrative available.')
 factor=analytics.get('factor_exposure',{})
 return f'''<html><head><style>{CSS}</style></head><body>
<h1>Systematic Swing Research Briefing</h1>
<div class="tile">As-of: {report['date']} | Decision: <b>{report['decision']}</b> | Regime: {report['regime']}</div>
<h2>Market Dashboard & Regime Forecast</h2>
<table><tr><th>Regime</th><th>P(bear flip, 5d)</th><th>P(vol spike, 5d)</th><th>Universe loaded</th><th>Deep analysis</th><th>Failover</th></tr><tr><td>{report['regime']}</td><td>{_pct(forecast.get('flip_bearish_5d',0))}</td><td>{_pct(forecast.get('vol_spike_5d',0))}</td><td>{health.get('loaded',0)}/{health.get('requested',0)}</td><td>{health.get('deep_analyzed',0)}</td><td>{health.get('failover','none')}</td></tr></table>
<h2>Ranked Research Watchlist</h2>
<table><tr><th>Ticker</th><th>Action</th><th>Score</th><th>ML P(positive)</th><th>MC EV</th><th>MC VaR 95</th><th>P(target before stop)</th><th>Allocation</th></tr>{rows}</table>
<h2>Monte Carlo Summary</h2>
<table><tr><th>Ticker</th><th>P(+3%, 5d)</th><th>P(+5%, 10d)</th><th>P(+10%, 20d)</th><th>P(stop first)</th><th>CVaR 95</th><th>Median target</th></tr>{mc_rows}</table>
<h2>ML Confidence, Calibration & Drift</h2>
<p>Brier: {float(cal.get('brier',0)):.4f} | Drift PSI: {float(cal.get('drift',{}).get('psi',0)):.4f} | Drift flag: {cal.get('drift',{}).get('flag','unknown')}</p>
<h2>Stress Test Results</h2>
<table><tr><th>Scenario</th><th>Average PnL</th><th>Average Penalty</th></tr>{stress_rows}</table>
<h2>Paper Trading & Equity Curve</h2>
<p>Closed trades: {paper.get('closed_trades',0)} | Win rate: {_pct(paper.get('win_rate',0))} | Profit factor: {_num(paper.get('profit_factor',0))} | Max drawdown: {_pct(paper.get('max_drawdown',0))} | Risk of ruin proxy: {_pct(analytics.get('risk_of_ruin',0))}</p>
<p>Embedded chart artifact: <code>equity_curve_chart.html</code></p>
<h2>Agent Debate Summary</h2>
<p>Bull score: {debate.get('bull_score',0):.2f}; Bear score: {debate.get('bear_score',0):.2f}; Committee verdict: {debate.get('action','CASH')}</p>
<h2>Attribution Summary</h2>
<p>Portfolio factor exposure: {factor}</p>
<p>{narrative}</p>
{disclaimer()}</body></html>'''
