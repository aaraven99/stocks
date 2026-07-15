from paper_trading.execution_simulator import fill
from paper_trading.pnl_tracker import pnl
from portfolio.transaction_cost_model import estimate
class PaperEngine:
 def __init__(self,db,capital=100000): self.db=db;self.capital=float(capital)
 def mark_and_close(self,prices,date):
  for p in self.db.rows('SELECT * FROM positions'):
   last=prices.get(p['ticker'],p['last_price']); value=pnl(p['entry'],last,p['shares'])
   if last<=p['stop'] or last>=p['take_profit']:
    exit_price=fill(last,'SHORT'); net=value-estimate(exit_price,p['shares']); self.db.execute('INSERT INTO paper_trades(ticker,opened_date,closed_date,side,shares,entry,exit,pnl,reason,status) VALUES(?,?,?,?,?,?,?,?,?,?)',(p['ticker'],p['date'],date,'LONG',p['shares'],p['entry'],exit_price,net,'stop_or_target','closed')); self.db.execute('DELETE FROM positions WHERE ticker=?',(p['ticker'],))
   else:self.db.upsert('positions',{'ticker':p['ticker'],'date':p['date'],'shares':p['shares'],'entry':p['entry'],'last_price':last,'pnl':value,'stop':p['stop'],'take_profit':p['take_profit'],'weight':p['weight']},['ticker'])
 def open(self,plans,date):
  for x in plans:
   if x['shares']>0 and not self.db.rows('SELECT ticker FROM positions WHERE ticker=?',(x['ticker'],)):
    entry=fill(x['entry'],'LONG')+x.get('transaction_cost',0)/x['shares'];self.db.upsert('positions',{'ticker':x['ticker'],'date':date,'shares':x['shares'],'entry':entry,'last_price':x['entry'],'pnl':pnl(entry,x['entry'],x['shares']),'stop':x['stop'],'take_profit':x['target'],'weight':x['weight']},['ticker'])
 def snapshot(self,date):
  pos=self.db.rows('SELECT * FROM positions'); equity=self.capital+sum(p['pnl'] for p in pos); prev=self.db.rows('SELECT equity FROM equity_curve ORDER BY date DESC LIMIT 1'); prior=prev[0]['equity'] if prev else self.capital; peak=max([r['equity'] for r in self.db.rows('SELECT equity FROM equity_curve')]+[equity]); self.db.upsert('equity_curve',{'date':date,'equity':equity,'cash':self.capital-sum(p['entry']*p['shares'] for p in pos),'exposure':sum(p['entry']*p['shares'] for p in pos)/max(equity,1),'daily_pnl':equity-prior,'drawdown':equity/peak-1},['date']);return equity
