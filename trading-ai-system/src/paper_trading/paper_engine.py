"""Persistent, cash-accounted paper trading with conservative OHLC barrier fills."""
from datetime import datetime,timezone

from paper_trading.execution_simulator import execution_fill
from paper_trading.pnl_tracker import pnl


class PaperEngine:
 def __init__(self,db,capital=100000):
  self.db=db;self.capital=float(capital);self.account_id='paper'
  if not self.db.rows('SELECT account_id FROM paper_account WHERE account_id=?',(self.account_id,)):
   self.db.upsert('paper_account',{'account_id':self.account_id,'starting_capital':self.capital,'cash':self.capital,'realized_pnl':0.0,'updated_at':self._now()},['account_id'])

 @staticmethod
 def _now():return datetime.now(timezone.utc).isoformat()

 def _account(self):return self.db.rows('SELECT * FROM paper_account WHERE account_id=?',(self.account_id,))[0]

 @staticmethod
 def _bar(value):
  if isinstance(value,dict):
   lowered={str(k).lower():float(v) for k,v in value.items() if v is not None}
   close=lowered.get('close',lowered.get('last_price'))
   if close is None:raise ValueError('bar requires close')
   return {'open':lowered.get('open',close),'high':lowered.get('high',close),'low':lowered.get('low',close),'close':close,'volume':lowered.get('volume',0.0)}
  price=float(value);return {'open':price,'high':price,'low':price,'close':price,'volume':0.0}

 def _execution(self,date,ticker,event_type,side,execution,shares,details=None):
  self.db.execute('INSERT INTO execution_events(date,ticker,event_type,side,reference_price,fill_price,shares,spread_cost,slippage_cost,commission,details_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(date,ticker,event_type,side,execution['reference_price'],execution['fill_price'],shares,execution['spread_cost'],execution['slippage_cost'],0.0,__import__('json').dumps(details or {})))

 def mark_and_close(self,prices,date):
  """Mark positions and execute stops/targets; if both hit, assume the stop hit first."""
  account=self._account();cash=float(account['cash']);realized=float(account['realized_pnl'])
  for position in self.db.rows('SELECT * FROM positions'):
   if position['ticker'] not in prices:continue
   bar=self._bar(prices[position['ticker']]);reference=None;reason=None
   if bar['open']<=position['stop']:reference=bar['open'];reason='gap_through_stop'
   elif bar['open']>=position['take_profit']:reference=bar['open'];reason='gap_through_target'
   elif bar['low']<=position['stop']:reference=position['stop'];reason='intraday_stop'
   elif bar['high']>=position['take_profit']:reference=position['take_profit'];reason='intraday_target'
   if reference is not None:
    execution=execution_fill(reference,'SELL',position['shares'],bar['close']*bar['volume'] if bar['volume'] else None)
    exit_price=execution['fill_price'];trade_pnl=pnl(position['entry'],exit_price,position['shares'])
    cash+=exit_price*position['shares'];realized+=trade_pnl
    self.db.execute('INSERT INTO paper_trades(ticker,opened_date,closed_date,side,shares,entry,exit,pnl,reason,status) VALUES(?,?,?,?,?,?,?,?,?,?)',(position['ticker'],position['date'],date,'LONG',position['shares'],position['entry'],exit_price,trade_pnl,reason,'closed'))
    self._execution(date,position['ticker'],'exit','SELL',execution,position['shares'],{'reason':reason,'bar':bar})
    self.db.execute('DELETE FROM positions WHERE ticker=?',(position['ticker'],))
   else:
    unrealized=pnl(position['entry'],bar['close'],position['shares'])
    self.db.upsert('positions',{'ticker':position['ticker'],'date':position['date'],'shares':position['shares'],'entry':position['entry'],'last_price':bar['close'],'pnl':unrealized,'stop':position['stop'],'take_profit':position['take_profit'],'weight':position['weight']},['ticker'])
  self.db.upsert('paper_account',{'account_id':self.account_id,'starting_capital':account['starting_capital'],'cash':cash,'realized_pnl':realized,'updated_at':self._now()},['account_id'])

 def open(self,plans,date):
  account=self._account();cash=float(account['cash'])
  for plan in plans:
   if plan['shares']<=0 or self.db.rows('SELECT ticker FROM positions WHERE ticker=?',(plan['ticker'],)):continue
   desired=float(plan['shares']);dollar_volume=plan.get('dollar_volume')
   preview=execution_fill(plan['entry'],'BUY',desired,dollar_volume)
   shares=min(desired,int(cash/max(preview['fill_price'],1e-9)))
   if shares<=0:continue
   execution=execution_fill(plan['entry'],'BUY',shares,dollar_volume);entry=execution['fill_price'];cash-=entry*shares
   self.db.upsert('positions',{'ticker':plan['ticker'],'date':date,'shares':shares,'entry':entry,'last_price':plan['entry'],'pnl':(plan['entry']-entry)*shares,'stop':plan['stop'],'take_profit':plan['target'],'weight':plan['weight']},['ticker'])
   self._execution(date,plan['ticker'],'entry','BUY',execution,shares,{'planned_transaction_cost':plan.get('transaction_cost',0.0)})
  self.db.upsert('paper_account',{'account_id':self.account_id,'starting_capital':account['starting_capital'],'cash':cash,'realized_pnl':account['realized_pnl'],'updated_at':self._now()},['account_id'])

 def snapshot(self,date):
  account=self._account();positions=self.db.rows('SELECT * FROM positions');market_value=sum(float(p['last_price'])*float(p['shares']) for p in positions);equity=float(account['cash'])+market_value
  previous=self.db.rows('SELECT equity FROM equity_curve WHERE date<? ORDER BY date DESC LIMIT 1',(date,));prior=float(previous[0]['equity']) if previous else float(account['starting_capital'])
  historical=[float(row['equity']) for row in self.db.rows('SELECT equity FROM equity_curve WHERE date<?',(date,))];peak=max(historical+[equity])
  exposure=market_value/max(equity,1.0)
  self.db.upsert('equity_curve',{'date':date,'equity':equity,'cash':account['cash'],'exposure':exposure,'daily_pnl':equity-prior,'drawdown':equity/peak-1.0},['date'])
  for position in positions:
   self.db.upsert('paper_positions_daily',{'date':date,'ticker':position['ticker'],'shares':position['shares'],'entry':position['entry'],'mark':position['last_price'],'market_value':position['last_price']*position['shares'],'unrealized_pnl':position['pnl'],'stop':position['stop'],'take_profit':position['take_profit'],'weight':position['weight']},['date','ticker'])
  return equity
