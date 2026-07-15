def trade_attribution(trade): return {'ticker':trade['ticker'],'pnl':trade.get('pnl',0),'reason':trade.get('reason','signal')}
