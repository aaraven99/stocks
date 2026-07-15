import numpy as np
def metrics(trades):
 p=[t['pnl'] for t in trades if t.get('pnl') is not None];gross_win=sum(x for x in p if x>0);gross_loss=abs(sum(x for x in p if x<0)); return {'closed_trades':len(p),'win_rate':float(np.mean([x>0 for x in p])) if p else 0.,'avg_trade_pnl':float(np.mean(p)) if p else 0.,'profit_factor':float(gross_win/gross_loss) if gross_loss else None}
