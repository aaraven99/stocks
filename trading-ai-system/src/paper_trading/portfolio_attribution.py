def summarize(trades): return {'total_pnl':sum(t.get('pnl',0) or 0 for t in trades),'count':len(trades)}
