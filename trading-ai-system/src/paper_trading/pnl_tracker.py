def pnl(entry,last,shares,side='LONG'): return (last-entry)*shares*(1 if side=='LONG' else -1)
