def fill(price,side,slippage_bps=5): return price*(1+slippage_bps/10000*(1 if side=='LONG' else -1))
