"""Deterministic paper fills with explicit half-spread, slippage, and impact."""

def execution_fill(price,side,shares=0,dollar_volume=None,spread_bps=8,slippage_bps=5):
 price=float(price);shares=abs(float(shares));side=side.upper()
 direction=1 if side in ('BUY','LONG') else -1
 participation=(price*shares/max(float(dollar_volume or price*shares*1000),1.0))
 impact_bps=min(50.0,2.0+35.0*participation**0.5)
 half_spread=float(spread_bps)/2.0
 total_bps=half_spread+float(slippage_bps)+impact_bps
 fill_price=price*(1+direction*total_bps/10000.0)
 return {'reference_price':price,'fill_price':fill_price,'spread_cost':price*shares*half_spread/10000.0,'slippage_cost':price*shares*(float(slippage_bps)+impact_bps)/10000.0,'impact_bps':impact_bps,'total_bps':total_bps}

def fill(price,side,slippage_bps=5):
 """Backward-compatible scalar fill used by older callers."""
 return execution_fill(price,side,slippage_bps=slippage_bps)['fill_price']
