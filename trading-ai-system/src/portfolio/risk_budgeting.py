def inverse_vol_weights(items):
 inv={x['ticker']:1/max(x.get('volatility',.2),.02) for x in items};z=sum(inv.values())or 1;return {k:v/z for k,v in inv.items()}
