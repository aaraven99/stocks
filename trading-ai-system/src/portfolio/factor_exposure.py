def aggregate(positions,factors): return {k:sum(p.get('weight',0)*factors.get(p['ticker'],{}).get(k,0) for p in positions) for k in ['alpha','beta']}
