def allowed(sector_weights,sector,new_weight,limit=.25): return sector_weights.get(sector,0)+new_weight<=limit
