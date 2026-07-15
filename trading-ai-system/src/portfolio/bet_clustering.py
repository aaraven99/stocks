def cluster_metrics(positions):
 sectors={};
 for p in positions:sectors[p.get('sector','Unknown')]=sectors.get(p.get('sector','Unknown'),0)+p.get('weight',0)
 return {'largest_sector_weight':max(sectors.values(),default=0),'sector_count':len(sectors),'sectors':sectors}
