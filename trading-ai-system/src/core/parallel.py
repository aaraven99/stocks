from concurrent.futures import ThreadPoolExecutor, as_completed
def map_resilient(fn, items, workers=8):
 out=[]
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(fn,x):x for x in items}
  for future in as_completed(futures):
   try: out.append((futures[future],future.result(),None))
   except Exception as exc: out.append((futures[future],None,str(exc)))
 return out
