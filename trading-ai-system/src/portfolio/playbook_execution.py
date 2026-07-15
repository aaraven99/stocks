def execution_plan(action,price,stop_pct,target_pct): return {'action':action,'entry':price,'stop':price*(1-stop_pct),'target':price*(1+target_pct),'order_type':'limit_with_protective_stop'}
