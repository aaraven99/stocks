from backtesting.performance_metrics import calculate
def run(signals,returns): return calculate([s*r for s,r in zip(signals,returns)])
