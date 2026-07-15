import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from features import indicators_trend,indicators_momentum,indicators_volatility,indicators_volume,indicators_orderflow_proxy,indicators_smc,indicators_patterns,indicators_candles,indicators_harmonics,indicators_pivots,indicators_breadth,indicators_macro,indicators_sentiment,indicators_theory,indicators_cross_asset,indicators_event_risk,indicators_discovered,indicators_intraday_proxy
from features.indicator_registry import calculate_all,definitions
def make_features(df,context=None):
 out=calculate_all(df,context); out={k:float(v) for k,v in out.items()}; return out
def feature_definitions(): return definitions()
