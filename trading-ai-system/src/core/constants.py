from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts'
DB_PATH = ARTIFACTS / 'trading_system.sqlite'
CHICAGO_TZ = 'America/Chicago'
SEED = 42
BENCHMARKS = ['SPY','QQQ','IWM','DIA','TLT','GLD','USO','BTC-USD','^VIX']
REGIMES = ['bull_low_vol','bull_high_vol','bear_high_vol','sideways_chop']
