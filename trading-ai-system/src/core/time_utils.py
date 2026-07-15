from datetime import datetime, time
from zoneinfo import ZoneInfo
def now_local(): return datetime.now(ZoneInfo('America/Chicago'))
def trading_date(): return now_local().date().isoformat()
def is_weekday(dt=None): return (dt or now_local()).weekday()<5
def email_due(dt=None): return (dt or now_local()).time() >= time(7,0)
