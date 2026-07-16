import os,sqlite3,json
from pathlib import Path
from datetime import datetime,timezone
from core.constants import DB_PATH, ROOT
class Database:
 def __init__(self,path=None):
  if path is None:
   configured=os.getenv('TRADING_DB_PATH')
   test_mode=os.getenv('TEST_MODE','').lower() in ('1','true','yes')
   path=Path(configured) if configured else (ROOT/'artifacts/test_mode/trading_system_test.sqlite' if test_mode else DB_PATH)
  self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.init()
 def connect(self):
  c=sqlite3.connect(self.path,timeout=30); c.row_factory=sqlite3.Row
  c.execute('PRAGMA foreign_keys=ON');c.execute('PRAGMA busy_timeout=30000')
  return c
 def init(self):
  with self.connect() as c:
   c.executescript((ROOT/'src/storage/schema.sql').read_text())
   # Additive migrations keep existing action artifacts usable after schema evolution.
   for table,column,definition in [('backtest_trades','holding_days','INTEGER'),('backtest_trades','notional_fraction','REAL')]:
    columns={row[1] for row in c.execute(f'PRAGMA table_info({table})').fetchall()}
    if column not in columns:c.execute(f'ALTER TABLE {table} ADD COLUMN {column} {definition}')
 def execute(self,sql,params=()):
  try:
   with self.connect() as c: c.execute(sql,params)
  except sqlite3.OperationalError as exc:
   raise sqlite3.OperationalError(f'{exc}; sql={sql}') from exc
 def upsert(self,table,row,keys):
  self.upsert_many(table,[row],keys)
 def upsert_many(self,table,rows,keys):
  """Atomically upsert a batch; this keeps large historical backfills bounded."""
  if not rows:return
  cols=list(rows[0]); update=[c for c in cols if c not in keys]
  sql=f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)}) ON CONFLICT({','.join(keys)}) DO UPDATE SET "+','.join(f'{c}=excluded.{c}' for c in update)
  values=[[json.dumps(row[c],default=str) if isinstance(row[c],(dict,list)) else row[c] for c in cols] for row in rows]
  try:
   with self.connect() as connection:connection.executemany(sql,values)
  except sqlite3.OperationalError as exc:
   raise sqlite3.OperationalError(f'{exc}; sql={sql}') from exc
 def rows(self,sql,params=()):
  with self.connect() as c:return [dict(r) for r in c.execute(sql,params).fetchall()]
 def log(self,kind,message,details=None): self.execute('INSERT INTO journal_logs(ts,kind,message,details_json) VALUES(?,?,?,?)',(datetime.now(timezone.utc).isoformat(),kind,message,json.dumps(details or {},default=str)))
 def quarantine(self,source_table,record_key,reason,payload=None):
  """Persist rejected source records instead of silently dropping them."""
  self.execute('INSERT INTO data_quarantine(source_table,record_key,reason,payload_json,quarantined_at) VALUES(?,?,?,?,?)',(str(source_table),str(record_key),str(reason),json.dumps(payload or {},default=str),datetime.now(timezone.utc).isoformat()))
 def integrity(self):
  with self.connect() as c:return c.execute('PRAGMA integrity_check').fetchone()[0]
