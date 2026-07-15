import sqlite3,json
from pathlib import Path
from datetime import datetime,timezone
from core.constants import DB_PATH, ROOT
class Database:
 def __init__(self,path=None):
  self.path=Path(path or DB_PATH); self.path.parent.mkdir(parents=True,exist_ok=True); self.init()
 def connect(self):
  c=sqlite3.connect(self.path); c.row_factory=sqlite3.Row; return c
 def init(self):
  with self.connect() as c: c.executescript((ROOT/'src/storage/schema.sql').read_text())
 def execute(self,sql,params=()):
  try:
   with self.connect() as c: c.execute(sql,params)
  except sqlite3.OperationalError as exc:
   raise sqlite3.OperationalError(f'{exc}; sql={sql}') from exc
 def upsert(self,table,row,keys):
  cols=list(row); update=[c for c in cols if c not in keys]
  sql=f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)}) ON CONFLICT({','.join(keys)}) DO UPDATE SET "+','.join(f'{c}=excluded.{c}' for c in update)
  self.execute(sql,[json.dumps(row[c],default=str) if isinstance(row[c],(dict,list)) else row[c] for c in cols])
 def rows(self,sql,params=()):
  with self.connect() as c:return [dict(r) for r in c.execute(sql,params).fetchall()]
 def log(self,kind,message,details=None): self.execute('INSERT INTO journal_logs(ts,kind,message,details_json) VALUES(?,?,?,?)',(datetime.now(timezone.utc).isoformat(),kind,message,json.dumps(details or {},default=str)))
 def integrity(self):
  with self.connect() as c:return c.execute('PRAGMA integrity_check').fetchone()[0]
