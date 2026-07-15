import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from storage.db import Database
def main():
 db=Database(); tables=db.rows("SELECT name FROM sqlite_master WHERE type='table'"); ok=db.integrity()=='ok' and len(tables)>20; print({'integrity':db.integrity(),'tables':len(tables),'ok':ok});return 0 if ok else 2
if __name__=='__main__':raise SystemExit(main())
