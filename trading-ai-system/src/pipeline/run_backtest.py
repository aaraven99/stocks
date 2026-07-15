import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from storage.db import Database
from backtesting.performance_metrics import calculate
def main():
 db=Database(); rows=db.rows('SELECT daily_pnl,equity FROM equity_curve ORDER BY date'); r=[x['daily_pnl']/max(x['equity'],1) for x in rows]; print(calculate(r))
if __name__=='__main__':main()
