"""Opt-in live paper-account lifecycle coverage; never runs in ordinary CI."""
import os,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storage.db import Database
from paper_trading.alpaca_broker import AlpacaPaperBroker


@unittest.skipUnless(
    os.getenv('RUN_ALPACA_INTEGRATION') == '1'
    and os.getenv('ALPACA_API_KEY')
    and os.getenv('ALPACA_SECRET_KEY'),
    'set RUN_ALPACA_INTEGRATION=1 and Alpaca paper credentials to run',
)
class AlpacaPaperLifecycleTests(unittest.TestCase):
    def test_reconcile_submit_persist_cancel(self):
        db=Database(Path.cwd()/'artifacts'/'test_dbs'/'alpaca_integration.sqlite')
        broker=AlpacaPaperBroker(db)
        self.assertTrue(broker.reconcile('integration')['enabled'])
        market_price=broker.latest_trade_price('SPY')
        self.assertGreater(market_price,0)
        plan={'ticker':'SPY','action':'LONG','shares':1,'entry':market_price,'stop':round(market_price*.90,2),'target':round(market_price*1.10,2)}
        submitted=broker.submit_long_bracket('integration',plan)
        self.assertTrue(submitted['submitted'])
        broker.reconcile('integration')
        row=db.rows('SELECT * FROM broker_orders WHERE client_order_id=?',(submitted['client_order_id'],))[0]
        self.assertIn(row['status'],{'new','accepted','pending_new','partially_filled','filled','done_for_day','canceled','replaced'})
        if row['broker_order_id'] and row['status'] not in {'filled','canceled','replaced'}:
            broker.cancel_order(row['broker_order_id'])
            final=broker.get_order(row['broker_order_id'])
            self.assertIn(final['status'],{'canceled','filled','replaced'})
            persisted=db.rows('SELECT status FROM broker_orders WHERE broker_order_id=?',(row['broker_order_id'],))[0]
            self.assertEqual(persisted['status'],final['status'])


if __name__=='__main__': unittest.main()
