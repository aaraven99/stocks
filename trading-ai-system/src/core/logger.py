import json, logging, sys
from datetime import datetime, timezone
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({'ts': datetime.now(timezone.utc).isoformat(), 'level': record.levelname, 'stage': getattr(record,'stage','system'), 'message': record.getMessage()})
def get_logger(name='trading_ai'):
    logger=logging.getLogger(name)
    if not logger.handlers:
        h=logging.StreamHandler(sys.stdout); h.setFormatter(JsonFormatter()); logger.addHandler(h); logger.setLevel(logging.INFO); logger.propagate=False
    return logger
def stage(logger, name): return logging.LoggerAdapter(logger, {'stage':name})
