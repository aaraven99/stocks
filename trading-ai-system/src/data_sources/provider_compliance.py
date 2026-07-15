"""Data-provider governance declarations used in audit reports."""
import os

def provider_compliance():
    finnhub=bool(os.getenv('FINNHUB_API_KEY')); polygon=bool(os.getenv('POLYGON_API_KEY')); fred=bool(os.getenv('FRED_API_KEY'))
    return {
    'price_history': {'status':'provider-backed', 'provider':'Yahoo Finance via yfinance', 'decision_use':True},
    'universe': {'status':'provider-backed', 'provider':'Nasdaq Trader snapshot plus Nasdaq/NYSE runtime retrieval', 'decision_use':True},
    'macro': {'status':'provider-backed' if fred else 'unavailable-provider-credentials', 'provider':'FRED API', 'decision_use':fred},
    'news_sentiment': {'status':'provider-backed' if finnhub else 'unavailable-provider-credentials', 'provider':'Finnhub company news', 'decision_use':finnhub},
    'options_flow': {'status':'provider-backed' if polygon else 'unavailable-provider-credentials', 'provider':'Polygon options snapshot', 'decision_use':polygon},
    'earnings_calendar': {'status':'provider-backed' if finnhub else 'unavailable-provider-credentials', 'provider':'Finnhub earnings calendar', 'decision_use':finnhub},
    'market_breadth': {'status':'derived-from-price-data', 'provider':'loaded OHLCV universe', 'decision_use':True},
    }

PROVIDER_COMPLIANCE=provider_compliance()
