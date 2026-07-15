"""Data-provider governance declarations used in audit reports."""

PROVIDER_COMPLIANCE={
    'price_history': {'status':'provider-backed', 'provider':'Yahoo Finance via yfinance', 'decision_use':True},
    'universe': {'status':'provider-backed', 'provider':'Nasdaq Trader snapshot plus Nasdaq/NYSE runtime retrieval', 'decision_use':True},
    'macro': {'status':'provider-backed with deterministic fallback', 'provider':'FRED', 'decision_use':True},
    'news_sentiment': {'status':'proxy-non-compliant', 'provider':'deterministic fallback until licensed news feed is configured', 'decision_use':False},
    'options_flow': {'status':'proxy-non-compliant', 'provider':'heuristic proxy until OPRA/vendor feed is configured', 'decision_use':False},
    'earnings_calendar': {'status':'proxy-non-compliant', 'provider':'heuristic calendar proxy until licensed corporate-actions feed is configured', 'decision_use':False},
    'market_breadth': {'status':'derived-from-price-data', 'provider':'loaded OHLCV universe', 'decision_use':True},
}
