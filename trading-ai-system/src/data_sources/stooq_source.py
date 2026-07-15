from data_sources.yfinance_source import fetch_ohlcv
def fetch_stooq(ticker, test_mode=False): return fetch_ohlcv(ticker,test_mode=test_mode)
