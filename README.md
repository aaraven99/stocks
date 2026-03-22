# Universal Trading Workstation

Standalone Python/Streamlit swing-trading dashboard with:

- Startup mode for **single ticker analysis** or **S&P 500 + Nasdaq-100 scan mode**.
- A built-in **optimization loop** that calibrates RSI/MACD parameter sets using the last ~2 years and picks the best Profit Factor + Win Rate blend.
- A dark, low-clutter interactive candlestick chart with EMA20/50/200 + anchored VWAP.
- A 25+ signal **conviction engine** (0-100) spanning volume/flow, trend, momentum, volatility, and price action.
- Scanner output for **Top 5 High-Conviction Picks** using relative strength vs SPY + breakout confirmation.
- Risk card with ATR-based stop and 3 targets + strategy diagnostics.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

Streamlit Community Cloud deploys from **GitHub only** (not local folders).  
If you see: `The app’s code is not connected to a remote GitHub repository`, do this:

```bash
# from this repo folder
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin work
```

Then in Streamlit Community Cloud:
1. Click **New app**
2. Select your repository and branch (`work` or `main`)
3. Set **Main file path** to `app.py`
4. Click **Deploy**

## Notes

- Market data source: Yahoo Finance via `yfinance`.
- Universe source: live S&P 500 and Nasdaq-100 tables from Wikipedia.
- If remote universe sources are unavailable (rate limits/network blocks), scanner falls back to a curated liquid-ticker universe so the app stays operational.
- Scanner can take several minutes due to broad universe size and per-symbol indicator computation.
