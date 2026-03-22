import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Universal Trading Workstation", layout="wide")


# -------------------------------
# Data / universe helpers
# -------------------------------
@st.cache_data(ttl=60 * 60)
def fetch_universe() -> List[str]:
    """Build a broad scan universe using S&P 500 + Nasdaq-100 members."""
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
    ndx = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]["Ticker"].tolist()
    cleaned = [s.replace(".", "-") for s in sp500 + ndx]
    return sorted(list(dict.fromkeys(cleaned)))


@st.cache_data(ttl=60 * 10)
def load_price_data(ticker: str, years: int = 2) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years + 30)
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, interval="1d")
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    return df


# -------------------------------
# Indicator calculations
# -------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).sum() / atr_s.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).sum() / atr_s.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.rolling(period).mean()


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def keltner(df: pd.DataFrame, period: int = 20, mult: float = 1.5):
    mid = ema(df["Close"], period)
    rng = atr(df, period)
    return mid + mult * rng, mid, mid - mult * rng


def stochastic(df: pd.DataFrame, period: int = 14, d_period: int = 3):
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    k = 100 * ((df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan))
    d = k.rolling(d_period).mean()
    return k, d


def williams_r(df: pd.DataFrame, period: int = 14):
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    return -100 * ((high_max - df["Close"]) / (high_max - low_min).replace(0, np.nan))


def roc(series: pd.Series, period: int = 12):
    return ((series - series.shift(period)) / series.shift(period)) * 100


def obv(df: pd.DataFrame):
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


def cmf(df: pd.DataFrame, period: int = 20):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfv = mfm * df["Volume"]
    return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, period: int = 14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = tp * df["Volume"]
    pos = np.where(tp > tp.shift(), money_flow, 0.0)
    neg = np.where(tp < tp.shift(), money_flow, 0.0)
    pos_mf = pd.Series(pos, index=df.index).rolling(period).sum()
    neg_mf = pd.Series(neg, index=df.index).rolling(period).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def parabolic_sar(df: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.2):
    high = df["High"].values
    low = df["Low"].values
    psar = np.zeros(len(df))
    bull = True
    af = af_start
    ep = high[0]
    psar[0] = low[0]

    for i in range(1, len(df)):
        psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
        if bull:
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
    return pd.Series(psar, index=df.index)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df["High"] + df["Low"]) / 2
    atr_val = atr(df, period)
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val

    st = pd.Series(index=df.index, dtype=float)
    trend_up = True
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > upper.iloc[i - 1]:
            trend_up = True
        elif df["Close"].iloc[i] < lower.iloc[i - 1]:
            trend_up = False
        st.iloc[i] = lower.iloc[i] if trend_up else upper.iloc[i]
    return st.ffill()


def ichimoku(df: pd.DataFrame):
    tenkan = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
    kijun = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, senkou_a, senkou_b


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    body = (df["Close"] - df["Open"]).abs()
    lower_shadow = np.minimum(df["Open"], df["Close"]) - df["Low"]
    upper_shadow = df["High"] - np.maximum(df["Open"], df["Close"])
    return (lower_shadow > 2 * body) & (upper_shadow < body)


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    bullish = (prev_close < prev_open) & (df["Close"] > df["Open"]) & (df["Close"] >= prev_open) & (df["Open"] <= prev_close)
    bearish = (prev_close > prev_open) & (df["Close"] < df["Open"]) & (df["Open"] >= prev_close) & (df["Close"] <= prev_open)
    return bullish | bearish


def anchored_vwap(df: pd.DataFrame, lookback: int = 120) -> pd.Series:
    window = df.tail(lookback)
    lows = window["Low"]
    swing_idx = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)].index
    anchor = swing_idx[-1] if len(swing_idx) else window["Low"].idxmin()
    anchor_pos = df.index.get_loc(anchor)
    sub = df.iloc[anchor_pos:]
    tp = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    cum_tp_vol = (tp * sub["Volume"]).cumsum()
    cum_vol = sub["Volume"].cumsum().replace(0, np.nan)
    avwap = cum_tp_vol / cum_vol
    out = pd.Series(index=df.index, dtype=float)
    out.iloc[anchor_pos:] = avwap
    return out


def fibonacci_levels(df: pd.DataFrame, lookback: int = 120) -> Dict[str, float]:
    sw = df.tail(lookback)
    high = sw["High"].max()
    low = sw["Low"].min()
    diff = high - low
    return {
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
    }


def add_all_indicators(df: pd.DataFrame, rsi_period: int, macd_fast: int, macd_slow: int, macd_signal: int = 9) -> pd.DataFrame:
    x = df.copy()
    x["EMA20"] = ema(x["Close"], 20)
    x["EMA50"] = ema(x["Close"], 50)
    x["EMA200"] = ema(x["Close"], 200)
    x["RSI"] = rsi(x["Close"], rsi_period)
    x["ATR"] = atr(x, 14)
    x["ADX"] = adx(x, 14)
    m_line, s_line, h = macd(x["Close"], macd_fast, macd_slow, macd_signal)
    x["MACD"], x["MACD_SIG"], x["MACD_HIST"] = m_line, s_line, h
    x["OBV"] = obv(x)
    x["CMF"] = cmf(x)
    x["MFI"] = mfi(x)
    x["STO_K"], x["STO_D"] = stochastic(x)
    x["WILLR"] = williams_r(x)
    x["ROC"] = roc(x["Close"])
    x["PSAR"] = parabolic_sar(x)
    x["SUPER"] = supertrend(x)
    x["BB_UP"], x["BB_MID"], x["BB_LOW"] = bollinger(x["Close"])
    x["KC_UP"], x["KC_MID"], x["KC_LOW"] = keltner(x)
    x["TENKAN"], x["KIJUN"], x["SENKOU_A"], x["SENKOU_B"] = ichimoku(x)
    x["HAMMER"] = detect_hammer(x)
    x["ENGULF"] = detect_engulfing(x)
    x["AVWAP"] = anchored_vwap(x)
    return x


# -------------------------------
# Scoring / strategy
# -------------------------------
SIGNAL_WEIGHTS = {
    "vwap": 8,
    "obv": 4,
    "cmf": 4,
    "mfi": 3,
    "trend_emas": 8,
    "adx": 5,
    "ichimoku": 5,
    "supertrend": 5,
    "psar": 3,
    "rsi": 6,
    "macd": 6,
    "stoch": 3,
    "williams": 2,
    "roc": 2,
    "bb": 3,
    "keltner": 3,
    "atr_trap": 2,
    "hammer": 2,
    "engulf": 2,
    "fibonacci": 2,
}


def conviction_score(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    score = pd.Series(50.0, index=df.index)
    reasons = []

    score += np.where(df["Close"] > df["AVWAP"], SIGNAL_WEIGHTS["vwap"], -SIGNAL_WEIGHTS["vwap"])
    score += np.where(df["OBV"] > df["OBV"].rolling(10).mean(), SIGNAL_WEIGHTS["obv"], -SIGNAL_WEIGHTS["obv"])
    score += np.where(df["CMF"] > 0, SIGNAL_WEIGHTS["cmf"], -SIGNAL_WEIGHTS["cmf"])
    score += np.where(df["MFI"] > 50, SIGNAL_WEIGHTS["mfi"], -SIGNAL_WEIGHTS["mfi"])

    ema_bull = (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"])
    score += np.where(ema_bull, SIGNAL_WEIGHTS["trend_emas"], -SIGNAL_WEIGHTS["trend_emas"])
    score += np.where(df["ADX"] > 22, SIGNAL_WEIGHTS["adx"], -SIGNAL_WEIGHTS["adx"] / 2)

    ich_bull = df["Close"] > np.maximum(df["SENKOU_A"], df["SENKOU_B"])
    score += np.where(ich_bull, SIGNAL_WEIGHTS["ichimoku"], -SIGNAL_WEIGHTS["ichimoku"])
    score += np.where(df["Close"] > df["SUPER"], SIGNAL_WEIGHTS["supertrend"], -SIGNAL_WEIGHTS["supertrend"])
    score += np.where(df["Close"] > df["PSAR"], SIGNAL_WEIGHTS["psar"], -SIGNAL_WEIGHTS["psar"])

    score += np.where((df["RSI"] > 50) & (df["RSI"] < 72), SIGNAL_WEIGHTS["rsi"], -SIGNAL_WEIGHTS["rsi"])
    score += np.where(df["MACD_HIST"] > 0, SIGNAL_WEIGHTS["macd"], -SIGNAL_WEIGHTS["macd"])
    score += np.where(df["STO_K"] > df["STO_D"], SIGNAL_WEIGHTS["stoch"], -SIGNAL_WEIGHTS["stoch"])
    score += np.where(df["WILLR"] > -50, SIGNAL_WEIGHTS["williams"], -SIGNAL_WEIGHTS["williams"])
    score += np.where(df["ROC"] > 0, SIGNAL_WEIGHTS["roc"], -SIGNAL_WEIGHTS["roc"])

    score += np.where(df["Close"] > df["BB_MID"], SIGNAL_WEIGHTS["bb"], -SIGNAL_WEIGHTS["bb"])
    score += np.where(df["Close"] > df["KC_MID"], SIGNAL_WEIGHTS["keltner"], -SIGNAL_WEIGHTS["keltner"])

    atr_pct = (df["ATR"] / df["Close"]).fillna(0)
    score += np.where(atr_pct < atr_pct.rolling(50).mean(), SIGNAL_WEIGHTS["atr_trap"], -SIGNAL_WEIGHTS["atr_trap"])

    score += np.where(df["HAMMER"], SIGNAL_WEIGHTS["hammer"], 0)
    score += np.where(df["ENGULF"], SIGNAL_WEIGHTS["engulf"], 0)

    fibs = fibonacci_levels(df)
    near_fib = (df["Close"] - fibs["0.618"]).abs() / df["Close"] < 0.01
    score += np.where(near_fib, SIGNAL_WEIGHTS["fibonacci"], 0)

    score = score.clip(0, 100)

    last = df.iloc[-1]
    if last["Close"] > last["AVWAP"]:
        reasons.append("Price is holding above anchored VWAP, confirming institution-level support.")
    if last["MACD_HIST"] > 0:
        reasons.append("MACD histogram is positive, signaling upside momentum expansion.")
    if last["ADX"] > 22:
        reasons.append("ADX above 22 suggests a sustained trend phase rather than chop.")
    if len(reasons) < 3 and last["CMF"] > 0:
        reasons.append("CMF is positive, indicating persistent net buying pressure.")
    if len(reasons) < 3 and last["Close"] > last["EMA50"]:
        reasons.append("Price remains above medium-term trend baseline (EMA50).")
    return score, reasons[:3]


def build_signals(df: pd.DataFrame, buy_threshold=68, sell_threshold=38) -> pd.Series:
    score, _ = conviction_score(df)
    signal = pd.Series(0, index=df.index)
    signal[(score > buy_threshold) & (score.shift(1) <= buy_threshold)] = 1
    signal[(score < sell_threshold) & (score.shift(1) >= sell_threshold)] = -1
    return signal


@dataclass
class StrategyMetrics:
    profit_factor: float
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_return: float
    buy_hold_return: float


def backtest_strategy(df: pd.DataFrame, signal: pd.Series) -> StrategyMetrics:
    position = signal.replace(0, np.nan).ffill().shift().fillna(0)
    position = np.where(position > 0, 1, 0)
    returns = df["Close"].pct_change().fillna(0)
    strat_ret = returns * position
    equity = (1 + strat_ret).cumprod()

    pnl = strat_ret[strat_ret != 0]
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    win_rate = (pnl > 0).mean() if len(pnl) else 0.0

    sharpe = 0.0
    if strat_ret.std() > 0:
        sharpe = math.sqrt(252) * strat_ret.mean() / strat_ret.std()

    running_max = equity.cummax()
    dd = (equity / running_max - 1).min()
    total_return = equity.iloc[-1] - 1
    buy_hold = (1 + returns).cumprod().iloc[-1] - 1

    return StrategyMetrics(
        profit_factor=float(profit_factor if np.isfinite(profit_factor) else 999.0),
        win_rate=float(win_rate),
        sharpe=float(sharpe),
        max_drawdown=float(dd),
        total_return=float(total_return),
        buy_hold_return=float(buy_hold),
    )


def optimize_params(raw: pd.DataFrame) -> Tuple[Dict[str, int], StrategyMetrics, pd.DataFrame]:
    search = [
        {"rsi": 9, "macd_fast": 8, "macd_slow": 21},
        {"rsi": 9, "macd_fast": 12, "macd_slow": 26},
        {"rsi": 14, "macd_fast": 12, "macd_slow": 26},
        {"rsi": 14, "macd_fast": 10, "macd_slow": 24},
    ]
    best = None
    best_metrics = None
    best_df = None

    for params in search:
        tmp = add_all_indicators(raw, params["rsi"], params["macd_fast"], params["macd_slow"])
        sig = build_signals(tmp)
        metrics = backtest_strategy(tmp, sig)
        objective = metrics.profit_factor * 0.6 + metrics.win_rate * 100 * 0.4
        if (best is None) or (objective > best):
            best = objective
            best_metrics = metrics
            best_df = tmp
            best_params = params
    return best_params, best_metrics, best_df


def verdict_from_score(score: float) -> str:
    if score >= 72:
        return "STRONG BUY"
    if score <= 32:
        return "STRONG SELL"
    return "NEUTRAL"


def render_chart(df: pd.DataFrame, show_extra: bool):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(width=1.2, color="#F7B500")))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(width=1.2, color="#00C2FF")))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(width=1.2, color="#A855F7")))
    fig.add_trace(go.Scatter(x=df.index, y=df["AVWAP"], name="Anchored VWAP", line=dict(width=1.5, color="#00FF9C")))

    if show_extra:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], name="BB Upper", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="BB Lower", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["KC_UP"], name="KC Upper", line=dict(width=1, dash="dash")))
        fig.add_trace(go.Scatter(x=df.index, y=df["KC_LOW"], name="KC Lower", line=dict(width=1, dash="dash")))

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(t=25, b=20, l=10, r=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def risk_levels(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    entry = last["Close"]
    vol = last["ATR"]
    return {
        "entry": float(entry),
        "stop": float(entry - 2 * vol),
        "tp1": float(entry + 1.5 * vol),
        "tp2": float(entry + 3.0 * vol),
        "tp3": float(entry + 4.5 * vol),
    }


def relative_strength_score(df: pd.DataFrame, spy: pd.DataFrame) -> float:
    aligned = pd.DataFrame({"stk": df["Close"], "spy": spy["Close"]}).dropna()
    if len(aligned) < 60:
        return -999
    stk_ret = aligned["stk"].iloc[-1] / aligned["stk"].iloc[-60] - 1
    spy_ret = aligned["spy"].iloc[-1] / aligned["spy"].iloc[-60] - 1
    return float(stk_ret - spy_ret)


def run_single_ticker(ticker: str):
    raw = load_price_data(ticker, years=2)
    if raw.empty or len(raw) < 260:
        st.error("Insufficient data for this ticker.")
        return

    with st.status("Optimization Loop: calibrating indicator parameters on the last 2 years...", expanded=False) as status:
        params, metrics, df = optimize_params(raw)
        score_series, reasons = conviction_score(df)
        status.update(label="Optimization complete", state="complete")

    current_score = float(score_series.iloc[-1])
    verdict = verdict_from_score(current_score)

    left, right = st.columns([2.8, 1.2])
    with left:
        show_extra = st.toggle("Show secondary overlays (BB/Keltner)", value=False)
        render_chart(df.tail(250), show_extra)

    with right:
        st.markdown("### Action Card")
        st.metric("Verdict", verdict)
        st.metric("Conviction Score", f"{current_score:.1f}/100")
        st.write("**Top Mathematical Reasons**")
        for i, r in enumerate(reasons, start=1):
            st.write(f"{i}. {r}")

        rl = risk_levels(df)
        st.write("**Risk Management**")
        st.write(f"Stop Loss (2x ATR): **${rl['stop']:.2f}**")
        st.write(f"Target 1: **${rl['tp1']:.2f}**")
        st.write(f"Target 2: **${rl['tp2']:.2f}**")
        st.write(f"Target 3: **${rl['tp3']:.2f}**")

        st.write("**Optimized Parameters**")
        st.json(params)

        st.write("**Backtest Summary**")
        st.write(f"Profit Factor: **{metrics.profit_factor:.2f}**")
        st.write(f"Win Rate: **{metrics.win_rate * 100:.1f}%**")
        st.write(f"Sharpe Ratio: **{metrics.sharpe:.2f}**")
        st.write(f"Max Drawdown: **{metrics.max_drawdown * 100:.1f}%**")
        st.write(f"Total Return: **{metrics.total_return * 100:.1f}%**")
        st.write(f"Buy & Hold Return: **{metrics.buy_hold_return * 100:.1f}%**")



def run_scanner():
    st.write("Running standout scanner across S&P 500 + Nasdaq-100 universe...")
    universe = fetch_universe()
    st.caption(f"Universe size: {len(universe)} symbols")

    spy = load_price_data("SPY", years=1)
    rows = []
    progress = st.progress(0)

    for i, t in enumerate(universe):
        df = load_price_data(t, years=1)
        if df.empty or len(df) < 220:
            progress.progress((i + 1) / len(universe))
            continue

        enriched = add_all_indicators(df, 14, 12, 26)
        score, _ = conviction_score(enriched)
        rs = relative_strength_score(df, spy)

        breakout = (enriched["Close"].iloc[-1] > enriched["High"].rolling(20).max().shift(1).iloc[-1])
        if breakout and score.iloc[-1] > 65 and rs > 0:
            rows.append(
                {
                    "Ticker": t,
                    "Conviction": round(float(score.iloc[-1]), 2),
                    "RelStrength(60d)": round(rs * 100, 2),
                    "Price": round(float(df["Close"].iloc[-1]), 2),
                }
            )
        progress.progress((i + 1) / len(universe))

    if not rows:
        st.warning("No high-conviction breakouts found today.")
        return

    top = pd.DataFrame(rows).sort_values(["Conviction", "RelStrength(60d)"], ascending=False).head(5)
    st.subheader("Top 5 High-Conviction Picks")
    st.dataframe(top, use_container_width=True)


# -------------------------------
# App shell
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {background: radial-gradient(circle at top, #111827 0%, #05070d 55%, #04050a 100%); color: #E5E7EB;}
    .block-container {padding-top: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Universal Trading Workstation")
st.caption("Professional Swing Trading Dashboard | Optimized Signal Engine")

mode = st.sidebar.radio("Startup Intelligence", ["Single Ticker", "Scan Mode"], index=0)

if mode == "Single Ticker":
    ticker = st.sidebar.text_input("Enter ticker", value="AAPL").upper().strip()
    if st.sidebar.button("Run Workstation", use_container_width=True):
        run_single_ticker(ticker)
else:
    if st.sidebar.button("Run Standout Scanner", use_container_width=True):
        run_scanner()
