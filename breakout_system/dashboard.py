from __future__ import annotations

import time
from typing import List

import pandas as pd
import streamlit as st

from .demo import SimulatedSource, build_historical
from .engine import Backtester, RealTimeBreakoutEngine


st.set_page_config(page_title="Breakout ML Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .big-title {font-size: 2rem; font-weight: 700; color: #38bdf8;}
      .pill {display:inline-block;padding:.2rem .6rem;border-radius:999px;background:#1f2937;color:#f8fafc;margin-right:.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">📊 Real-Time ML Breakout Dashboard</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    tickers_text = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,AMZN")
    bars = st.slider("Simulated stream bars", min_value=100, max_value=1200, value=400, step=50)
    refresh = st.slider("Refresh interval (seconds)", min_value=1, max_value=10, value=2)
    run = st.button("Run Simulation")

if "active_df" not in st.session_state:
    st.session_state.active_df = pd.DataFrame()
    st.session_state.watch_df = pd.DataFrame()
    st.session_state.metrics = {}

if run:
    tickers: List[str] = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    historical = build_historical(tickers, bars=max(1000, bars * 3))

    engine = RealTimeBreakoutEngine(timeframe="5m")
    precision = engine.fit(historical)
    src = SimulatedSource(tickers=tickers, bars=bars)

    for bar in src.stream():
        engine.on_bar(bar)
        engine.invalidate_broken_structures()

    st.session_state.active_df = engine.active_setups_frame()
    st.session_state.watch_df = engine.watchlist_frame()
    st.session_state.metrics = {
        "Model OOS Precision": round(precision, 3),
        "Walk-Forward Precision": round(engine.model.walk_forward_precision, 3),
        "Walk-Forward Sharpe": round(engine.model.walk_forward_sharpe, 3),
        "Model Ready": engine.model.trained,
    }
    st.session_state.metrics.update(Backtester().run(engine.active_signals, historical))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model OOS Precision", st.session_state.metrics.get("Model OOS Precision", "-"))
col2.metric("WF Precision", st.session_state.metrics.get("Walk-Forward Precision", "-"))
col3.metric("WF Sharpe", st.session_state.metrics.get("Walk-Forward Sharpe", "-"))
col4.metric("Model Ready", st.session_state.metrics.get("Model Ready", "-"))

st.markdown('<span class="pill">High-Confidence Active Setups</span><span class="pill">Watchlist (Forming)</span>', unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    st.subheader("✅ High-Confidence Active Setups")
    active_df = st.session_state.active_df.copy()
    if not active_df.empty:
        active_df = active_df.sort_values("ml_probability", ascending=False)
    st.dataframe(active_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("👀 Watchlist (Forming Patterns)")
    watch_df = st.session_state.watch_df.copy()
    st.dataframe(watch_df.tail(100), use_container_width=True, hide_index=True)

st.subheader("📈 Backtest Snapshot (from current active signals)")
if st.session_state.metrics:
    backtest_cols = ["win_rate", "avg_r", "max_drawdown", "sharpe"]
    st.dataframe(pd.DataFrame([{k: st.session_state.metrics.get(k, None) for k in backtest_cols}]), hide_index=True)
else:
    st.info("Press 'Run Simulation' to generate dashboard results.")

if run:
    with st.spinner("Refreshing dashboard..."):
        time.sleep(refresh)
