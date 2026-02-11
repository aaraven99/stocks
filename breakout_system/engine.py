from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler


@dataclass
class Bar:
    ticker: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    ticker: str
    timeframe: str
    pattern: str
    entry: float
    stop: float
    target: float
    rr: float
    quantity: int
    ml_probability: float
    timestamp: pd.Timestamp
    status: str = "active"


class StreamingOHLCVSource:
    """Near-real-time feed abstraction."""

    def stream(self) -> Iterator[Bar]:
        raise NotImplementedError


class DataBuffer:
    def __init__(self, maxlen: int = 700):
        self.maxlen = maxlen
        self.frames: Dict[str, pd.DataFrame] = {}

    def update(self, bar: Bar) -> pd.DataFrame:
        row = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            ]
        )
        df = self.frames.get(bar.ticker)
        df = row if df is None else pd.concat([df, row], ignore_index=True)
        if len(df) > self.maxlen:
            df = df.iloc[-self.maxlen :].reset_index(drop=True)
        self.frames[bar.ticker] = df
        return df


class IndicatorEngine:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        delta = out["close"].diff()
        gain = pd.Series(np.where(delta > 0, delta, 0.0))
        loss = pd.Series(np.where(delta < 0, -delta, 0.0))
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out["rsi"] = 100 - (100 / (1 + rs))

        ema12 = out["close"].ewm(span=12, adjust=False).mean()
        ema26 = out["close"].ewm(span=26, adjust=False).mean()
        out["macd"] = ema12 - ema26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

        out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["sma50"] = out["close"].rolling(50).mean()

        tr = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - out["close"].shift(1)).abs(),
                (out["low"] - out["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(14).mean()

        out["vol_sma20"] = out["volume"].rolling(20).mean()
        out["volume_breakout_ratio"] = out["volume"] / out["vol_sma20"]
        out["distance_to_ema20"] = (out["close"] - out["ema20"]) / out["ema20"]
        out["distance_to_sma50"] = (out["close"] - out["sma50"]) / out["sma50"]

        out["htf_close"] = out["close"].rolling(12).mean()
        out["htf_sma"] = out["htf_close"].rolling(20).mean()
        out["htf_trend_alignment"] = (out["htf_close"] > out["htf_sma"]).astype(float)
        return out


class PatternDetector:
    def __init__(self, min_volume_ratio: float = 1.5):
        self.min_volume_ratio = min_volume_ratio

    def detect(self, df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
        if len(df) < 90:
            return None

        recent = df.iloc[-70:].copy()
        close = recent["close"].to_numpy()
        high = recent["high"].to_numpy()
        low = recent["low"].to_numpy()

        resistance = float(np.max(high[-30:-1]))
        breakout = close[-1] > resistance
        vol_confirm = recent["volume_breakout_ratio"].iloc[-1] >= self.min_volume_ratio
        if not (breakout and vol_confirm):
            return None

        candidates = [
            self._ascending_triangle(low, high),
            self._double_bottom(low, close),
            self._bullish_flag(close),
            self._cup_handle(close),
            self._falling_wedge(high, low),
        ]

        for name, valid, invalidation in candidates:
            if valid:
                return name, resistance, invalidation
        return None

    def _ascending_triangle(self, low: np.ndarray, high: np.ndarray) -> Tuple[str, bool, float]:
        plateau = np.std(high[-25:]) / max(np.mean(high[-25:]), 1e-9) < 0.012
        rising_lows = np.polyfit(np.arange(25), low[-25:], 1)[0] > 0
        return "ascending_triangle", bool(plateau and rising_lows), float(np.min(low[-25:]))

    def _double_bottom(self, low: np.ndarray, close: np.ndarray) -> Tuple[str, bool, float]:
        trough_window = low[-50:]
        idx = np.argpartition(trough_window, 2)[:2]
        t1, t2 = sorted(trough_window[idx])
        similar_depth = abs(t1 - t2) / max(t1, 1e-9) <= 0.02
        neckline_recovery = close[-1] > np.mean(close[-12:-2])
        return "double_bottom", bool(similar_depth and neckline_recovery), float(min(t1, t2))

    def _bullish_flag(self, close: np.ndarray) -> Tuple[str, bool, float]:
        pole = close[-50:-25]
        flag = close[-25:-1]
        strong_pole = (pole[-1] - pole[0]) / max(pole[0], 1e-9) >= 0.06
        mild_pullback = np.polyfit(np.arange(len(flag)), flag, 1)[0] < 0
        contained = np.max(flag) < close[-1]
        return "bullish_flag", bool(strong_pole and mild_pullback and contained), float(np.min(flag))

    def _cup_handle(self, close: np.ndarray) -> Tuple[str, bool, float]:
        cup = close[-70:-18]
        handle = close[-18:-1]
        left, trough, right = cup[0], float(np.min(cup)), cup[-1]
        u_shape = trough < left and trough < right and abs(left - right) / max(left, 1e-9) <= 0.06
        shallow_handle = (np.max(handle) - np.min(handle)) / max(np.max(handle), 1e-9) <= 0.08
        return "cup_handle", bool(u_shape and shallow_handle), float(np.min(handle))

    def _falling_wedge(self, high: np.ndarray, low: np.ndarray) -> Tuple[str, bool, float]:
        hs = np.polyfit(np.arange(35), high[-35:], 1)[0]
        ls = np.polyfit(np.arange(35), low[-35:], 1)[0]
        converging = hs < 0 and ls < 0 and abs(hs) > abs(ls)
        return "falling_wedge", bool(converging), float(np.min(low[-35:]))


class MarketRegimeFilter:
    def favorable(self, asset_df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> bool:
        if len(asset_df) < 60:
            return False

        trend = asset_df["close"].iloc[-1] > asset_df["sma50"].iloc[-1]
        momentum = asset_df["macd"].iloc[-1] > asset_df["macd_signal"].iloc[-1]
        low_chop = asset_df["atr"].iloc[-1] / max(asset_df["close"].iloc[-1], 1e-9) > 0.004

        if index_df is None or len(index_df) < 60:
            index_ok = True
        else:
            idx = IndicatorEngine.add_indicators(index_df)
            index_ok = idx["close"].iloc[-1] > idx["sma50"].iloc[-1]

        return bool(trend and momentum and low_chop and index_ok)


class FeatureBuilder:
    FEATURES = [
        "rsi",
        "macd",
        "ema20",
        "sma50",
        "atr",
        "volume_breakout_ratio",
        "htf_trend_alignment",
        "distance_to_ema20",
        "distance_to_sma50",
    ]

    def latest(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[[-1]][self.FEATURES].copy()


class LabelBuilder:
    @staticmethod
    def build(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
        closes = df["close"].to_numpy()
        atr = df["atr"].to_numpy()
        labels = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue
            entry = closes[i]
            stop = entry - atr[i]
            target = entry + 2 * (entry - stop)
            future = closes[i + 1 : i + 1 + horizon]
            hit_target = np.where(future >= target)[0]
            hit_stop = np.where(future <= stop)[0]

            if len(hit_target) == 0 and len(hit_stop) == 0:
                labels[i] = 0
            elif len(hit_target) > 0 and (len(hit_stop) == 0 or hit_target[0] < hit_stop[0]):
                labels[i] = 1
            else:
                labels[i] = 0

        return pd.Series(labels, index=df.index)


class ModelManager:
    def __init__(self, learning_rate: float = 0.05, n_estimators: int = 250, max_depth: int = 2) -> None:
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            random_state=42,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        self.trained = False
        self.precision_oos: float = 0.0
        self.walk_forward_precision: float = 0.0
        self.walk_forward_sharpe: float = 0.0

    @staticmethod
    def _trade_r_returns(proba: np.ndarray, y_true: np.ndarray, threshold: float = 0.75) -> np.ndarray:
        take = proba >= threshold
        if not np.any(take):
            return np.array([0.0])
        return np.where(y_true[take] == 1, 2.0, -1.0)

    def train_with_walk_forward(self, frame: pd.DataFrame, features: List[str]) -> float:
        data = frame.dropna(subset=features + ["label"]).reset_index(drop=True)
        if len(data) < 400:
            self.trained = False
            return 0.0

        split = int(len(data) * 0.7)
        train, test = data.iloc[:split], data.iloc[split:]

        x_train = self.scaler.fit_transform(train[features])
        x_test = self.scaler.transform(test[features])

        self.model.fit(x_train, train["label"])
        preds = self.model.predict(x_test)
        proba = self.model.predict_proba(x_test)[:, 1]

        p_oos = precision_score(test["label"], preds, zero_division=0)
        trade_rs = self._trade_r_returns(proba, test["label"].to_numpy(), threshold=0.75)
        std_oos = np.std(trade_rs)
        sharpe_oos = 0.0 if std_oos < 1e-6 else float(np.mean(trade_rs) / std_oos * np.sqrt(252))

        fold_precisions: List[float] = []
        fold_sharpes: List[float] = []
        window = max(180, int(len(data) * 0.25))
        step = max(60, int(window * 0.35))

        for start in range(0, len(data) - window - step + 1, step):
            tr = data.iloc[start : start + window]
            te = data.iloc[start + window : start + window + step]
            if len(tr) < 100 or te.empty:
                continue
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(tr[features])
            x_te = scaler.transform(te[features])
            model = GradientBoostingClassifier(
                random_state=42,
                learning_rate=self.model.learning_rate,
                n_estimators=self.model.n_estimators,
                max_depth=self.model.max_depth,
            )
            model.fit(x_tr, tr["label"])
            pred = model.predict(x_te)
            pro = model.predict_proba(x_te)[:, 1]
            fold_precisions.append(precision_score(te["label"], pred, zero_division=0))
            r_fold = self._trade_r_returns(pro, te["label"].to_numpy(), threshold=0.75)
            std_fold = np.std(r_fold)
            fold_sharpes.append(0.0 if std_fold < 1e-6 else float(np.mean(r_fold) / std_fold * np.sqrt(252)))

        self.precision_oos = float(p_oos)
        self.walk_forward_precision = float(np.mean(fold_precisions)) if fold_precisions else 0.0
        self.walk_forward_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0

        stable_precision = len(fold_precisions) >= 2 and self.walk_forward_precision >= 0.55 and np.std(fold_precisions) <= 0.25
        stable_sharpe = len(fold_sharpes) >= 2 and self.walk_forward_sharpe > 0
        self.trained = bool(self.precision_oos >= 0.55 and sharpe_oos > 0 and stable_precision and stable_sharpe)
        return self.precision_oos

    def probability(self, x: pd.DataFrame) -> float:
        if not self.trained:
            return 0.0
        x_scaled = self.scaler.transform(x)
        return float(self.model.predict_proba(x_scaled)[0][1])


class RiskManager:
    def __init__(
        self,
        account_size: float = 100000,
        risk_per_trade: float = 0.01,
        max_concurrent: int = 3,
        max_portfolio_risk: float = 0.03,
    ):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.max_concurrent = max_concurrent
        self.max_portfolio_risk = max_portfolio_risk

    def allowed(self, active_signals: List[Signal]) -> bool:
        if len(active_signals) >= self.max_concurrent:
            return False
        return (len(active_signals) * self.risk_per_trade) < self.max_portfolio_risk

    def position_size(self, entry: float, stop: float, atr: float) -> int:
        risk_cash = self.account_size * self.risk_per_trade
        stop_distance = max(entry - stop, 1e-6)
        volatility_penalty = 1.0 / max(1.0, (atr / max(entry, 1e-9)) * 120)
        return max(int((risk_cash / stop_distance) * volatility_penalty), 0)


class Backtester:
    def run(
        self,
        signals: List[Signal],
        historical: Dict[str, pd.DataFrame],
        horizon: int = 20,
        slippage_bps: float = 5,
        fee_bps: float = 1,
    ) -> Dict[str, float]:
        if not signals:
            return {"win_rate": 0.0, "avg_r": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

        rs: List[float] = []
        for signal in signals:
            if signal.ticker not in historical:
                continue
            fut = historical[signal.ticker][historical[signal.ticker]["timestamp"] > signal.timestamp].head(horizon)
            if fut.empty:
                continue

            entry = signal.entry * (1 + slippage_bps / 10000)
            stop = signal.stop
            target = signal.target
            fee = entry * fee_bps / 10000
            r = 0.0

            for px in fut["close"]:
                if px >= target:
                    r = (target - entry - fee) / max(entry - stop, 1e-9)
                    break
                if px <= stop:
                    r = -1.0
                    break
            rs.append(r)

        if not rs:
            return {"win_rate": 0.0, "avg_r": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

        equity = np.cumsum(rs)
        dd = np.max(np.maximum.accumulate(equity) - equity)
        return {
            "win_rate": float(np.mean(np.array(rs) > 0)),
            "avg_r": float(np.mean(rs)),
            "max_drawdown": float(dd),
            "sharpe": 0.0 if np.std(rs) < 1e-6 else float(np.mean(rs) / np.std(rs) * np.sqrt(252)),
        }


class RealTimeBreakoutEngine:
    def __init__(
        self,
        timeframe: str = "5m",
        min_volume_ratio: float = 1.5,
        model_learning_rate: float = 0.05,
        model_estimators: int = 250,
        model_depth: int = 2,
    ):
        self.timeframe = timeframe
        self.buffer = DataBuffer(maxlen=700)
        self.detector = PatternDetector(min_volume_ratio=min_volume_ratio)
        self.regime = MarketRegimeFilter()
        self.features = FeatureBuilder()
        self.labels = LabelBuilder()
        self.model = ModelManager(
            learning_rate=model_learning_rate,
            n_estimators=model_estimators,
            max_depth=model_depth,
        )
        self.risk = RiskManager()
        self.active_signals: List[Signal] = []
        self.watchlist: List[Dict[str, str]] = []

    def fit(self, historical_by_ticker: Dict[str, pd.DataFrame]) -> float:
        rows: List[pd.DataFrame] = []
        for ticker, df in historical_by_ticker.items():
            enriched = IndicatorEngine.add_indicators(df)
            enriched["label"] = self.labels.build(enriched)
            enriched["ticker"] = ticker
            rows.append(enriched)
        merged = pd.concat(rows, ignore_index=True)
        return self.model.train_with_walk_forward(merged, self.features.FEATURES)

    def on_bar(
        self,
        bar: Bar,
        index_df: Optional[pd.DataFrame] = None,
        earnings_calendar: Optional[Dict[str, List[pd.Timestamp]]] = None,
    ) -> Optional[Signal]:
        df = self.buffer.update(bar)
        df = IndicatorEngine.add_indicators(df)
        self.buffer.frames[bar.ticker] = df

        if len(df) < 90:
            return None

        detection = self.detector.detect(df)
        if not detection:
            self.watchlist.append({"ticker": bar.ticker, "timeframe": self.timeframe, "state": "forming", "timestamp": str(bar.timestamp)})
            return None

        pattern_name, resistance, invalidation = detection
        entry = float(df["close"].iloc[-1])

        if entry <= resistance:
            return None

        if entry <= invalidation:
            return None

        if earnings_calendar and bar.ticker in earnings_calendar:
            upcoming = [d for d in earnings_calendar[bar.ticker] if d >= bar.timestamp]
            if upcoming and (upcoming[0] - bar.timestamp).days <= 2:
                return None

        if not self.regime.favorable(df, index_df=index_df):
            return None

        risk_per_share = entry - invalidation
        target = entry + 2 * risk_per_share
        rr = (target - entry) / max(risk_per_share, 1e-9)
        if rr < 2.0:
            return None

        if not self.risk.allowed(self.active_signals):
            return None

        probability = self.model.probability(self.features.latest(df))
        if probability < 0.75:
            return None

        qty = self.risk.position_size(entry=entry, stop=invalidation, atr=float(df["atr"].iloc[-1]))
        if qty <= 0:
            return None

        signal = Signal(
            ticker=bar.ticker,
            timeframe=self.timeframe,
            pattern=pattern_name,
            entry=entry,
            stop=float(invalidation),
            target=float(target),
            rr=float(rr),
            quantity=qty,
            ml_probability=float(probability),
            timestamp=bar.timestamp,
        )
        self.active_signals.append(signal)
        return signal

    def invalidate_broken_structures(self) -> None:
        kept: List[Signal] = []
        for signal in self.active_signals:
            df = self.buffer.frames.get(signal.ticker)
            if df is None or df.empty:
                continue
            if float(df["close"].iloc[-1]) <= signal.stop:
                continue
            improved = float(df["low"].iloc[-1] - df["atr"].iloc[-1] * 0.5)
            signal.stop = max(signal.stop, improved)
            kept.append(signal)
        self.active_signals = kept

    def active_setups_frame(self) -> pd.DataFrame:
        if not self.active_signals:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "timeframe",
                    "pattern",
                    "entry",
                    "stop",
                    "target",
                    "risk_reward",
                    "quantity",
                    "ml_probability",
                    "timestamp",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "ticker": s.ticker,
                    "timeframe": s.timeframe,
                    "pattern": s.pattern,
                    "entry": s.entry,
                    "stop": s.stop,
                    "target": s.target,
                    "risk_reward": s.rr,
                    "quantity": s.quantity,
                    "ml_probability": s.ml_probability,
                    "timestamp": s.timestamp,
                }
                for s in self.active_signals
            ]
        )

    def watchlist_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.watchlist[-200:])

    def display_state(self) -> pd.DataFrame:
        return self.active_setups_frame()


def run_realtime_loop(
    source: StreamingOHLCVSource,
    engine: RealTimeBreakoutEngine,
    index_df: Optional[pd.DataFrame] = None,
    earnings_calendar: Optional[Dict[str, List[pd.Timestamp]]] = None,
) -> None:
    for bar in source.stream():
        signal = engine.on_bar(bar, index_df=index_df, earnings_calendar=earnings_calendar)
        engine.invalidate_broken_structures()
        if signal:
            print(
                f"[HIGH-CONFIDENCE] {signal.ticker} tf={signal.timeframe} pattern={signal.pattern} "
                f"entry={signal.entry:.2f} stop={signal.stop:.2f} target={signal.target:.2f} rr={signal.rr:.2f} "
                f"qty={signal.quantity} ml={signal.ml_probability:.2%} t={signal.timestamp}"
            )
        if engine.watchlist:
            w = engine.watchlist[-1]
            print(f"[WATCHLIST] {w['ticker']} tf={w['timeframe']} state={w['state']} at {w['timestamp']}")
