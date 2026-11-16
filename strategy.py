"""Selectable closed-bar Forex strategies with common ATR-based exits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from config import STRATEGIES
from indicators import add_indicators

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Side
    reference_price: float
    stop_loss: float
    take_profit: float
    timestamp: pd.Timestamp


class TradingStrategy:
    """Evaluate one of the ten configured strategies against closed OHLC bars."""

    def __init__(
        self,
        name: str = "ma_crossover",
        atr_stop_multiplier: float = 1.5,
        reward_to_risk: float = 2.0,
    ):
        if name not in STRATEGIES:
            raise ValueError(f"Unknown strategy {name!r}; choose from {', '.join(STRATEGIES)}.")
        if atr_stop_multiplier <= 0 or reward_to_risk < 2:
            raise ValueError("ATR stop multiplier must be positive and reward-to-risk at least 2.")
        self.name = name
        self.atr_stop_multiplier = atr_stop_multiplier
        self.reward_to_risk = reward_to_risk

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        data = add_indicators(bars)
        if len(data) < 2:
            return None
        current = data.iloc[-1]
        if pd.isna(current["atr"]) or current["atr"] <= 0:
            return None
        methods = {
            "ma_crossover": self._ma_crossover,
            "support_resistance": self._support_resistance,
            "breakout_retest": self._breakout_retest,
            "rsi": self._rsi_reversal,
            "fibonacci": self._fibonacci_retracement,
            "liquidity_sweep": self._liquidity_sweep,
            "fair_value_gap": self._fair_value_gap,
            "london_breakout": self._london_breakout,
            "macd": self._macd_crossover,
            "supply_demand": self._supply_demand,
        }
        side = methods[self.name](data)
        if side is None:
            return None
        return self._make_signal(symbol, data, side)

    def _make_signal(self, symbol: str, data: pd.DataFrame, side: Side) -> Signal:
        current = data.iloc[-1]
        price = float(current["close"])
        stop_distance = float(current["atr"]) * self.atr_stop_multiplier
        if side == "buy":
            stop_loss = price - stop_distance
            take_profit = price + stop_distance * self.reward_to_risk
        else:
            stop_loss = price + stop_distance
            take_profit = price - stop_distance * self.reward_to_risk
        timestamp = current.name
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
        return Signal(symbol, side, price, stop_loss, take_profit, timestamp)

    @staticmethod
    def _bullish(current: pd.Series) -> bool:
        return current["close"] > current["open"]

    @staticmethod
    def _bearish(current: pd.Series) -> bool:
        return current["close"] < current["open"]

    def _ma_crossover(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 201:
            return None
        previous, current = data.iloc[-2], data.iloc[-1]
        if previous["ema_50"] <= previous["ema_200"] and current["ema_50"] > current["ema_200"]:
            return "buy"
        if previous["ema_50"] >= previous["ema_200"] and current["ema_50"] < current["ema_200"]:
            return "sell"
        return None

    def _support_resistance(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 21:
            return None
        current = data.iloc[-1]
        prior = data.iloc[-21:-1]
        tolerance = float(current["atr"]) * 0.2
        support = float(prior["low"].min())
        resistance = float(prior["high"].max())
        if current["low"] <= support + tolerance and current["close"] > support and self._bullish(current):
            return "buy"
        if current["high"] >= resistance - tolerance and current["close"] < resistance and self._bearish(current):
            return "sell"
        return None

    def _breakout_retest(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 27:
            return None
        current = data.iloc[-1]
        tolerance = float(current["atr"]) * 0.2
        first_candidate = max(20, len(data) - 7)
        for index in range(len(data) - 2, first_candidate - 1, -1):
            breakout = data.iloc[index]
            prior = data.iloc[index - 20 : index]
            resistance = float(prior["high"].max())
            support = float(prior["low"].min())
            if breakout["close"] > resistance and current["low"] <= resistance + tolerance:
                if current["close"] > resistance and self._bullish(current):
                    return "buy"
            if breakout["close"] < support and current["high"] >= support - tolerance:
                if current["close"] < support and self._bearish(current):
                    return "sell"
        return None

    def _rsi_reversal(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 2:
            return None
        previous, current = data.iloc[-2], data.iloc[-1]
        if previous["rsi"] < 30 <= current["rsi"] and self._bullish(current):
            return "buy"
        if previous["rsi"] > 70 >= current["rsi"] and self._bearish(current):
            return "sell"
        return None

    def _fibonacci_retracement(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 52:
            return None
        swing = data.iloc[-51:-1]
        low_index = int(np.argmin(swing["low"].to_numpy()))
        high_index = int(np.argmax(swing["high"].to_numpy()))
        swing_low = float(swing.iloc[low_index]["low"])
        swing_high = float(swing.iloc[high_index]["high"])
        size = swing_high - swing_low
        current = data.iloc[-1]
        if size <= 2 * float(current["atr"]):
            return None

        levels = (0.382, 0.5, 0.618)
        tolerance = float(current["atr"]) * 0.15
        if low_index < high_index:
            retracement_prices = [swing_high - size * level for level in levels]
            in_zone = any(
                current["low"] <= level + tolerance and current["low"] >= level - tolerance
                for level in retracement_prices
            )
            if in_zone and self._bullish(current) and current["close"] > min(retracement_prices):
                return "buy"
        elif high_index < low_index:
            retracement_prices = [swing_low + size * level for level in levels]
            in_zone = any(
                current["high"] >= level - tolerance and current["high"] <= level + tolerance
                for level in retracement_prices
            )
            if in_zone and self._bearish(current) and current["close"] < max(retracement_prices):
                return "sell"
        return None

    def _liquidity_sweep(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 22:
            return None
        current = data.iloc[-1]
        prior = data.iloc[-21:-1]
        tolerance = float(current["atr"]) * 0.1
        highs = prior["high"].to_numpy(dtype=float)
        lows = prior["low"].to_numpy(dtype=float)
        equal_high = next(
            (level for level in highs if np.count_nonzero(np.abs(highs - level) <= tolerance) >= 2),
            None,
        )
        equal_low = next(
            (level for level in lows if np.count_nonzero(np.abs(lows - level) <= tolerance) >= 2),
            None,
        )
        if equal_low is not None and current["low"] < equal_low - tolerance and current["close"] > equal_low and self._bullish(current):
            return "buy"
        if equal_high is not None and current["high"] > equal_high + tolerance and current["close"] < equal_high and self._bearish(current):
            return "sell"
        return None

    def _fair_value_gap(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 4:
            return None
        older, impulse, current = data.iloc[-4], data.iloc[-2], data.iloc[-1]
        if older["high"] < impulse["low"]:
            gap_low, gap_high = float(older["high"]), float(impulse["low"])
            if current["low"] <= gap_high and current["low"] >= gap_low and current["close"] > gap_low and self._bullish(current):
                return "buy"
        if older["low"] > impulse["high"]:
            gap_low, gap_high = float(impulse["high"]), float(older["low"])
            if current["high"] >= gap_low and current["high"] <= gap_high and current["close"] < gap_high and self._bearish(current):
                return "sell"
        return None

    def _london_breakout(self, data: pd.DataFrame) -> Side | None:
        if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
            return None
        current = data.iloc[-1]
        timestamp = data.index[-1].tz_convert("UTC")
        if not 7 <= timestamp.hour < 11:
            return None
        same_day = data.loc[data.index.tz_convert("UTC").date == timestamp.date()]
        asia = same_day.loc[same_day.index.tz_convert("UTC").hour < 7]
        if len(asia) < 2:
            return None
        asia_high = float(asia["high"].max())
        asia_low = float(asia["low"].min())
        if current["close"] > asia_high and self._bullish(current):
            return "buy"
        if current["close"] < asia_low and self._bearish(current):
            return "sell"
        return None

    def _macd_crossover(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 200:
            return None
        previous, current = data.iloc[-2], data.iloc[-1]
        bullish = previous["macd"] <= previous["macd_signal"] and current["macd"] > current["macd_signal"]
        bearish = previous["macd"] >= previous["macd_signal"] and current["macd"] < current["macd_signal"]
        if bullish and current["ema_50"] > current["ema_200"]:
            return "buy"
        if bearish and current["ema_50"] < current["ema_200"]:
            return "sell"
        return None

    def _supply_demand(self, data: pd.DataFrame) -> Side | None:
        if len(data) < 30:
            return None
        current = data.iloc[-1]
        tolerance = float(current["atr"]) * 0.2
        first_base = max(5, len(data) - 15)
        for index in range(len(data) - 6, first_base - 1, -1):
            base = data.iloc[index]
            atr = float(base["atr"])
            if not np.isfinite(atr) or atr <= 0 or base["high"] - base["low"] > atr * 0.8:
                continue
            departure = data.iloc[index + 1 : min(index + 5, len(data) - 1)]
            if departure.empty:
                continue
            zone_low, zone_high = float(base["low"]), float(base["high"])
            moved_up = float(departure["close"].max()) >= zone_high + 1.2 * atr
            moved_down = float(departure["close"].min()) <= zone_low - 1.2 * atr
            if moved_up and current["low"] <= zone_high + tolerance and current["close"] > zone_high and self._bullish(current):
                return "buy"
            if moved_down and current["high"] >= zone_low - tolerance and current["close"] < zone_low and self._bearish(current):
                return "sell"
        return None


class EmaRsiAtrStrategy:
    """Compatibility wrapper for the original 20/50 EMA + RSI setup."""

    def __init__(self, atr_stop_multiplier: float = 1.5, reward_to_risk: float = 2.0):
        if atr_stop_multiplier <= 0 or reward_to_risk <= 0:
            raise ValueError("Stop multiplier and reward-to-risk must be positive.")
        self.atr_stop_multiplier = atr_stop_multiplier
        self.reward_to_risk = reward_to_risk

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        data = add_indicators(bars)
        previous, current = data.iloc[-2], data.iloc[-1]
        side: Side | None = None
        if previous["ema_fast"] <= previous["ema_slow"] and current["ema_fast"] > current["ema_slow"] and 50 <= current["rsi"] < 70:
            side = "buy"
        elif previous["ema_fast"] >= previous["ema_slow"] and current["ema_fast"] < current["ema_slow"] and 30 < current["rsi"] <= 50:
            side = "sell"
        if side is None:
            return None
        return TradingStrategy(
            "rsi", self.atr_stop_multiplier, max(2.0, self.reward_to_risk)
        )._make_signal(symbol, data, side)
