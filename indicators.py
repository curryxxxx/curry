"""Technical indicators implemented with pandas and numpy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"OHLC data is missing columns: {', '.join(sorted(missing))}")
    if len(bars) < 50:
        raise ValueError("At least 50 bars are required to calculate indicators.")

    result = bars.copy()
    result["ema_fast"] = result["close"].ewm(span=20, adjust=False).mean()
    result["ema_slow"] = result["close"].ewm(span=50, adjust=False).mean()
    result["ema_50"] = result["close"].ewm(span=50, adjust=False).mean()
    result["ema_200"] = result["close"].ewm(span=200, adjust=False).mean()

    delta = result["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi"] = (100 - (100 / (1 + relative_strength))).fillna(50)
    macd = result["close"].ewm(span=12, adjust=False).mean() - result["close"].ewm(
        span=26, adjust=False
    ).mean()
    result["macd"] = macd
    result["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    return result
