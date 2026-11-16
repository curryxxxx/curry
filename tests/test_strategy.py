import unittest

import numpy as np
import pandas as pd

from indicators import add_indicators
from strategy import EmaRsiAtrStrategy, TradingStrategy


class StrategyTests(unittest.TestCase):
    def test_indicators_add_expected_columns(self):
        close = np.linspace(1.0, 1.2, 80)
        bars = pd.DataFrame(
            {"open": close, "high": close + 0.01, "low": close - 0.01, "close": close}
        )
        enriched = add_indicators(bars)
        self.assertTrue({"ema_fast", "ema_slow", "rsi", "atr"}.issubset(enriched.columns))
        self.assertTrue({"ema_50", "ema_200", "macd", "macd_signal"}.issubset(enriched.columns))
        self.assertTrue(np.isfinite(enriched["atr"].iloc[-1]))

    def test_signal_stop_and_target_are_directional(self):
        bars = pd.DataFrame(
            {"open": [1.0] * 80, "high": [1.01] * 80, "low": [0.99] * 80, "close": [1.0] * 80},
            index=pd.date_range("2025-01-01", periods=80, freq="15min", tz="UTC"),
        )
        strategy = EmaRsiAtrStrategy()
        signal = strategy.evaluate("EURUSD", bars)
        self.assertIsNone(signal)

    def test_all_ten_strategies_accept_valid_bars(self):
        rng = np.random.default_rng(42)
        close = 1.1 + np.cumsum(rng.normal(0, 0.001, 300))
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.001,
                "low": close - 0.001,
                "close": close,
            },
            index=pd.date_range("2025-01-01", periods=len(close), freq="15min", tz="UTC"),
        )
        names = (
            "ma_crossover",
            "support_resistance",
            "breakout_retest",
            "rsi",
            "fibonacci",
            "liquidity_sweep",
            "fair_value_gap",
            "london_breakout",
            "macd",
            "supply_demand",
        )
        for name in names:
            with self.subTest(strategy=name):
                result = TradingStrategy(name).evaluate("EURUSD", bars)
                self.assertTrue(result is None or result.side in ("buy", "sell"))

    def test_generated_signal_uses_at_least_two_to_one_reward_risk(self):
        strategy = TradingStrategy("rsi")
        bars = pd.DataFrame(
            {
                "open": [1.0] * 80,
                "high": [1.01] * 80,
                "low": [0.99] * 80,
                "close": [1.0] * 80,
            },
            index=pd.date_range("2025-01-01", periods=80, freq="15min", tz="UTC"),
        )
        data = add_indicators(bars)
        signal = strategy._make_signal("EURUSD", data, "buy")
        risk = signal.reference_price - signal.stop_loss
        reward = signal.take_profit - signal.reference_price
        self.assertAlmostEqual(reward / risk, 2.0)

    def test_missing_ohlc_columns_raise(self):
        with self.assertRaises(ValueError):
            add_indicators(pd.DataFrame({"close": [1.0] * 60}))


if __name__ == "__main__":
    unittest.main()
