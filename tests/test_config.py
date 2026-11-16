import os
import unittest
from unittest.mock import patch

from config import Settings


class SettingsTests(unittest.TestCase):
    def test_paper_mode_is_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings.from_env().mode, "paper")

    def test_live_mode_requires_separate_confirmation(self):
        with patch.dict(os.environ, {"CURRY_TRADING_MODE": "live"}, clear=True):
            with self.assertRaisesRegex(ValueError, "CURRY_LIVE_TRADING_CONFIRM"):
                Settings.from_env()

    def test_live_mode_can_only_be_explicitly_confirmed(self):
        env = {
            "CURRY_TRADING_MODE": "live",
            "CURRY_LIVE_TRADING_CONFIRM": "I_ACCEPT_LIVE_RISK",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(Settings.from_env().mode, "live")

    def test_strategy_and_risk_bounds_are_validated(self):
        with self.assertRaisesRegex(ValueError, "Unsupported strategy"):
            Settings(strategy="unknown")
        with self.assertRaisesRegex(ValueError, "between 0.5% and 1%"):
            Settings(risk_per_trade_pct=0.25)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            Settings(reward_to_risk=1.5)


if __name__ == "__main__":
    unittest.main()
