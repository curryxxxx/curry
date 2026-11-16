import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from config import STRATEGIES
from filters.sessions import (
    current_sessions,
    is_active,
    is_overlap,
    load_strategy_sessions,
)


class SessionWindowTests(unittest.TestCase):
    def test_session_boundaries_are_start_inclusive_and_end_exclusive(self):
        start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
        self.assertTrue(is_active("Asian", start))
        self.assertFalse(is_active("asian", end))

    def test_new_york_dst_start_changes_utc_open_time(self):
        before_dst = datetime(2026, 3, 6, 12, 30, tzinfo=timezone.utc)
        after_dst = datetime(2026, 3, 9, 12, 30, tzinfo=timezone.utc)
        self.assertFalse(is_active("new_york", before_dst))
        self.assertTrue(is_active("new_york", after_dst))

    def test_london_dst_start_changes_utc_open_time(self):
        before_dst = datetime(2026, 3, 27, 7, 30, tzinfo=timezone.utc)
        after_dst = datetime(2026, 3, 30, 7, 30, tzinfo=timezone.utc)
        self.assertFalse(is_active("london", before_dst))
        self.assertTrue(is_active("london", after_dst))

    def test_new_york_dst_end_changes_utc_open_time(self):
        before_dst = datetime(2026, 10, 30, 12, 30, tzinfo=timezone.utc)
        after_dst = datetime(2026, 11, 2, 12, 30, tzinfo=timezone.utc)
        self.assertTrue(is_active("new_york", before_dst))
        self.assertFalse(is_active("new_york", after_dst))

    def test_london_dst_end_changes_utc_open_time(self):
        before_dst = datetime(2026, 10, 23, 7, 30, tzinfo=timezone.utc)
        after_dst = datetime(2026, 10, 26, 7, 30, tzinfo=timezone.utc)
        self.assertTrue(is_active("london", before_dst))
        self.assertFalse(is_active("london", after_dst))

    def test_current_sessions_and_london_new_york_overlap(self):
        during_overlap = datetime(2026, 3, 9, 12, 30, tzinfo=timezone.utc)
        outside_overlap = datetime(2026, 3, 6, 12, 30, tzinfo=timezone.utc)
        self.assertEqual(current_sessions(during_overlap), frozenset({"london", "new_york"}))
        self.assertTrue(is_overlap(during_overlap))
        self.assertFalse(is_overlap(outside_overlap))

    def test_weekend_session_clock_is_not_active(self):
        tokyo_saturday_morning = datetime(2026, 1, 10, 0, 30, tzinfo=timezone.utc)
        self.assertFalse(is_active("asian", tokyo_saturday_morning))

    def test_naive_datetime_and_unknown_session_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            is_active("london", datetime(2026, 1, 5, 12))
        with self.assertRaisesRegex(ValueError, "Unknown session"):
            is_active("sydney", datetime(2026, 1, 5, tzinfo=timezone.utc))


class StrategySessionConfigTests(unittest.TestCase):
    def test_default_yaml_covers_every_strategy_and_restricts_london_breakout(self):
        configured = load_strategy_sessions()
        self.assertEqual(set(configured), set(STRATEGIES))
        self.assertEqual(configured["london_breakout"], frozenset({"london"}))
        self.assertIn("asian", configured["rsi"])

    def test_invalid_yaml_session_is_rejected(self):
        invalid_config = (
            "strategies:\n"
            + "".join(f"  {strategy}: [london]\n" for strategy in STRATEGIES[:-1])
            + f"  {STRATEGIES[-1]}: [sydney]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.yaml"
            path.write_text(invalid_config, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown session"):
                load_strategy_sessions(path)


if __name__ == "__main__":
    unittest.main()
