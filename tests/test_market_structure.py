import unittest

import pandas as pd

from analysis.structure import analyze_structure, find_swings
from analysis.zones import build_supply_demand_zones, build_support_resistance_zones


def bars_from(high, low=None, close=None):
    low = low if low is not None else [value - 1 for value in high]
    close = close if close is not None else [(top + bottom) / 2 for top, bottom in zip(high, low)]
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
        }
    )


class SwingStructureTests(unittest.TestCase):
    def test_fractal_is_not_visible_until_n_bars_after_pivot(self):
        bars = bars_from([1, 2, 5, 3, 2, 1, 2], [0, 1, 2, 1, 0, -1, 0])
        before_confirmation = find_swings(bars.iloc[:4], n=2)
        self.assertFalse(any(swing.index == 2 and swing.kind == "HIGH" for swing in before_confirmation))

        confirmed = find_swings(bars, n=2)
        pivot = next(swing for swing in confirmed if swing.index == 2 and swing.kind == "HIGH")
        self.assertEqual(pivot.known_at, 4)

    def test_future_bars_cannot_change_an_already_confirmed_swing(self):
        bars = bars_from(
            [1, 2, 6, 3, 2, 1, 2, 1, 2, 1],
            [0, 1, 2, 1, 0, -1, 0, -1, 0, -1],
        )
        prefix = bars.iloc[:5]
        original = next(
            swing
            for swing in find_swings(prefix, n=2)
            if swing.index == 2 and swing.kind == "HIGH"
        )
        changed_future = bars.copy()
        changed_future.loc[5:, "high"] = [100, 200, 300, 400, 500]
        changed_future.loc[5:, "low"] = [-100, -200, -300, -400, -500]
        after_extension = next(
            swing
            for swing in find_swings(changed_future, n=2)
            if swing.index == 2 and swing.kind == "HIGH"
        )
        self.assertEqual(original, after_extension)

    def test_trend_history_only_updates_after_swing_confirmation(self):
        bars = bars_from(
            [2, 3, 2, 4, 2, 5, 2, 6, 4],
            [0, 1, 0, 2, 1, 3, 2, 4, 3],
        )
        result = analyze_structure(bars, n=1)
        for index, point in enumerate(result.trend_history):
            self.assertEqual(point.known_at, index)
        classifications = {swing.classification for swing in result.swings}
        self.assertIn("HH", classifications)
        self.assertIn("HL", classifications)
        self.assertEqual(result.trend_history[2].state, "RANGE")
        self.assertEqual(result.trend, "BULLISH")
        self.assertEqual(result.known_at, len(bars) - 1)

    def test_future_bars_cannot_change_past_trend_history(self):
        bars = bars_from(
            [2, 3, 2, 4, 2, 5, 2, 6, 4],
            [0, 1, 0, 2, 1, 3, 2, 4, 3],
        )
        cutoff = 6
        prefix_result = analyze_structure(bars.iloc[: cutoff + 1], n=1)
        extended = bars.copy()
        extended.loc[cutoff + 1 :, "high"] = [100, 200]
        extended.loc[cutoff + 1 :, "low"] = [-100, -200]
        extended_result = analyze_structure(extended, n=1)
        self.assertEqual(
            prefix_result.trend_history,
            extended_result.trend_history[: cutoff + 1],
        )


class ZoneTests(unittest.TestCase):
    def test_support_resistance_zone_known_at_covers_all_cluster_touches(self):
        highs = [10, 12, 10, 11, 10, 12, 10, 11, 10, 12, 10]
        lows = [8, 9, 7, 9, 8, 9, 7, 9, 8, 9, 8]
        bars = bars_from(highs, lows)
        zones = build_support_resistance_zones(bars, n=1, tolerance_atr=0.5)
        self.assertTrue(zones)
        swings = find_swings(bars, n=1)
        for zone in zones:
            self.assertGreaterEqual(zone.known_at, 0)
            self.assertEqual(zone.touch_count, len(zone.swing_indices))
            matching_swings = [
                swing for swing in swings if swing.index in zone.swing_indices
            ]
            self.assertTrue(matching_swings)
            self.assertGreaterEqual(
                zone.known_at, max(swing.known_at for swing in matching_swings)
            )
            self.assertGreaterEqual(zone.strength_score, 0)
            self.assertLessEqual(zone.strength_score, 100)

    def test_supply_zone_status_uses_only_bars_after_its_creation(self):
        bars = pd.DataFrame(
            {
                "open": [10.0, 10.0, 10.0, 10.05, 10.0, 9.95],
                "high": [10.05, 10.05, 10.05, 10.4, 10.1, 10.0],
                "low": [9.95, 9.95, 9.95, 10.03, 9.94, 9.8],
                "close": [10.0, 10.0, 10.0, 10.35, 10.0, 9.9],
            }
        )
        created_only = build_supply_demand_zones(bars.iloc[:4])
        demand = next(zone for zone in created_only if zone.displacement_index == 3)
        self.assertEqual(demand.kind, "DEMAND")
        self.assertEqual(demand.status, "FRESH")
        self.assertEqual(demand.known_at, 3)
        self.assertEqual(demand.status_known_at, 3)

        retested = build_supply_demand_zones(bars.iloc[:5])
        tested_zone = next(zone for zone in retested if zone.displacement_index == 3)
        self.assertEqual(tested_zone.status, "TESTED")
        self.assertEqual(tested_zone.status_known_at, 4)

        invalidated = build_supply_demand_zones(bars)
        invalid_zone = next(zone for zone in invalidated if zone.displacement_index == 3)
        self.assertEqual(invalid_zone.status, "INVALIDATED")
        self.assertEqual(invalid_zone.status_known_at, 5)
        self.assertEqual(invalid_zone.known_at, 3)

    def test_zone_does_not_exist_before_displacement_close(self):
        bars = pd.DataFrame(
            {
                "open": [10.0, 10.0, 10.0, 10.05],
                "high": [10.05, 10.05, 10.05, 10.4],
                "low": [9.95, 9.95, 9.95, 10.03],
                "close": [10.0, 10.0, 10.0, 10.35],
            }
        )
        self.assertEqual(build_supply_demand_zones(bars.iloc[:3]), [])
        zone = build_supply_demand_zones(bars)[0]
        self.assertEqual(zone.known_at, zone.displacement_index)


if __name__ == "__main__":
    unittest.main()
