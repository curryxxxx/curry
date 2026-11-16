import unittest

import pandas as pd

from analysis.smc import (
    detect_equal_highs_lows,
    detect_fair_value_gaps,
    detect_liquidity_sweeps,
    detect_order_blocks,
    detect_structure_breaks,
)


def frame(open_, high, low, close):
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}
    )


def bullish_structure_bars():
    return frame(
        [10.0, 11.7, 10.0, 10.8, 12.2, 12.0, 11.3, 12.4, 11.0],
        [10.2, 12.0, 10.3, 11.3, 13.0, 12.2, 11.5, 14.0, 11.2],
        [9.8, 11.5, 9.5, 10.5, 11.7, 10.5, 10.8, 12.2, 10.4],
        [10.0, 11.8, 9.8, 11.0, 12.5, 11.0, 11.0, 13.5, 11.0],
    )


def fvg_bars(future=False):
    open_ = [9.8, 9.8, 10.7]
    high = [10.0, 10.8, 11.2]
    low = [9.5, 9.7, 10.5]
    close = [9.8, 10.5, 11.0]
    if future:
        open_ += [10.8, 10.2]
        high += [11.0, 10.3]
        low += [10.25, 9.9]
        close += [10.6, 10.1]
    return frame(open_, high, low, close)


class StructureBreakTests(unittest.TestCase):
    def test_bos_requires_a_close_beyond_confirmed_swing(self):
        events = detect_structure_breaks(bullish_structure_bars(), n=1)
        bos = [event for event in events if event.kind == "BOS"]
        self.assertTrue(bos)
        self.assertEqual(bos[0].direction, "BULLISH")
        self.assertEqual(bos[0].level, 13.0)
        self.assertEqual(bos[0].known_at, 7)
        self.assertEqual(bos[0].swing_index, 4)

    def test_wick_only_break_does_not_create_bos(self):
        bars = bullish_structure_bars()
        bars.loc[7, "close"] = 12.9
        bars.loc[7, "open"] = 12.5
        self.assertFalse(
            any(event.kind == "BOS" for event in detect_structure_breaks(bars, n=1))
        )

    def test_choch_requires_a_close_beyond_the_countertrend_swing(self):
        bars = bullish_structure_bars()
        bars.loc[7, ["open", "high", "low", "close"]] = [11.2, 11.4, 9.8, 10.2]
        events = detect_structure_breaks(bars, n=1)
        choch = [event for event in events if event.kind == "CHoCH"]
        self.assertTrue(choch)
        self.assertEqual(choch[0].direction, "BEARISH")
        self.assertEqual(choch[0].level, 10.5)
        self.assertEqual(choch[0].known_at, 7)

    def test_countertrend_wick_without_close_does_not_create_choch(self):
        bars = bullish_structure_bars()
        bars.loc[7, ["open", "high", "low", "close"]] = [11.2, 11.4, 9.8, 10.6]
        self.assertFalse(
            any(event.kind == "CHoCH" for event in detect_structure_breaks(bars, n=1))
        )


class FvgTests(unittest.TestCase):
    def test_bullish_fvg_progresses_open_partial_filled(self):
        open_gap = detect_fair_value_gaps(fvg_bars())
        gap = next(event for event in open_gap if event.formed_at == 2)
        self.assertEqual((gap.lower, gap.upper), (10.0, 10.5))
        self.assertEqual(gap.status, "OPEN")
        self.assertEqual(gap.known_at, 2)

        partial = detect_fair_value_gaps(fvg_bars(future=True)[:4])
        partial_gap = max(
            (event for event in partial if event.formed_at == 2),
            key=lambda event: event.known_at,
        )
        self.assertEqual(partial_gap.status, "PARTIAL")
        self.assertEqual(partial_gap.known_at, 3)

        filled = detect_fair_value_gaps(fvg_bars(future=True))
        filled_gap = max(
            (event for event in filled if event.formed_at == 2),
            key=lambda event: event.known_at,
        )
        self.assertEqual(filled_gap.status, "FILLED")
        self.assertEqual(filled_gap.known_at, 4)
        self.assertEqual(
            [event.status for event in filled if event.formed_at == 2],
            ["OPEN", "PARTIAL", "FILLED"],
        )

    def test_bearish_fvg_fills_at_its_opposite_edge(self):
        bars = frame(
            [10.7, 10.5, 9.8, 10.2],
            [11.0, 10.8, 10.0, 10.6],
            [10.5, 9.7, 9.5, 10.1],
            [10.7, 10.0, 9.8, 10.2],
        )
        gap = max(
            (event for event in detect_fair_value_gaps(bars) if event.formed_at == 2),
            key=lambda event: event.known_at,
        )
        self.assertEqual(gap.direction, "BEARISH")
        self.assertEqual((gap.lower, gap.upper), (10.0, 10.5))
        self.assertEqual(gap.status, "FILLED")
        self.assertEqual(gap.known_at, 3)

    def test_small_or_nonexistent_gap_is_ignored(self):
        too_small = frame(
            [9.8, 9.8, 10.2],
            [10.0, 10.8, 10.9],
            [9.5, 9.7, 10.1],
            [9.8, 10.5, 10.7],
        )
        self.assertFalse(
            any(event.formed_at == 2 for event in detect_fair_value_gaps(too_small))
        )
        no_gap = frame(
            [9.8, 9.8, 10.0],
            [10.0, 10.8, 11.0],
            [9.5, 9.7, 9.99],
            [9.8, 10.5, 10.5],
        )
        self.assertFalse(detect_fair_value_gaps(no_gap))


class OrderBlockTests(unittest.TestCase):
    def test_bos_displacement_creates_and_later_mitigates_order_block(self):
        bars = bullish_structure_bars()
        block = next(
            event
            for event in detect_order_blocks(bars, n=1)
            if event.displacement_index == 7
        )
        self.assertEqual(block.direction, "BULLISH")
        self.assertEqual(block.candle_index, 6)
        self.assertEqual((block.lower, block.upper), (10.8, 11.5))
        self.assertFalse(block.mitigated)
        self.assertEqual(block.known_at, 7)

        mitigated = bars.copy()
        mitigated.loc[8, "close"] = 10.7
        states = [
            event
            for event in detect_order_blocks(mitigated, n=1)
            if event.displacement_index == 7
        ]
        self.assertEqual(
            [(event.known_at, event.mitigated) for event in states],
            [(7, False), (8, True)],
        )
        updated = states[-1]
        self.assertTrue(updated.mitigated)
        self.assertEqual(updated.known_at, 8)

    def test_small_body_bos_does_not_create_displacement_order_block(self):
        bars = bullish_structure_bars()
        bars.loc[7, "open"] = 13.3
        self.assertTrue(
            any(
                event.kind == "BOS"
                for event in detect_structure_breaks(bars, n=1)
            )
        )
        self.assertFalse(detect_order_blocks(bars, n=1))


class LiquidityTests(unittest.TestCase):
    def test_equal_highs_and_lows_use_atr_tolerance(self):
        highs = [9.9, 10.0, 9.8, 9.95, 9.85, 10.01, 9.8]
        lows = [9.7, 9.8, 9.6, 9.7, 9.605, 9.8, 9.6]
        closes = [(top + bottom) / 2 for top, bottom in zip(highs, lows)]
        bars = frame(closes, highs, lows, closes)
        levels = detect_equal_highs_lows(bars, n=1)
        equal_high = next(level for level in levels if level.kind == "EQUAL_HIGH")
        equal_low = next(level for level in levels if level.kind == "EQUAL_LOW")
        self.assertEqual(equal_high.swing_indices, (1, 5))
        self.assertEqual(equal_low.swing_indices, (2, 4))
        self.assertEqual(equal_high.known_at, 6)
        self.assertEqual(equal_low.known_at, 5)

    def test_swings_outside_atr_tolerance_are_not_equal(self):
        highs = [9.9, 10.0, 9.8, 9.95, 9.85, 10.1, 9.8]
        lows = [9.7, 9.8, 9.6, 9.7, 9.68, 9.8, 9.6]
        closes = [(top + bottom) / 2 for top, bottom in zip(highs, lows)]
        levels = detect_equal_highs_lows(frame(closes, highs, lows, closes), n=1)
        self.assertFalse(levels)

    def test_sweep_uses_the_equal_level_snapshot_known_at_that_time(self):
        highs = [9.9, 10.0, 9.8, 9.95, 9.85, 10.01, 9.8, 10.1, 10.0, 10.01, 9.8]
        lows = [9.7, 9.8, 9.6, 9.7, 9.605, 9.8, 9.6, 9.8, 9.7, 9.8, 9.6]
        closes = [(top + bottom) / 2 for top, bottom in zip(highs, lows)]
        closes[7] = 10.04
        closes[8] = 9.9
        open_ = closes.copy()
        open_[7] = 9.95
        events = detect_liquidity_sweeps(
            frame(open_, highs, lows, closes), n=1
        )
        event = next(
            item
            for item in events
            if item.source == "EQUAL_HIGH" and item.sweep_index == 7
        )
        self.assertEqual(event.known_at, 8)

    def test_high_and_low_sweeps_need_a_later_close_back_inside(self):
        high_sweep = frame(
            [9.8, 9.9, 9.7, 9.9, 10.0, 9.8],
            [9.9, 10.0, 9.8, 10.2, 10.05, 9.9],
            [9.7, 9.8, 9.6, 9.85, 9.7, 9.6],
            [9.8, 9.9, 9.7, 10.05, 9.9, 9.8],
        )
        events = detect_liquidity_sweeps(high_sweep, n=1)
        upper = next(event for event in events if event.sweep_index == 3)
        self.assertEqual(upper.direction, "BEARISH")
        self.assertEqual(upper.source, "SWING")
        self.assertEqual(upper.known_at, 4)
        self.assertFalse(detect_liquidity_sweeps(high_sweep.iloc[:4], n=1))

        low_sweep = frame(
            [9.4, 9.1, 9.5, 9.1, 9.0, 9.3],
            [9.5, 9.3, 9.6, 9.1, 9.4, 9.5],
            [9.2, 9.0, 9.3, 8.8, 9.1, 9.2],
            [9.4, 9.1, 9.5, 8.95, 9.2, 9.4],
        )
        lower = next(
            event
            for event in detect_liquidity_sweeps(low_sweep, n=1)
            if event.sweep_index == 3
        )
        self.assertEqual(lower.direction, "BULLISH")
        self.assertEqual(lower.known_at, 4)

    def test_sweep_without_close_back_within_three_bars_is_ignored(self):
        bars = frame(
            [9.8, 9.9, 9.7, 9.9, 10.0, 10.0],
            [9.9, 10.0, 9.8, 10.2, 10.1, 10.1],
            [9.7, 9.8, 9.6, 9.85, 9.9, 9.9],
            [9.8, 9.9, 9.7, 10.05, 10.02, 10.01],
        )
        self.assertFalse(detect_liquidity_sweeps(bars, n=1))

    def test_sweep_bar_that_already_closed_inside_does_not_count_as_delayed_return(self):
        bars = frame(
            [9.8, 9.9, 9.7, 9.9, 9.9, 9.8],
            [9.9, 10.0, 9.8, 10.2, 10.05, 9.9],
            [9.7, 9.8, 9.6, 9.85, 9.7, 9.6],
            [9.8, 9.9, 9.7, 9.95, 9.9, 9.8],
        )
        self.assertFalse(detect_liquidity_sweeps(bars, n=1))


if __name__ == "__main__":
    unittest.main()
