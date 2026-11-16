import logging
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from config import Settings
from filters.news import (
    CalendarUnavailableError,
    ForexFactoryCalendar,
    NewsFilter,
    map_currencies_to_pairs,
)
from trade_executor import PaperPosition, TradeExecutor


FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<weeklyevents>
  <event>
    <title>Federal Funds Rate Decision</title>
    <country>USD</country>
    <date>11-14-2026</date>
    <time>10:00am</time>
    <impact>High</impact>
    <url>https://example.test/rate</url>
  </event>
  <event>
    <title>Non-Farm Payrolls</title>
    <country>USD</country>
    <date>11-14-2026</date>
    <time>12:00pm</time>
    <impact>High</impact>
  </event>
  <event>
    <title>Consumer Price Index y/y</title>
    <country>EUR</country>
    <date>11-14-2026</date>
    <time>2:00pm</time>
    <impact>Medium</impact>
  </event>
</weeklyevents>
"""


class ForexFactoryCalendarTests(unittest.TestCase):
    def test_parses_impacts_flags_and_utc_times(self):
        calendar = ForexFactoryCalendar(
            timezone_name="UTC",
            fetcher=lambda: FEED,
        )
        events = calendar.get_events()
        self.assertEqual([event.impact for event in events], ["HIGH", "HIGH", "MEDIUM"])
        self.assertTrue(events[0].is_rate_decision)
        self.assertTrue(events[1].is_nfp)
        self.assertTrue(events[2].is_cpi)
        self.assertEqual(
            events[0].event_time_utc,
            datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc),
        )

    def test_cache_avoids_fetching_until_ttl_expires(self):
        calls = []
        current_time = [100.0]

        def fetch():
            calls.append(1)
            return FEED

        calendar = ForexFactoryCalendar(
            timezone_name="UTC",
            cache_seconds=60,
            fetcher=fetch,
            clock=lambda: current_time[0],
        )
        calendar.get_events()
        current_time[0] += 59
        calendar.get_events()
        self.assertEqual(len(calls), 1)
        current_time[0] += 1
        calendar.get_events()
        self.assertEqual(len(calls), 2)

    def test_failed_refresh_does_not_return_stale_cache(self):
        current_time = [100.0]
        should_fail = [False]

        def fetch():
            if should_fail[0]:
                raise OSError("calendar offline")
            return FEED

        calendar = ForexFactoryCalendar(
            timezone_name="UTC",
            cache_seconds=5,
            fetcher=fetch,
            clock=lambda: current_time[0],
        )
        cached = calendar.get_events()
        self.assertTrue(cached)
        current_time[0] += 5
        should_fail[0] = True
        with self.assertRaises(CalendarUnavailableError):
            calendar.get_events()


class NewsFilterTests(unittest.TestCase):
    def setUp(self):
        self.calendar = ForexFactoryCalendar(
            timezone_name="UTC",
            fetcher=lambda: FEED,
        )
        self.news_filter = NewsFilter(
            calendar=self.calendar,
            symbols=("EURUSD.a", "USDJPY", "EURJPY", "XUAUSD"),
            block_before_minutes=30,
            block_after_minutes=30,
            timezone_name="UTC",
        )
        self.assertTrue(self.news_filter.refresh())

    def test_high_impact_window_blocks_only_pairs_with_event_currency(self):
        now = datetime(2026, 11, 14, 9, 30, tzinfo=timezone.utc)
        self.assertIsNotNone(self.news_filter.entry_rejection("EURUSD.a", now))
        self.assertIsNotNone(self.news_filter.entry_rejection("USDJPY", now))
        self.assertIsNotNone(self.news_filter.entry_rejection("XUAUSD", now))
        self.assertIsNone(self.news_filter.entry_rejection("EURJPY", now))

    def test_blackout_boundaries_are_inclusive_and_outside_window_is_allowed(self):
        before = datetime(2026, 11, 14, 9, 29, 59, tzinfo=timezone.utc)
        start = datetime(2026, 11, 14, 9, 30, tzinfo=timezone.utc)
        end = datetime(2026, 11, 14, 10, 30, tzinfo=timezone.utc)
        after = datetime(2026, 11, 14, 10, 30, 1, tzinfo=timezone.utc)
        self.assertIsNone(self.news_filter.entry_rejection("EURUSD.a", before))
        self.assertIsNotNone(self.news_filter.entry_rejection("EURUSD.a", start))
        self.assertIsNotNone(self.news_filter.entry_rejection("EURUSD.a", end))
        self.assertIsNone(self.news_filter.entry_rejection("EURUSD.a", after))

    def test_medium_impact_event_does_not_block_entries(self):
        now = datetime(2026, 11, 14, 13, 30, tzinfo=timezone.utc)
        self.assertIsNone(self.news_filter.entry_rejection("EURUSD.a", now))
        self.assertFalse(self.news_filter.affected_symbols(now))

    def test_currency_mapping_handles_broker_suffixes_and_requested_symbol(self):
        mapping = map_currencies_to_pairs(("EURUSD.a", "XUAUSD"))
        self.assertEqual(mapping["EUR"], frozenset({"EURUSD.a"}))
        self.assertEqual(mapping["USD"], frozenset({"EURUSD.a", "XUAUSD"}))
        self.assertEqual(mapping["XUA"], frozenset({"XUAUSD"}))

    def test_fetch_failure_fails_closed_and_alerts_once_until_recovery(self):
        logger = Mock(spec=logging.Logger)
        alerts = []
        calendar = ForexFactoryCalendar(
            timezone_name="UTC",
            fetcher=lambda: (_ for _ in ()).throw(OSError("offline")),
        )
        news_filter = NewsFilter(
            calendar=calendar,
            symbols=("EURUSD",),
            logger=logger,
            alert=alerts.append,
            timezone_name="UTC",
        )
        self.assertFalse(news_filter.refresh())
        self.assertFalse(news_filter.refresh())
        self.assertIsNotNone(news_filter.entry_rejection("EURUSD"))
        logger.error.assert_called_once()
        self.assertEqual(len(alerts), 1)

    def test_unknown_impact_makes_calendar_unavailable(self):
        invalid_feed = FEED.replace(b"<impact>High</impact>", b"<impact>Critical</impact>", 1)
        calendar = ForexFactoryCalendar(
            timezone_name="UTC",
            fetcher=lambda: invalid_feed,
        )
        with self.assertRaises(CalendarUnavailableError):
            calendar.get_events()

    def test_invalid_currency_makes_calendar_unavailable(self):
        invalid_feed = FEED.replace(b"<country>USD</country>", b"<country>US</country>", 1)
        calendar = ForexFactoryCalendar(
            timezone_name="UTC",
            fetcher=lambda: invalid_feed,
        )
        with self.assertRaises(CalendarUnavailableError):
            calendar.get_events()


class NewsBreakevenTests(unittest.TestCase):
    def test_paper_positions_move_to_entry_only_for_affected_symbols(self):
        market_data = Mock()
        market_data.tick.return_value = SimpleNamespace(bid=1.12, ask=1.1201)
        executor = TradeExecutor(Settings(), market_data=market_data, logger=Mock())
        affected = PaperPosition("EURUSD", "buy", 0.1, 1.1, 1.09, 1.12, datetime.now(timezone.utc))
        other = PaperPosition("GBPUSD", "sell", 0.1, 1.3, 1.31, 1.27, datetime.now(timezone.utc))
        executor._paper_positions.extend((affected, other))

        executor.move_to_breakeven({"EURUSD"})

        self.assertEqual(affected.stop_loss, affected.entry)
        self.assertEqual(other.stop_loss, 1.31)

    def test_paper_breakeven_waits_until_position_is_profitable(self):
        market_data = Mock()
        market_data.tick.return_value = SimpleNamespace(bid=1.09, ask=1.0901)
        executor = TradeExecutor(Settings(), market_data=market_data, logger=Mock())
        position = PaperPosition(
            "EURUSD", "buy", 0.1, 1.1, 1.09, 1.12, datetime.now(timezone.utc)
        )
        executor._paper_positions.append(position)

        executor.move_to_breakeven({"EURUSD"})

        self.assertEqual(position.stop_loss, 1.09)


if __name__ == "__main__":
    unittest.main()
