"""Forex Factory calendar ingestion, caching, pair mapping, and fail-closed gate."""

from __future__ import annotations

import logging
import re
import time
import http.client
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Callable, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

Impact = Literal["LOW", "MEDIUM", "HIGH"]
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
_PAIR_PATTERN = re.compile(r"([A-Z]{3})([A-Z]{3})")
_RATE_DECISION_PATTERN = re.compile(
    r"\b(rate decision|interest rate|cash rate|bank rate|policy rate|"
    r"federal funds rate|main refinancing rate|monetary policy decision)\b",
    re.IGNORECASE,
)
_NFP_PATTERN = re.compile(
    r"\b(nfp|non[- ]farm(?: payrolls?| employment(?: change)?)?)\b",
    re.IGNORECASE,
)
_CPI_PATTERN = re.compile(r"\b(cpi|consumer price index)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    currency: str
    impact: Impact
    event_time_utc: datetime | None
    event_date: date
    is_rate_decision: bool
    is_nfp: bool
    is_cpi: bool
    url: str = ""

    @property
    def all_day(self) -> bool:
        return self.event_time_utc is None


class CalendarUnavailableError(RuntimeError):
    """Calendar data could not be fetched or validated."""


def map_currencies_to_pairs(symbols: tuple[str, ...] | list[str]) -> dict[str, frozenset[str]]:
    """Map ISO-like three-character base/quote codes to exact broker symbols."""
    mapping: dict[str, set[str]] = {}
    for symbol in symbols:
        match = _PAIR_PATTERN.search(symbol.upper())
        if match is None:
            raise ValueError(f"Cannot extract a currency pair from symbol {symbol!r}.")
        for currency in match.groups():
            mapping.setdefault(currency, set()).add(symbol)
    return {currency: frozenset(pairs) for currency, pairs in mapping.items()}


def _impact(value: str) -> Impact:
    normalized = value.strip().casefold()
    if normalized == "high":
        return "HIGH"
    if normalized == "medium":
        return "MEDIUM"
    if normalized in ("low", "holiday", "none", "non-economic"):
        return "LOW"
    raise ValueError(f"Unknown Forex Factory impact value: {value!r}")


def _text(element: ET.Element, field: str, required: bool = True) -> str:
    child = element.find(field)
    value = "" if child is None or child.text is None else child.text.strip()
    if required and not value:
        raise ValueError(f"Forex Factory event is missing {field!r}.")
    return value


class ForexFactoryCalendar:
    """Fetch and cache the public weekly Forex Factory XML feed.

    Feed times have no timezone field. They are interpreted using the configured
    Forex Factory calendar timezone and converted to UTC before filtering.
    Expired cache entries are never used after a failed refresh.
    """

    def __init__(
        self,
        timezone_name: str = "America/New_York",
        cache_seconds: int = 900,
        timeout_seconds: float = 5.0,
        url: str = CALENDAR_URL,
        fetcher: Callable[[], bytes] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if cache_seconds < 1:
            raise ValueError("Calendar cache duration must be positive.")
        if timeout_seconds <= 0:
            raise ValueError("Calendar timeout must be positive.")
        if not url.lower().startswith("https://"):
            raise ValueError("The economic calendar URL must use HTTPS.")
        try:
            self._source_timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown calendar timezone: {timezone_name}") from exc
        self.url = url
        self.cache_seconds = cache_seconds
        self.timeout_seconds = timeout_seconds
        self._fetcher = fetcher or self._fetch
        self._clock = clock
        self._cached_events: tuple[CalendarEvent, ...] | None = None
        self._cache_expires_at = 0.0

    def _fetch(self) -> bytes:
        request = urllib.request.Request(
            self.url,
            headers={"User-Agent": "CURRY-Forex-Trading-Bot/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read()

    def get_events(self) -> tuple[CalendarEvent, ...]:
        now = self._clock()
        if self._cached_events is not None and now < self._cache_expires_at:
            return self._cached_events

        try:
            payload = self._fetcher()
        except (OSError, TimeoutError, http.client.HTTPException, urllib.error.URLError) as exc:
            raise CalendarUnavailableError(f"Forex Factory fetch failed: {exc}") from exc

        try:
            events = self._parse(payload)
        except (ET.ParseError, TypeError, UnicodeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise CalendarUnavailableError(f"Forex Factory feed is invalid: {exc}") from exc
        self._cached_events = tuple(events)
        self._cache_expires_at = now + self.cache_seconds
        return self._cached_events

    def _parse(self, payload: bytes) -> list[CalendarEvent]:
        root = ET.fromstring(payload)
        if root.tag.casefold() != "weeklyevents":
            raise ValueError(f"Unexpected calendar root element: {root.tag!r}")
        elements = root.findall("event")
        if not elements:
            raise ValueError("Calendar feed contains no events.")

        events: list[CalendarEvent] = []
        for element in elements:
            title = _text(element, "title")
            currency = _text(element, "country").upper()
            raw_date = _text(element, "date")
            raw_time = _text(element, "time", required=False)
            impact = _impact(_text(element, "impact"))
            if currency not in ("ALL", "GLOBAL") and not re.fullmatch(r"[A-Z]{3}", currency):
                raise ValueError(f"Invalid event currency: {currency!r}")
            event_date = datetime.strptime(raw_date, "%m-%d-%Y").date()
            event_time = self._parse_time(event_date, raw_time)
            events.append(
                CalendarEvent(
                    title=title,
                    currency=currency,
                    impact=impact,
                    event_time_utc=event_time,
                    event_date=event_date,
                    is_rate_decision=bool(_RATE_DECISION_PATTERN.search(title)),
                    is_nfp=bool(_NFP_PATTERN.search(title)),
                    is_cpi=bool(_CPI_PATTERN.search(title)),
                    url=_text(element, "url", required=False),
                )
            )
        return events

    def _parse_time(self, event_date: date, raw_time: str) -> datetime | None:
        normalized = raw_time.strip().casefold()
        if normalized in ("", "all day", "tentative", "time not supplied"):
            return None
        parsed_time: datetime_time | None = None
        for pattern in ("%I:%M%p", "%I:%M %p"):
            try:
                parsed_time = datetime.strptime(normalized.upper(), pattern).time()
                break
            except ValueError:
                continue
        if parsed_time is None:
            raise ValueError(f"Unsupported Forex Factory event time: {raw_time!r}")
        local_time = datetime.combine(event_date, parsed_time, tzinfo=self._source_timezone)
        return local_time.astimezone(timezone.utc)


class NewsFilter:
    """HIGH-impact blackout filter with fail-closed refresh and optional alert."""

    def __init__(
        self,
        calendar: ForexFactoryCalendar,
        symbols: tuple[str, ...],
        block_before_minutes: int = 30,
        block_after_minutes: int = 30,
        logger: logging.Logger | None = None,
        alert: Callable[[str], None] | None = None,
        timezone_name: str = "America/New_York",
    ):
        if block_before_minutes < 0 or block_after_minutes < 0:
            raise ValueError("News blackout durations must be non-negative.")
        self.calendar = calendar
        self.currency_pairs = map_currencies_to_pairs(symbols)
        self._symbols = frozenset(symbols)
        self.block_before = timedelta(minutes=block_before_minutes)
        self.block_after = timedelta(minutes=block_after_minutes)
        try:
            self._timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown calendar timezone: {timezone_name}") from exc
        self.logger = logger
        self.alert = alert
        self._events: tuple[CalendarEvent, ...] | None = None
        self._failure_alerted = False

    def refresh(self) -> bool:
        try:
            self._events = self.calendar.get_events()
        except CalendarUnavailableError as exc:
            self._events = None
            message = f"NEWS FILTER FAIL-CLOSED: no new entries; calendar unavailable: {exc}"
            if not self._failure_alerted:
                if self.logger is not None:
                    self.logger.error(message)
                if self.alert is not None:
                    self.alert(message)
            self._failure_alerted = True
            return False
        self._failure_alerted = False
        return True

    def affected_symbols(self, now: datetime | None = None) -> frozenset[str]:
        """Return symbols inside a currently active HIGH-impact news window."""
        if self._events is None:
            return frozenset()
        current_utc = self._utc_now(now)
        affected: set[str] = set()
        for event in self._events:
            if event.impact != "HIGH":
                continue
            if not self._event_is_blocked(event, current_utc):
                continue
            affected.update(self._pairs_for_event(event))
        return frozenset(affected)

    def entry_rejection(self, symbol: str, now: datetime | None = None) -> str | None:
        if self._events is None:
            return "economic calendar unavailable; entries are blocked"
        if symbol not in self._symbols:
            return f"symbol {symbol!r} is not configured for news filtering"
        current_utc = self._utc_now(now)
        for event in self._events:
            if event.impact != "HIGH" or symbol not in self._pairs_for_event(event):
                continue
            if self._event_is_blocked(event, current_utc):
                return (
                    f"HIGH-impact {event.currency} event {event.title!r} "
                    "is within the configured blackout window"
                )
        return None

    def _utc_now(self, now: datetime | None) -> datetime:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("News-filter time must be timezone-aware.")
        return current.astimezone(timezone.utc)

    def _event_is_blocked(self, event: CalendarEvent, current_utc: datetime) -> bool:
        if event.all_day:
            return current_utc.astimezone(self._timezone).date() == event.event_date
        assert event.event_time_utc is not None
        start = event.event_time_utc - self.block_before
        end = event.event_time_utc + self.block_after
        return start <= current_utc <= end

    def _pairs_for_event(self, event: CalendarEvent) -> frozenset[str]:
        if event.currency in ("ALL", "GLOBAL"):
            return self._symbols
        return self.currency_pairs.get(event.currency, frozenset())
