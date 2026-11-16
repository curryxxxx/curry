"""DST-aware Forex session windows and strategy session configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from config import STRATEGIES


@dataclass(frozen=True)
class SessionDefinition:
    timezone: str
    start: time
    end: time


SESSION_DEFINITIONS = {
    "asian": SessionDefinition("Asia/Tokyo", time(9), time(18)),
    "london": SessionDefinition("Europe/London", time(8), time(17)),
    "new_york": SessionDefinition("America/New_York", time(8), time(17)),
}
DEFAULT_SESSION_CONFIG = Path(__file__).with_name("sessions.yaml")


def _normalize_session(session: str) -> str:
    if not isinstance(session, str):
        raise ValueError(f"Session names must be strings, got {type(session).__name__}.")
    normalized = session.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in SESSION_DEFINITIONS:
        raise ValueError(
            f"Unknown session {session!r}; choose from {', '.join(SESSION_DEFINITIONS)}."
        )
    return normalized


def _aware_utc(now: datetime | None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("Session-check time must be timezone-aware.")
    return current.astimezone(timezone.utc)


def is_active(session: str, now: datetime | None = None) -> bool:
    """Return whether the given session's local clock is within its half-open window."""
    name = _normalize_session(session)
    definition = SESSION_DEFINITIONS[name]
    local_now = _aware_utc(now).astimezone(ZoneInfo(definition.timezone))
    if local_now.weekday() >= 5:
        return False
    local_time = local_now.timetz().replace(tzinfo=None)
    return definition.start <= local_time < definition.end


def current_sessions(now: datetime | None = None) -> frozenset[str]:
    """Return all sessions active at the same timezone-aware instant."""
    current = _aware_utc(now)
    return frozenset(
        name for name in SESSION_DEFINITIONS if is_active(name, current)
    )


def is_overlap(now: datetime | None = None) -> bool:
    """Return whether London and New York are simultaneously active."""
    current = _aware_utc(now)
    return is_active("london", current) and is_active("new_york", current)


def load_strategy_sessions(
    path: str | Path = DEFAULT_SESSION_CONFIG,
) -> dict[str, frozenset[str]]:
    """Load and validate the per-strategy allowed sessions YAML mapping."""
    config_path = Path(path)
    try:
        data: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not load session configuration {config_path}: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("strategies"), dict):
        raise ValueError("Session configuration must contain a 'strategies' mapping.")

    configured = data["strategies"]
    if any(not isinstance(strategy, str) for strategy in configured):
        raise ValueError("Strategy names in session configuration must be strings.")
    expected = set(STRATEGIES)
    actual = set(configured)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing strategies: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown strategies: {', '.join(sorted(unknown))}")
        raise ValueError(
            "Invalid strategy session configuration (" + "; ".join(details) + ")."
        )

    result: dict[str, frozenset[str]] = {}
    for strategy, sessions in configured.items():
        if not isinstance(sessions, list) or any(
            not isinstance(session, str) for session in sessions
        ):
            raise ValueError(f"Sessions for strategy {strategy!r} must be a YAML list of names.")
        result[strategy] = frozenset(_normalize_session(session) for session in sessions)
    return result
