"""Validated application settings for CURRY."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

TradingMode = Literal["paper", "live"]
NewsOpenTradeAction = Literal["none", "move_to_breakeven"]

DEFAULT_SYMBOLS = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURJPY",
    "XUAUSD",
    "GBPJPY",
)

TIMEFRAMES = {
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
}

STRATEGIES = (
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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class Settings:
    mode: TradingMode = "paper"
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    timeframe: str = "M15"
    strategy: str = "ma_crossover"
    poll_seconds: int = 15
    bars_to_fetch: int = 250
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 3.0
    max_open_trades: int = 3
    max_spread_points: int = 30
    reward_to_risk: float = 2.0
    atr_stop_multiplier: float = 1.5
    magic_number: int = 241114
    deviation_points: int = 20
    log_dir: str = "logs"
    news_calendar_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    news_timezone: str = "America/New_York"
    news_cache_seconds: int = 900
    news_timeout_seconds: float = 5.0
    news_block_before_minutes: int = 30
    news_block_after_minutes: int = 30
    news_open_trade_action: NewsOpenTradeAction = "move_to_breakeven"
    mt5_login: int | None = None
    mt5_password: str | None = None
    mt5_server: str | None = None
    mt5_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ValueError("CURRY_TRADING_MODE must be 'paper' or 'live'.")
        if not self.symbols or any(not symbol.strip() for symbol in self.symbols):
            raise ValueError("At least one non-empty MT5 symbol is required.")
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"Unsupported strategy: {self.strategy}. Choose from {', '.join(STRATEGIES)}."
            )
        if self.poll_seconds < 1 or self.bars_to_fetch < 250:
            raise ValueError("poll_seconds must be positive and bars_to_fetch at least 250.")
        if not 0.5 <= self.risk_per_trade_pct <= 1:
            raise ValueError("risk_per_trade_pct must be between 0.5% and 1%.")
        if not 0 < self.max_daily_loss_pct <= 100:
            raise ValueError("max_daily_loss_pct must be between 0 and 100.")
        if self.max_open_trades < 1 or self.max_spread_points < 0:
            raise ValueError("max_open_trades must be positive and max_spread_points non-negative.")
        if self.reward_to_risk < 2 or self.atr_stop_multiplier <= 0:
            raise ValueError("Reward-to-risk must be at least 2 and ATR stop multiplier positive.")
        if self.news_cache_seconds < 1 or self.news_timeout_seconds <= 0:
            raise ValueError("News cache duration and timeout must be positive.")
        if self.news_block_before_minutes < 0 or self.news_block_after_minutes < 0:
            raise ValueError("News blackout durations must be non-negative.")
        if self.news_open_trade_action not in ("none", "move_to_breakeven"):
            raise ValueError("Unsupported CURRY_NEWS_OPEN_TRADE_ACTION.")
        if self.mode == "live" and os.getenv("CURRY_LIVE_TRADING_CONFIRM") != "I_ACCEPT_LIVE_RISK":
            raise ValueError(
                "Live mode requires CURRY_LIVE_TRADING_CONFIRM=I_ACCEPT_LIVE_RISK."
            )

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("CURRY_TRADING_MODE", "paper").strip().lower()
        symbols = tuple(
            symbol.strip().upper()
            for symbol in os.getenv("CURRY_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",")
            if symbol.strip()
        )
        login_text = os.getenv("MT5_LOGIN")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            symbols=symbols,
            timeframe=os.getenv("CURRY_TIMEFRAME", "M15").strip().upper(),
            strategy=os.getenv("CURRY_STRATEGY", "ma_crossover").strip().lower(),
            poll_seconds=_env_int("CURRY_POLL_SECONDS", 15),
            bars_to_fetch=_env_int("CURRY_BARS_TO_FETCH", 250),
            risk_per_trade_pct=_env_float("CURRY_RISK_PER_TRADE_PCT", 0.5),
            max_daily_loss_pct=_env_float("CURRY_MAX_DAILY_LOSS_PCT", 3.0),
            max_open_trades=_env_int("CURRY_MAX_OPEN_TRADES", 3),
            max_spread_points=_env_int("CURRY_MAX_SPREAD_POINTS", 30),
            reward_to_risk=_env_float("CURRY_REWARD_TO_RISK", 2.0),
            atr_stop_multiplier=_env_float("CURRY_ATR_STOP_MULTIPLIER", 1.5),
            magic_number=_env_int("CURRY_MAGIC_NUMBER", 241114),
            deviation_points=_env_int("CURRY_DEVIATION_POINTS", 20),
            log_dir=os.getenv("CURRY_LOG_DIR", "logs"),
            news_calendar_url=os.getenv(
                "CURRY_NEWS_CALENDAR_URL",
                "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
            ),
            news_timezone=os.getenv("CURRY_NEWS_TIMEZONE", "America/New_York"),
            news_cache_seconds=_env_int("CURRY_NEWS_CACHE_SECONDS", 900),
            news_timeout_seconds=_env_float("CURRY_NEWS_TIMEOUT_SECONDS", 5.0),
            news_block_before_minutes=_env_int("CURRY_NEWS_BLOCK_BEFORE_MINUTES", 30),
            news_block_after_minutes=_env_int("CURRY_NEWS_BLOCK_AFTER_MINUTES", 30),
            news_open_trade_action=os.getenv(
                "CURRY_NEWS_OPEN_TRADE_ACTION", "move_to_breakeven"
            ).strip().lower(),  # type: ignore[arg-type]
            mt5_login=int(login_text) if login_text else None,
            mt5_password=os.getenv("MT5_PASSWORD"),
            mt5_server=os.getenv("MT5_SERVER"),
            mt5_path=os.getenv("MT5_TERMINAL_PATH"),
        )
