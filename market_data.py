"""MT5 connection and market-data adapter; no synthetic data fallbacks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config import Settings, TIMEFRAMES


class MarketData:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._mt5: Any = None

    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
            except ImportError as exc:
                raise RuntimeError(
                    "MetaTrader5 is unavailable. Install requirements on Windows with MT5 installed."
                ) from exc
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> None:
        options: dict[str, Any] = {}
        if self.settings.mt5_path:
            options["path"] = self.settings.mt5_path
        if self.settings.mt5_login is not None:
            options["login"] = self.settings.mt5_login
        if self.settings.mt5_password:
            options["password"] = self.settings.mt5_password
        if self.settings.mt5_server:
            options["server"] = self.settings.mt5_server
        if not self.mt5.initialize(**options):
            raise RuntimeError(f"MT5 initialization failed: {self.mt5.last_error()}")
        account = self.mt5.account_info()
        if account is None:
            raise RuntimeError(f"MT5 account information unavailable: {self.mt5.last_error()}")

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()

    def account_info(self) -> Any:
        account = self.mt5.account_info()
        if account is None:
            raise RuntimeError(f"Cannot read MT5 account: {self.mt5.last_error()}")
        return account

    def symbol_info(self, symbol: str) -> Any:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol {symbol} is not available in MT5: {self.mt5.last_error()}")
        if not info.visible and not self.mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not enable MT5 symbol {symbol}: {self.mt5.last_error()}")
        return self.mt5.symbol_info(symbol)

    def tick(self, symbol: str) -> Any:
        self.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid <= 0 or tick.ask <= 0 or tick.ask < tick.bid:
            raise RuntimeError(f"Invalid or unavailable live quote for {symbol}: {self.mt5.last_error()}")
        return tick

    def fetch_bars(self, symbol: str, count: int, closed_only: bool = True) -> pd.DataFrame:
        self.symbol_info(symbol)
        timeframe = getattr(self.mt5, f"TIMEFRAME_{self.settings.timeframe}")
        requested = count + 1 if closed_only else count
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, requested)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No MT5 bars available for {symbol}: {self.mt5.last_error()}")
        bars = pd.DataFrame(rates)
        bars["time"] = pd.to_datetime(bars["time"], unit="s", utc=True)
        bars = bars.set_index("time").sort_index()
        if closed_only:
            bars = bars.iloc[:-1]
        if len(bars) < min(count, 50):
            raise RuntimeError(f"Insufficient closed MT5 bars for {symbol}: received {len(bars)}.")
        return bars.tail(count)

    def fetch_history(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("Historical range must use timezone-aware start < end.")
        self.symbol_info(symbol)
        timeframe = getattr(self.mt5, f"TIMEFRAME_{self.settings.timeframe}")
        rates = self.mt5.copy_rates_range(
            symbol,
            timeframe,
            start.astimezone(timezone.utc),
            end.astimezone(timezone.utc),
        )
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No historical MT5 bars for {symbol}: {self.mt5.last_error()}")
        bars = pd.DataFrame(rates)
        bars["time"] = pd.to_datetime(bars["time"], unit="s", utc=True)
        return bars.set_index("time").sort_index()
