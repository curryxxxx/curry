"""Entry filters, daily loss protection, and broker-aware lot sizing."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from config import Settings
from strategy import Signal


class RiskManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def entry_rejection(
        self,
        signal: Signal,
        tick: Any,
        symbol_info: Any,
        open_trades: int,
        daily_pnl: float,
        day_start_balance: float,
        now: datetime | None = None,
    ) -> str | None:
        now = now or datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return "Forex market is closed for the weekend"
        spread_points = (tick.ask - tick.bid) / symbol_info.point
        if spread_points > self.settings.max_spread_points:
            return f"spread {spread_points:.1f} points exceeds limit"
        if open_trades >= self.settings.max_open_trades:
            return "maximum open trades reached"
        if day_start_balance <= 0:
            return "account balance is not positive"
        if daily_pnl <= -(day_start_balance * self.settings.max_daily_loss_pct / 100):
            return "maximum daily loss reached"
        return None

    def calculate_volume(
        self,
        signal: Signal,
        account_balance: float,
        symbol_info: Any,
        mt5: Any,
        entry_price: float | None = None,
    ) -> float | None:
        if account_balance <= 0:
            raise ValueError("Account balance must be positive.")
        entry = signal.reference_price if entry_price is None else entry_price
        if signal.side == "buy":
            entry_type = mt5.ORDER_TYPE_BUY
            stop_distance = entry - signal.stop_loss
        else:
            entry_type = mt5.ORDER_TYPE_SELL
            stop_distance = signal.stop_loss - entry
        if stop_distance <= 0:
            raise ValueError("Stop loss is on the wrong side of the entry price.")
        loss_per_lot = mt5.order_calc_profit(
            entry_type, signal.symbol, 1.0, entry, signal.stop_loss
        )
        if loss_per_lot is None or not math.isfinite(loss_per_lot) or loss_per_lot == 0:
            raise RuntimeError(
                f"MT5 could not calculate stop-loss value for {signal.symbol}: {mt5.last_error()}"
            )
        risk_cash = account_balance * self.settings.risk_per_trade_pct / 100
        raw_volume = risk_cash / abs(loss_per_lot)
        step = float(symbol_info.volume_step)
        minimum = float(symbol_info.volume_min)
        maximum = float(symbol_info.volume_max)
        if step <= 0 or minimum <= 0 or maximum < minimum:
            raise RuntimeError(f"Invalid MT5 volume constraints for {signal.symbol}.")
        volume = min(math.floor(raw_volume / step) * step, maximum)
        if volume < minimum:
            return None
        precision = max(0, min(8, int(round(-math.log10(step))) if step < 1 else 0))
        return round(volume, precision)
