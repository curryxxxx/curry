"""Paper/live order execution with broker-side SL/TP for live orders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any

from config import Settings
from logger import record_trade
from market_data import MarketData
from strategy import Signal


@dataclass
class PaperPosition:
    symbol: str
    side: str
    volume: float
    entry: float
    stop_loss: float
    take_profit: float
    opened_at: datetime


class TradeExecutor:
    def __init__(self, settings: Settings, market_data: MarketData, logger: Any):
        self.settings = settings
        self.market_data = market_data
        self.logger = logger
        self._paper_positions: list[PaperPosition] = []
        self._paper_daily_pnl = 0.0
        self._paper_pnl_date = datetime.now(timezone.utc).date()

    def open_positions_count(self) -> int:
        if self.settings.mode == "paper":
            return len(self._paper_positions)
        positions = self.market_data.mt5.positions_get()
        if positions is None:
            raise RuntimeError(f"Cannot read open MT5 positions: {self.market_data.mt5.last_error()}")
        return sum(1 for position in positions if position.magic == self.settings.magic_number)

    def daily_realized_pnl(self) -> float:
        today = datetime.now(timezone.utc).date()
        if self.settings.mode == "paper":
            if today != self._paper_pnl_date:
                self._paper_daily_pnl = 0.0
                self._paper_pnl_date = today
            return self._paper_daily_pnl
        start = datetime.combine(today, time.min, tzinfo=timezone.utc)
        deals = self.market_data.mt5.history_deals_get(start, datetime.now(timezone.utc))
        if deals is None:
            raise RuntimeError(f"Cannot read today's MT5 deals: {self.market_data.mt5.last_error()}")
        return sum(
            deal.profit + deal.commission + deal.swap + getattr(deal, "fee", 0.0)
            for deal in deals
            if deal.magic == self.settings.magic_number
        )

    def open_trade(self, signal: Signal, volume: float, tick: Any) -> None:
        entry = float(tick.ask if signal.side == "buy" else tick.bid)
        if signal.side == "buy" and not signal.stop_loss < entry < signal.take_profit:
            raise ValueError("BUY stop/target levels do not bracket the current market price.")
        if signal.side == "sell" and not signal.take_profit < entry < signal.stop_loss:
            raise ValueError("SELL stop/target levels do not bracket the current market price.")

        if self.settings.mode == "paper":
            position = PaperPosition(
                signal.symbol,
                signal.side,
                volume,
                entry,
                signal.stop_loss,
                signal.take_profit,
                datetime.now(timezone.utc),
            )
            self._paper_positions.append(position)
            self.logger.info(
                "PAPER opened %s %s %.2f lots at %.5f SL=%.5f TP=%.5f",
                signal.side.upper(),
                signal.symbol,
                volume,
                entry,
                signal.stop_loss,
                signal.take_profit,
            )
            return

        mt5 = self.market_data.mt5
        symbol_info = self.market_data.symbol_info(signal.symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal.side == "buy" else mt5.ORDER_TYPE_SELL,
            "price": entry,
            "sl": round(signal.stop_loss, symbol_info.digits),
            "tp": round(signal.take_profit, symbol_info.digits),
            "deviation": self.settings.deviation_points,
            "magic": self.settings.magic_number,
            "comment": "CURRY",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": symbol_info.filling_mode,
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"MT5 order_send returned no result: {mt5.last_error()}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(
                f"MT5 rejected {signal.side} {signal.symbol}: "
                f"retcode={result.retcode}, comment={result.comment}"
            )
        self.logger.info(
            "LIVE opened %s %s %.2f lots at %.5f ticket=%s",
            signal.side.upper(),
            signal.symbol,
            volume,
            entry,
            result.order,
        )

    def move_to_breakeven(self, symbols: set[str] | frozenset[str]) -> None:
        """Tighten this bot's affected positions to entry when broker rules allow."""
        if not symbols:
            return
        if self.settings.mode == "paper":
            for position in self._paper_positions:
                if position.symbol not in symbols:
                    continue
                tick = self.market_data.tick(position.symbol)
                if position.side == "buy" and position.stop_loss < position.entry:
                    if tick.bid <= position.entry:
                        continue
                    position.stop_loss = position.entry
                elif position.side == "sell" and position.stop_loss > position.entry:
                    if tick.ask >= position.entry:
                        continue
                    position.stop_loss = position.entry
                else:
                    continue
                self.logger.info(
                    "PAPER moved %s %s stop to breakeven %.5f",
                    position.side.upper(),
                    position.symbol,
                    position.entry,
                )
            return

        mt5 = self.market_data.mt5
        positions = mt5.positions_get()
        if positions is None:
            raise RuntimeError(f"Cannot read open MT5 positions: {mt5.last_error()}")
        for position in positions:
            if position.magic != self.settings.magic_number or position.symbol not in symbols:
                continue
            info = self.market_data.symbol_info(position.symbol)
            tick = self.market_data.tick(position.symbol)
            point = float(info.point)
            if point <= 0:
                raise RuntimeError(f"Invalid MT5 point size for {position.symbol}.")
            min_distance = max(
                int(getattr(info, "trade_stops_level", 0)),
                int(getattr(info, "trade_freeze_level", 0)),
            ) * point
            entry = float(position.price_open)
            old_stop = float(position.sl)
            new_stop = round(entry, info.digits)
            if position.type == mt5.POSITION_TYPE_BUY:
                if old_stop > 0 and old_stop >= new_stop:
                    continue
                if tick.bid - new_stop <= min_distance:
                    self.logger.info(
                        "LIVE breakeven stop skipped for %s ticket=%s: broker stop distance",
                        position.symbol,
                        position.ticket,
                    )
                    continue
            elif position.type == mt5.POSITION_TYPE_SELL:
                if old_stop > 0 and old_stop <= new_stop:
                    continue
                if new_stop - tick.ask <= min_distance:
                    self.logger.info(
                        "LIVE breakeven stop skipped for %s ticket=%s: broker stop distance",
                        position.symbol,
                        position.ticket,
                    )
                    continue
            else:
                continue

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "symbol": position.symbol,
                "sl": new_stop,
                "tp": position.tp,
                "magic": self.settings.magic_number,
                "comment": "CURRY news breakeven",
            }
            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError(
                    f"MT5 stop modification returned no result: {mt5.last_error()}"
                )
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(
                    f"MT5 rejected breakeven stop for ticket={position.ticket}: "
                    f"retcode={result.retcode}, comment={result.comment}"
                )
            self.logger.info(
                "LIVE moved %s ticket=%s stop to breakeven %.5f",
                position.symbol,
                position.ticket,
                new_stop,
            )

    def update_paper_positions(self, symbol: str, tick: Any) -> None:
        if self.settings.mode != "paper":
            return
        remaining: list[PaperPosition] = []
        for position in self._paper_positions:
            if position.symbol != symbol:
                remaining.append(position)
                continue
            exit_price: float | None = None
            reason = ""
            if position.side == "buy":
                if tick.bid <= position.stop_loss:
                    exit_price, reason = position.stop_loss, "stop_loss"
                elif tick.bid >= position.take_profit:
                    exit_price, reason = position.take_profit, "take_profit"
            else:
                if tick.ask >= position.stop_loss:
                    exit_price, reason = position.stop_loss, "stop_loss"
                elif tick.ask <= position.take_profit:
                    exit_price, reason = position.take_profit, "take_profit"
            if exit_price is None:
                remaining.append(position)
                continue

            mt5 = self.market_data.mt5
            order_type = mt5.ORDER_TYPE_BUY if position.side == "buy" else mt5.ORDER_TYPE_SELL
            profit = mt5.order_calc_profit(
                order_type, symbol, position.volume, position.entry, exit_price
            )
            if profit is None:
                raise RuntimeError(f"Cannot calculate paper P&L for {symbol}: {mt5.last_error()}")
            self._paper_daily_pnl += profit
            record_trade(
                self.settings.log_dir,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": "paper",
                    "symbol": symbol,
                    "side": position.side,
                    "volume": position.volume,
                    "entry": position.entry,
                    "exit": exit_price,
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "profit": profit,
                    "reason": reason,
                    "ticket": "",
                },
            )
            self.logger.info("PAPER closed %s %s P&L=%.2f via %s", symbol, position.side, profit, reason)
        self._paper_positions = remaining
