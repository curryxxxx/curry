"""CURRY MT5 trading-bot entry point."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from config import Settings
from dashboard import render_status
from filters.news import ForexFactoryCalendar, NewsFilter
from filters.sessions import current_sessions, load_strategy_sessions
from logger import configure_logging
from market_data import MarketData
from risk_manager import RiskManager
from strategy import TradingStrategy
from trade_executor import TradeExecutor


def run() -> None:
    settings = Settings.from_env()
    logger = configure_logging(settings.log_dir)
    market_data = MarketData(settings)
    risk_manager = RiskManager(settings)
    strategy = TradingStrategy(
        settings.strategy, settings.atr_stop_multiplier, settings.reward_to_risk
    )
    executor = TradeExecutor(settings, market_data, logger)
    strategy_sessions = load_strategy_sessions()
    news_filter = NewsFilter(
        calendar=ForexFactoryCalendar(
            timezone_name=settings.news_timezone,
            cache_seconds=settings.news_cache_seconds,
            timeout_seconds=settings.news_timeout_seconds,
            url=settings.news_calendar_url,
        ),
        symbols=settings.symbols,
        block_before_minutes=settings.news_block_before_minutes,
        block_after_minutes=settings.news_block_after_minutes,
        logger=logger,
        timezone_name=settings.news_timezone,
    )
    last_signal_time = {}

    try:
        market_data.connect()
        account = market_data.account_info()
        logger.info(
            "Connected to MT5 account %s in %s mode; live orders %s.",
            account.login,
            settings.mode,
            "enabled only by explicit confirmation" if settings.mode == "live" else "disabled",
        )
        while True:
            try:
                news_healthy = news_filter.refresh()
                active_sessions = current_sessions(datetime.now(timezone.utc))
                allowed_sessions = strategy_sessions[settings.strategy]
                if not allowed_sessions.intersection(active_sessions):
                    logger.debug(
                        "%s strategy idle; none of its configured sessions are active.",
                        settings.strategy,
                    )
                daily_pnl = executor.daily_realized_pnl()
                current_account = market_data.account_info()
                day_start_balance = (
                    current_account.balance - daily_pnl
                    if settings.mode == "live"
                    else current_account.balance
                )
                for symbol in settings.symbols:
                    tick = market_data.tick(symbol)
                    executor.update_paper_positions(symbol, tick)

                if (
                    news_healthy
                    and settings.news_open_trade_action == "move_to_breakeven"
                ):
                    executor.move_to_breakeven(news_filter.affected_symbols())

                daily_pnl = executor.daily_realized_pnl()
                render_status(
                    settings,
                    current_account,
                    executor.open_positions_count(),
                    daily_pnl,
                    settings.symbols,
                )
                for symbol in settings.symbols:
                    if not allowed_sessions.intersection(active_sessions):
                        continue
                    if not news_healthy:
                        continue
                    news_rejection = news_filter.entry_rejection(symbol)
                    if news_rejection:
                        logger.info("News filter rejected entry for %s: %s", symbol, news_rejection)
                        continue
                    tick = market_data.tick(symbol)
                    bars = market_data.fetch_bars(
                        symbol, settings.bars_to_fetch, closed_only=True
                    )
                    signal = strategy.evaluate(symbol, bars)
                    if signal is None:
                        continue
                    if last_signal_time.get(symbol) == signal.timestamp:
                        continue
                    last_signal_time[symbol] = signal.timestamp

                    rejection = risk_manager.entry_rejection(
                        signal,
                        tick,
                        market_data.symbol_info(symbol),
                        executor.open_positions_count(),
                        daily_pnl + (current_account.equity - current_account.balance),
                        day_start_balance,
                    )
                    if rejection:
                        logger.info("%s signal rejected for %s: %s", signal.side, symbol, rejection)
                        continue
                    volume = risk_manager.calculate_volume(
                        signal,
                        current_account.equity,
                        market_data.symbol_info(symbol),
                        market_data.mt5,
                        entry_price=tick.ask if signal.side == "buy" else tick.bid,
                    )
                    if volume is None:
                        logger.warning("%s signal skipped: calculated size is below broker minimum", symbol)
                        continue
                    executor.open_trade(signal, volume, tick)
                time.sleep(settings.poll_seconds)
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Trading cycle failed; retrying after configured poll interval.")
                time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    finally:
        market_data.disconnect()


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logging.getLogger("curry").exception("CURRY failed to start.")
        raise
