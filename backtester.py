"""Bar-by-bar backtest utilities for actual historical MT5 OHLC data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import add_indicators
from strategy import EmaRsiAtrStrategy, TradingStrategy


def run_backtest(
    symbol: str,
    bars: pd.DataFrame,
    strategy: EmaRsiAtrStrategy | TradingStrategy,
    spread_price: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if bars.empty:
        raise ValueError("Backtest requires actual historical OHLC bars.")
    if spread_price < 0:
        raise ValueError("spread_price must be non-negative.")
    data = add_indicators(bars)
    trades: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None

    for index in range(50, len(data)):
        bar = data.iloc[index]
        timestamp = data.index[index]
        if position is not None:
            exit_price: float | None = None
            reason = ""
            if position["side"] == "buy":
                if bar["low"] <= position["stop_loss"]:
                    exit_price, reason = position["stop_loss"], "stop_loss"
                elif bar["high"] >= position["take_profit"]:
                    exit_price, reason = position["take_profit"], "take_profit"
            else:
                if bar["high"] >= position["stop_loss"]:
                    exit_price, reason = position["stop_loss"], "stop_loss"
                elif bar["low"] <= position["take_profit"]:
                    exit_price, reason = position["take_profit"], "take_profit"
            if exit_price is not None:
                signed_change = (
                    exit_price - position["entry"]
                    if position["side"] == "buy"
                    else position["entry"] - exit_price
                )
                trades.append(
                    {
                        "symbol": symbol,
                        "side": position["side"],
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "entry": position["entry"],
                        "exit": exit_price,
                        "pnl_price": signed_change - spread_price,
                        "reason": reason,
                    }
                )
                position = None
            if position is not None:
                continue

        signal = strategy.evaluate(symbol, data.iloc[: index + 1])
        if signal is None:
            continue
        if index + 1 >= len(data):
            continue
        next_bar = data.iloc[index + 1]
        entry = float(next_bar["open"])
        if signal.side == "buy":
            position = {
                "side": "buy",
                "entry": entry + spread_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "entry_time": data.index[index + 1],
            }
        else:
            position = {
                "side": "sell",
                "entry": entry,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "entry_time": data.index[index + 1],
            }

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        summary = {"trades": 0.0, "net_pnl_price": 0.0, "win_rate_pct": 0.0}
    else:
        summary = {
            "trades": float(len(trade_frame)),
            "net_pnl_price": float(trade_frame["pnl_price"].sum()),
            "win_rate_pct": float((trade_frame["pnl_price"] > 0).mean() * 100),
        }
    return trade_frame, summary
