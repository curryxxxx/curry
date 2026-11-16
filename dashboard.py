"""Minimal terminal status dashboard."""

from __future__ import annotations

from typing import Any

from config import Settings


def render_status(
    settings: Settings,
    account: Any,
    open_trades: int,
    daily_pnl: float,
    symbols: tuple[str, ...],
) -> None:
    print(
        "\n".join(
            [
                "",
                "=" * 58,
                "CURRY | MT5 FOREX AUTOMATION",
                f"Mode: {settings.mode.upper()} | Strategy: {settings.strategy} | "
                f"Timeframe: {settings.timeframe}",
                f"Account: {account.login} | Currency: {account.currency}",
                f"Balance: {account.balance:.2f} | Equity: {account.equity:.2f}",
                f"Open trades: {open_trades}/{settings.max_open_trades}",
                f"Today's realized P&L: {daily_pnl:.2f}",
                f"Symbols: {', '.join(symbols)}",
                "=" * 58,
            ]
        ),
        flush=True,
    )
