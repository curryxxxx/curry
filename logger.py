"""Console/file logging and append-only trade history."""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Any


_history_lock = threading.Lock()
_TRADE_FIELDS = (
    "timestamp",
    "mode",
    "symbol",
    "side",
    "volume",
    "entry",
    "exit",
    "stop_loss",
    "take_profit",
    "profit",
    "reason",
    "ticket",
)


def configure_logging(directory: str) -> logging.Logger:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("curry")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
        file_handler = logging.FileHandler(path / "curry.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def record_trade(directory: str, trade: dict[str, Any]) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    history_path = path / "trade_history.csv"
    with _history_lock:
        file_exists = history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=_TRADE_FIELDS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow({field: trade.get(field, "") for field in _TRADE_FIELDS})
