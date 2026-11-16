"""Non-repainting fractal swings and market-structure trend classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

SwingKind = Literal["HIGH", "LOW"]
SwingClassification = Literal["HH", "HL", "LH", "LL"]
TrendState = Literal["BULLISH", "BEARISH", "RANGE"]

_OHLC = {"high", "low"}


@dataclass(frozen=True)
class SwingPoint:
    """A confirmed pivot; indices are zero-based row positions in the input."""

    index: int
    known_at: int
    price: float
    kind: SwingKind
    classification: SwingClassification | None


@dataclass(frozen=True)
class TrendPoint:
    """Trend state observable after the close of ``known_at``."""

    known_at: int
    state: TrendState


@dataclass(frozen=True)
class StructureAnalysis:
    """Confirmed swings and a causal, per-bar trend-state history."""

    known_at: int
    swings: tuple[SwingPoint, ...]
    trend_history: tuple[TrendPoint, ...]

    @property
    def trend(self) -> TrendState:
        return self.trend_history[-1].state if self.trend_history else "RANGE"


def _validate_bars(bars: pd.DataFrame) -> None:
    missing = _OHLC.difference(bars.columns)
    if missing:
        raise ValueError(f"OHLC data is missing columns: {', '.join(sorted(missing))}")
    if bars[["high", "low"]].isna().any().any():
        raise ValueError("High/low values must not contain NaN.")
    if (bars["high"] < bars["low"]).any():
        raise ValueError("Each bar's high must be greater than or equal to its low.")


def _validate_lookback(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Fractal lookback n must be a positive integer.")


def find_swings(bars: pd.DataFrame, n: int = 3) -> list[SwingPoint]:
    """Find strict N-left/N-right fractals, exposing them only at confirmation.

    A pivot at row ``i`` is knowable at ``i + n``. The pivot's right-hand
    comparison therefore never reads a bar later than its ``known_at`` index.
    """
    _validate_bars(bars)
    _validate_lookback(n)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    raw: list[tuple[int, int, float, SwingKind]] = []

    for index in range(n, len(bars) - n):
        known_at = index + n
        high_neighbors = highs[index - n : index].tolist() + highs[index + 1 : known_at + 1].tolist()
        low_neighbors = lows[index - n : index].tolist() + lows[index + 1 : known_at + 1].tolist()
        if all(highs[index] > value for value in high_neighbors):
            raw.append((known_at, index, float(highs[index]), "HIGH"))
        if all(lows[index] < value for value in low_neighbors):
            raw.append((known_at, index, float(lows[index]), "LOW"))

    raw.sort(key=lambda item: (item[0], item[1], 0 if item[3] == "HIGH" else 1))
    previous: dict[SwingKind, float] = {}
    swings: list[SwingPoint] = []
    for known_at, index, price, kind in raw:
        prior_price = previous.get(kind)
        classification: SwingClassification | None = None
        if prior_price is not None:
            if kind == "HIGH" and price != prior_price:
                classification = "HH" if price > prior_price else "LH"
            elif kind == "LOW" and price != prior_price:
                classification = "HL" if price > prior_price else "LL"
        previous[kind] = price
        swings.append(SwingPoint(index, known_at, price, kind, classification))
    return swings


def analyze_structure(bars: pd.DataFrame, n: int = 3) -> StructureAnalysis:
    """Return confirmed swings and a trend state for each observable bar."""
    swings = find_swings(bars, n)
    by_confirmation: dict[int, list[SwingPoint]] = {}
    for swing in swings:
        by_confirmation.setdefault(swing.known_at, []).append(swing)

    latest_high: SwingClassification | None = None
    latest_low: SwingClassification | None = None
    trend_history: list[TrendPoint] = []
    for bar_index in range(len(bars)):
        for swing in by_confirmation.get(bar_index, []):
            if swing.kind == "HIGH":
                latest_high = swing.classification
            else:
                latest_low = swing.classification

        if latest_high == "HH" and latest_low == "HL":
            state: TrendState = "BULLISH"
        elif latest_high == "LH" and latest_low == "LL":
            state = "BEARISH"
        else:
            state = "RANGE"
        trend_history.append(TrendPoint(known_at=bar_index, state=state))

    return StructureAnalysis(
        known_at=len(bars) - 1,
        swings=tuple(swings),
        trend_history=tuple(trend_history),
    )
