"""Causality-aware Smart Money Concepts pattern detection.

All indices are zero-based row positions. Events and state-transition snapshots are only emitted when knowable at their
``known_at`` close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

from analysis.structure import (
    StructureAnalysis,
    SwingPoint,
    analyze_structure,
)

Direction = Literal["BULLISH", "BEARISH"]
BreakKind = Literal["BOS", "CHoCH"]
FVGStatus = Literal["OPEN", "PARTIAL", "FILLED"]
LiquidityKind = Literal["EQUAL_HIGH", "EQUAL_LOW"]
SweepSource = Literal["SWING", "EQUAL_HIGH", "EQUAL_LOW"]

_OHLC = {"open", "high", "low", "close"}


@dataclass(frozen=True)
class StructureBreak:
    kind: BreakKind
    direction: Direction
    level: float
    swing_index: int
    known_at: int


@dataclass(frozen=True)
class FairValueGap:
    direction: Direction
    lower: float
    upper: float
    formed_at: int
    known_at: int
    status: FVGStatus


@dataclass(frozen=True)
class OrderBlock:
    direction: Direction
    lower: float
    upper: float
    candle_index: int
    displacement_index: int
    bos_swing_index: int
    known_at: int
    mitigated: bool


@dataclass(frozen=True)
class EqualHighLow:
    kind: LiquidityKind
    level: float
    swing_indices: tuple[int, ...]
    known_at: int
    tolerance: float


@dataclass(frozen=True)
class LiquiditySweep:
    direction: Direction
    source: SweepSource
    level: float
    reference_indices: tuple[int, ...]
    sweep_index: int
    known_at: int


@dataclass(frozen=True)
class SMCAnalysis:
    known_at: int
    structure: StructureAnalysis
    breaks: tuple[StructureBreak, ...]
    fair_value_gaps: tuple[FairValueGap, ...]
    order_blocks: tuple[OrderBlock, ...]
    equal_levels: tuple[EqualHighLow, ...]
    liquidity_sweeps: tuple[LiquiditySweep, ...]


@dataclass
class _EqualCluster:
    kind: LiquidityKind
    points: list[SwingPoint]
    known_at: int

    @property
    def level(self) -> float:
        return float(np.mean([point.price for point in self.points]))


@dataclass(frozen=True)
class _LiquidityReference:
    source: SweepSource
    direction: Direction
    level: float
    indices: tuple[int, ...]
    known_at: int


def _validate_bars(bars: pd.DataFrame) -> None:
    missing = _OHLC.difference(bars.columns)
    if missing:
        raise ValueError(f"OHLC data is missing columns: {', '.join(sorted(missing))}")
    values = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OHLC values must be finite.")
    if (bars["high"] < bars["low"]).any():
        raise ValueError("Each bar's high must be greater than or equal to its low.")
    if (
        (bars["open"] < bars["low"]).any()
        or (bars["open"] > bars["high"]).any()
        or (bars["close"] < bars["low"]).any()
        or (bars["close"] > bars["high"]).any()
    ):
        raise ValueError("Each bar's open and close must lie within its high/low range.")


def _atr_values(bars: pd.DataFrame) -> np.ndarray:
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    previous_close = np.roll(close, 1)
    if len(previous_close):
        previous_close[0] = close[0]
    true_range = np.maximum.reduce(
        (high - low, np.abs(high - previous_close), np.abs(low - previous_close))
    )
    return (
        pd.Series(true_range, index=bars.index)
        .ewm(alpha=1 / 14, min_periods=1, adjust=False)
        .mean()
        .to_numpy(dtype=float)
    )


def _validate_swings(points: list[SwingPoint], bar_count: int) -> None:
    for point in points:
        if (
            point.kind not in ("HIGH", "LOW")
            or point.index < 0
            or point.index >= bar_count
            or point.known_at < point.index
            or point.known_at < 0
            or point.known_at >= bar_count
            or not np.isfinite(point.price)
        ):
            raise ValueError("A supplied swing is invalid or not confirmed by the bars.")


def detect_structure_breaks(
    bars: pd.DataFrame,
    n: int = 3,
    structure: StructureAnalysis | None = None,
) -> list[StructureBreak]:
    """Detect close-to-close BOS/CHoCH crosses using confirmed swings.

    BULLISH trend: BOS crosses above the last confirmed high; CHoCH crosses
    below the last confirmed low. BEARISH trend uses the inverse levels.
    RANGE produces neither. Equality at the prior close is permitted; the
    current close must be strictly beyond the level, so wick-only breaks fail.
    """
    _validate_bars(bars)
    analysis = structure or analyze_structure(bars, n)
    if len(analysis.trend_history) != len(bars):
        raise ValueError("Structure analysis must correspond to the supplied bars.")
    _validate_swings(list(analysis.swings), len(bars))

    swings_by_kind: dict[str, list[SwingPoint]] = {"HIGH": [], "LOW": []}
    for swing in analysis.swings:
        swings_by_kind[swing.kind].append(swing)

    closes = bars["close"].to_numpy(dtype=float)
    events: list[StructureBreak] = []
    for index in range(1, len(bars)):
        trend = analysis.trend_history[index - 1].state
        if trend == "RANGE":
            continue
        checks: tuple[
            tuple[
                BreakKind,
                Literal["HIGH", "LOW"],
                Direction,
                Callable[[float, float, float], bool],
            ],
            ...,
        ]
        if trend == "BULLISH":
            checks = (
                ("BOS", "HIGH", "BULLISH", lambda prior, current, level: prior <= level < current),
                ("CHoCH", "LOW", "BEARISH", lambda prior, current, level: prior >= level > current),
            )
        else:
            checks = (
                ("BOS", "LOW", "BEARISH", lambda prior, current, level: prior >= level > current),
                ("CHoCH", "HIGH", "BULLISH", lambda prior, current, level: prior <= level < current),
            )

        for kind, swing_kind, direction, crossed in checks:
            confirmed = [swing for swing in swings_by_kind[swing_kind] if swing.known_at < index]
            if not confirmed:
                continue
            swing = confirmed[-1]
            if crossed(closes[index - 1], closes[index], swing.price):
                events.append(
                    StructureBreak(
                        kind=kind,
                        direction=direction,
                        level=swing.price,
                        swing_index=swing.index,
                        known_at=index,
                    )
                )
    return events


def detect_fair_value_gaps(
    bars: pd.DataFrame,
    minimum_atr: float = 0.3,
) -> list[FairValueGap]:
    """Detect three-candle imbalances and track later penetration.

    The gap must be at least ``minimum_atr`` times ATR known before candle 3.
    Returns a snapshot at formation and another at each later status change.
    PARTIAL means price entered the gap; FILLED means it reached the far edge.
    """
    _validate_bars(bars)
    if not np.isfinite(minimum_atr) or minimum_atr <= 0:
        raise ValueError("minimum_atr must be positive.")
    atr = _atr_values(bars)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    gaps: list[FairValueGap] = []

    for index in range(2, len(bars)):
        reference_atr = atr[index - 1]
        if reference_atr <= 0:
            continue
        if high[index - 2] < low[index]:
            lower, upper = float(high[index - 2]), float(low[index])
            direction: Direction = "BULLISH"
        elif low[index - 2] > high[index]:
            lower, upper = float(high[index]), float(low[index - 2])
            direction = "BEARISH"
        else:
            continue
        if upper - lower < minimum_atr * reference_atr:
            continue

        snapshots = [
            FairValueGap(
                direction=direction,
                lower=lower,
                upper=upper,
                formed_at=index,
                known_at=index,
                status="OPEN",
            )
        ]
        status: FVGStatus = "OPEN"
        for later in range(index + 1, len(bars)):
            if direction == "BULLISH":
                filled = low[later] <= lower
                entered = low[later] < upper and high[later] > lower
            else:
                filled = high[later] >= upper
                entered = high[later] > lower and low[later] < upper
            if filled:
                status = "FILLED"
                snapshots.append(
                    FairValueGap(
                        direction=direction,
                        lower=lower,
                        upper=upper,
                        formed_at=index,
                        known_at=later,
                        status=status,
                    )
                )
                break
            if entered and status == "OPEN":
                status = "PARTIAL"
                snapshots.append(
                    FairValueGap(
                        direction=direction,
                        lower=lower,
                        upper=upper,
                        formed_at=index,
                        known_at=later,
                        status=status,
                    )
                )
        gaps.extend(snapshots)
    return sorted(gaps, key=lambda gap: (gap.known_at, gap.formed_at))


def detect_order_blocks(
    bars: pd.DataFrame,
    n: int = 3,
    displacement_atr: float = 1.0,
    breaks: list[StructureBreak] | None = None,
) -> list[OrderBlock]:
    """Find the nearest opposite candle before an ATR-body BOS displacement.

    Displacement body size is measured against ATR from the preceding bar.
    The block is the opposite candle's full high/low range. Returns an
    unmitigated snapshot at BOS and another snapshot if a later close passes
    through the far edge.
    """
    _validate_bars(bars)
    if not np.isfinite(displacement_atr) or displacement_atr <= 0:
        raise ValueError("displacement_atr must be positive.")
    bos_events = breaks
    if bos_events is None:
        bos_events = detect_structure_breaks(bars, n)
    if any(event.known_at < 0 or event.known_at >= len(bars) for event in bos_events):
        raise ValueError("A supplied structure break is outside the provided bars.")
    atr = _atr_values(bars)
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    blocks: list[OrderBlock] = []

    for event in bos_events:
        index = event.known_at
        if event.kind != "BOS" or index < 1:
            continue
        body = abs(close[index] - open_[index])
        if body < displacement_atr * atr[index - 1]:
            continue
        if event.direction == "BULLISH" and close[index] <= open_[index]:
            continue
        if event.direction == "BEARISH" and close[index] >= open_[index]:
            continue

        opposite = next(
            (
                candle_index
                for candle_index in range(index - 1, -1, -1)
                if (
                    close[candle_index] < open_[candle_index]
                    if event.direction == "BULLISH"
                    else close[candle_index] > open_[candle_index]
                )
            ),
            None,
        )
        if opposite is None:
            continue

        snapshots = [
            OrderBlock(
                direction=event.direction,
                lower=float(low[opposite]),
                upper=float(high[opposite]),
                candle_index=opposite,
                displacement_index=index,
                bos_swing_index=event.swing_index,
                known_at=index,
                mitigated=False,
            )
        ]
        for later in range(index + 1, len(bars)):
            crossed_through = (
                close[later] < low[opposite]
                if event.direction == "BULLISH"
                else close[later] > high[opposite]
            )
            if crossed_through:
                snapshots.append(
                    OrderBlock(
                        direction=event.direction,
                        lower=float(low[opposite]),
                        upper=float(high[opposite]),
                        candle_index=opposite,
                        displacement_index=index,
                        bos_swing_index=event.swing_index,
                        known_at=later,
                        mitigated=True,
                    )
                )
                break
        blocks.extend(snapshots)
    return sorted(blocks, key=lambda block: (block.known_at, block.displacement_index))


def detect_equal_highs_lows(
    bars: pd.DataFrame,
    n: int = 3,
    tolerance_atr: float = 0.1,
    swings: tuple[SwingPoint, ...] | list[SwingPoint] | None = None,
) -> list[EqualHighLow]:
    """Return a cluster snapshot whenever a second or later swing joins it.

    Each new swing is compared with the cluster using ``tolerance_atr * ATR``
    at that swing's confirmation bar. Updated snapshots retain their own
    ``known_at``; later cluster members cannot be used before confirmation.
    """
    _validate_bars(bars)
    if not np.isfinite(tolerance_atr) or tolerance_atr <= 0:
        raise ValueError("tolerance_atr must be positive.")
    analysis = analyze_structure(bars, n) if swings is None else None
    points = list(analysis.swings if analysis is not None else swings or ())
    _validate_swings(points, len(bars))
    atr = _atr_values(bars)
    clusters: list[_EqualCluster] = []
    snapshots: list[EqualHighLow] = []

    for point in sorted(points, key=lambda swing: (swing.known_at, swing.index)):
        tolerance = float(atr[point.known_at]) * tolerance_atr
        if tolerance <= 0:
            continue
        kind: LiquidityKind = "EQUAL_HIGH" if point.kind == "HIGH" else "EQUAL_LOW"
        eligible = [
            cluster
            for cluster in clusters
            if cluster.kind == kind
            and max([entry.price for entry in cluster.points] + [point.price])
            - min([entry.price for entry in cluster.points] + [point.price])
            <= tolerance
        ]
        if eligible:
            cluster = min(eligible, key=lambda item: abs(point.price - item.level))
            cluster.points.append(point)
            cluster.known_at = point.known_at
        else:
            cluster = _EqualCluster(kind, [point], point.known_at)
            clusters.append(cluster)

        if len(cluster.points) >= 2:
            snapshots.append(
                EqualHighLow(
                    kind=kind,
                    level=cluster.level,
                    swing_indices=tuple(entry.index for entry in cluster.points),
                    known_at=point.known_at,
                    tolerance=tolerance,
                )
            )
    return snapshots


def detect_liquidity_sweeps(
    bars: pd.DataFrame,
    n: int = 3,
    confirmation_bars: int = 3,
    equal_levels: list[EqualHighLow] | None = None,
    swings: tuple[SwingPoint, ...] | list[SwingPoint] | None = None,
) -> list[LiquiditySweep]:
    """Detect a wick through known liquidity followed by a later close inside.

    Sweep bars must close at or beyond the level. A return close must occur on
    one of the next 1-3 bars (configurable); upper sweeps are bearish and lower
    sweeps bullish. A return on the wick bar itself is deliberately not enough.
    """
    _validate_bars(bars)
    if (
        isinstance(confirmation_bars, bool)
        or not isinstance(confirmation_bars, int)
        or confirmation_bars < 1
        or confirmation_bars > 3
    ):
        raise ValueError("confirmation_bars must be between 1 and 3.")
    analysis = analyze_structure(bars, n) if swings is None else None
    points = list(analysis.swings if analysis is not None else swings or ())
    _validate_swings(points, len(bars))
    levels = (
        detect_equal_highs_lows(bars, n, swings=points)
        if equal_levels is None
        else equal_levels
    )
    if any(
        level.kind not in ("EQUAL_HIGH", "EQUAL_LOW")
        or level.known_at < 0
        or level.known_at >= len(bars)
        or len(level.swing_indices) < 2
        or not np.isfinite(level.level)
        for level in levels
    ):
        raise ValueError("A supplied equal-level snapshot is invalid for the provided bars.")
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    events: list[LiquiditySweep] = []
    seen: set[tuple[Direction, int, int, str, float]] = set()
    for sweep_index in range(len(bars)):
        snapshots_by_cluster: dict[tuple[LiquidityKind, int], EqualHighLow] = {}
        for level in levels:
            if level.known_at >= sweep_index or len(level.swing_indices) < 2:
                continue
            key = (level.kind, level.swing_indices[0])
            previous = snapshots_by_cluster.get(key)
            if previous is None or level.known_at > previous.known_at:
                snapshots_by_cluster[key] = level
        active_equal = list(snapshots_by_cluster.values())
        clustered_indices = {
            index for level in active_equal for index in level.swing_indices
        }
        references: list[_LiquidityReference] = []
        for point in points:
            if point.known_at >= sweep_index or point.index in clustered_indices:
                continue
            if point.kind == "HIGH":
                references.append(
                    _LiquidityReference(
                        "SWING", "BEARISH", point.price, (point.index,), point.known_at
                    )
                )
            elif point.kind == "LOW":
                references.append(
                    _LiquidityReference(
                        "SWING", "BULLISH", point.price, (point.index,), point.known_at
                    )
                )
        for level in active_equal:
            references.append(
                _LiquidityReference(
                    level.kind,
                    "BEARISH" if level.kind == "EQUAL_HIGH" else "BULLISH",
                    level.level,
                    level.swing_indices,
                    level.known_at,
                )
            )

        for reference in references:
            is_upper = reference.direction == "BEARISH"
            pierced = (
                high[sweep_index] > reference.level
                and close[sweep_index] >= reference.level
                if is_upper
                else low[sweep_index] < reference.level
                and close[sweep_index] <= reference.level
            )
            if not pierced:
                continue
            last_confirmation = min(len(bars) - 1, sweep_index + confirmation_bars)
            for confirmation_index in range(sweep_index + 1, last_confirmation + 1):
                returned_inside = (
                    close[confirmation_index] < reference.level
                    if is_upper
                    else close[confirmation_index] > reference.level
                )
                if not returned_inside:
                    continue
                key = (
                    reference.direction,
                    sweep_index,
                    confirmation_index,
                    reference.source,
                    round(reference.level, 10),
                )
                if key not in seen:
                    seen.add(key)
                    events.append(
                        LiquiditySweep(
                            direction=reference.direction,
                            source=reference.source,
                            level=reference.level,
                            reference_indices=reference.indices,
                            sweep_index=sweep_index,
                            known_at=confirmation_index,
                        )
                    )
                break
    return sorted(events, key=lambda event: (event.known_at, event.sweep_index, event.level))


def analyze_smc(
    bars: pd.DataFrame,
    n: int = 3,
    fvg_minimum_atr: float = 0.3,
    ob_displacement_atr: float = 1.0,
    equal_tolerance_atr: float = 0.1,
    sweep_confirmation_bars: int = 3,
) -> SMCAnalysis:
    """Run all SMC detectors on the supplied OHLC history."""
    _validate_bars(bars)
    structure = analyze_structure(bars, n)
    breaks = detect_structure_breaks(bars, n, structure)
    equal_levels = detect_equal_highs_lows(
        bars, n, equal_tolerance_atr, structure.swings
    )
    return SMCAnalysis(
        known_at=len(bars) - 1,
        structure=structure,
        breaks=tuple(breaks),
        fair_value_gaps=tuple(detect_fair_value_gaps(bars, fvg_minimum_atr)),
        order_blocks=tuple(
            detect_order_blocks(bars, n, ob_displacement_atr, breaks)
        ),
        equal_levels=tuple(equal_levels),
        liquidity_sweeps=tuple(
            detect_liquidity_sweeps(
                bars,
                n,
                sweep_confirmation_bars,
                equal_levels,
                structure.swings,
            )
        ),
    )
