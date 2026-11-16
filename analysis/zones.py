"""Causality-aware support/resistance and supply/demand zone detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from analysis.structure import SwingPoint, find_swings

ZoneKind = Literal["SUPPORT", "RESISTANCE", "DEMAND", "SUPPLY"]
ZoneStatus = Literal["FRESH", "TESTED", "INVALIDATED"]

_OHLC = {"open", "high", "low", "close"}


@dataclass(frozen=True)
class SupportResistanceZone:
    kind: Literal["SUPPORT", "RESISTANCE"]
    lower: float
    upper: float
    center: float
    touch_count: int
    strength_score: float
    known_at: int
    swing_indices: tuple[int, ...]


@dataclass(frozen=True)
class SupplyDemandZone:
    kind: Literal["DEMAND", "SUPPLY"]
    lower: float
    upper: float
    base_start: int
    base_end: int
    displacement_index: int
    known_at: int
    status_known_at: int
    status: ZoneStatus
    displacement_atr: float


@dataclass
class _Cluster:
    kind: Literal["SUPPORT", "RESISTANCE"]
    prices: list[float]
    tolerances: list[float]
    swing_indices: list[int]
    known_at: int

    @property
    def center(self) -> float:
        return float(np.mean(self.prices))

    @property
    def tolerance(self) -> float:
        return float(np.mean(self.tolerances))


def _validate_bars(bars: pd.DataFrame) -> None:
    missing = _OHLC.difference(bars.columns)
    if missing:
        raise ValueError(f"OHLC data is missing columns: {', '.join(sorted(missing))}")
    if bars[list(_OHLC)].isna().any().any():
        raise ValueError("OHLC values must not contain NaN.")
    if (bars["high"] < bars["low"]).any():
        raise ValueError("Each bar's high must be greater than or equal to its low.")


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


def build_support_resistance_zones(
    bars: pd.DataFrame,
    n: int = 3,
    tolerance_atr: float = 0.5,
    swings: list[SwingPoint] | None = None,
) -> list[SupportResistanceZone]:
    """Cluster confirmed swing points; scores use only the cluster's known touches.

    The 0-100 score is 70% touch-count (saturating at five touches) and 30%
    price tightness (the cluster's level spread relative to its ATR tolerance).
    """
    _validate_bars(bars)
    if not np.isfinite(tolerance_atr) or tolerance_atr <= 0:
        raise ValueError("tolerance_atr must be positive.")
    atr = _atr_values(bars)
    points = find_swings(bars, n) if swings is None else list(swings)
    last_index = len(bars) - 1
    for point in points:
        if (
            point.kind not in ("HIGH", "LOW")
            or point.index < 0
            or point.known_at < point.index
            or point.known_at > last_index
            or not np.isfinite(point.price)
        ):
            raise ValueError("A supplied swing is invalid or not confirmed by the provided bars.")

    clusters: list[_Cluster] = []
    for point in sorted(points, key=lambda swing: (swing.known_at, swing.index)):
        tolerance = float(atr[point.known_at]) * tolerance_atr
        if tolerance <= 0:
            continue
        kind: Literal["SUPPORT", "RESISTANCE"] = (
            "SUPPORT" if point.kind == "LOW" else "RESISTANCE"
        )
        matching = [
            cluster
            for cluster in clusters
            if cluster.kind == kind
            and abs(point.price - cluster.center) <= max(tolerance, cluster.tolerance)
        ]
        if matching:
            cluster = min(matching, key=lambda item: abs(point.price - item.center))
            cluster.prices.append(point.price)
            cluster.tolerances.append(tolerance)
            cluster.swing_indices.append(point.index)
            cluster.known_at = max(cluster.known_at, point.known_at)
        else:
            clusters.append(
                _Cluster(kind, [point.price], [tolerance], [point.index], point.known_at)
            )

    zones: list[SupportResistanceZone] = []
    for cluster in clusters:
        center = cluster.center
        tolerance = cluster.tolerance
        spread = max(cluster.prices) - min(cluster.prices)
        touch_component = min(len(cluster.prices) / 5, 1.0) * 70
        precision_component = max(0.0, 1.0 - spread / (2 * tolerance)) * 30
        zones.append(
            SupportResistanceZone(
                kind=cluster.kind,
                lower=min(cluster.prices) - tolerance / 2,
                upper=max(cluster.prices) + tolerance / 2,
                center=center,
                touch_count=len(cluster.prices),
                strength_score=round(touch_component + precision_component, 2),
                known_at=cluster.known_at,
                swing_indices=tuple(cluster.swing_indices),
            )
        )
    return sorted(zones, key=lambda zone: (zone.known_at, zone.center))


def build_supply_demand_zones(
    bars: pd.DataFrame,
    displacement_atr: float = 1.5,
    max_base_candles: int = 3,
    base_range_atr: float = 1.0,
) -> list[SupplyDemandZone]:
    """Detect compact bases preceding ATR-sized displacement candles.

    ``known_at`` is the displacement candle's close, when zone geometry first
    becomes knowable. ``status_known_at`` records the later bar (if any) where
    FRESH changes to TESTED or INVALIDATED; status never changes the origin time.
    """
    _validate_bars(bars)
    if not np.isfinite(displacement_atr) or displacement_atr <= 0:
        raise ValueError("displacement_atr must be positive.")
    if (
        isinstance(max_base_candles, bool)
        or not isinstance(max_base_candles, int)
        or max_base_candles < 1
    ):
        raise ValueError("max_base_candles must be a positive integer.")
    if not np.isfinite(base_range_atr) or base_range_atr <= 0:
        raise ValueError("base_range_atr must be positive.")

    atr = _atr_values(bars)
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    candle_ranges = high - low
    zones: list[SupplyDemandZone] = []

    for displacement_index in range(1, len(bars)):
        reference_atr = atr[displacement_index - 1]
        if reference_atr <= 0 or candle_ranges[displacement_index] < displacement_atr * reference_atr:
            continue

        base_start = displacement_index - 1
        earliest = max(0, displacement_index - max_base_candles)
        while base_start > earliest:
            candidate = base_start - 1
            if candle_ranges[candidate] <= atr[candidate] * base_range_atr:
                base_start = candidate
            else:
                break
        if candle_ranges[base_start] > atr[base_start] * base_range_atr:
            continue

        zone_low = float(np.min(low[base_start:displacement_index]))
        zone_high = float(np.max(high[base_start:displacement_index]))
        if close[displacement_index] > open_[displacement_index] and close[displacement_index] > zone_high:
            kind: Literal["DEMAND", "SUPPLY"] = "DEMAND"
        elif close[displacement_index] < open_[displacement_index] and close[displacement_index] < zone_low:
            kind = "SUPPLY"
        else:
            continue

        status: ZoneStatus = "FRESH"
        status_known_at = displacement_index
        for later_index in range(displacement_index + 1, len(bars)):
            invalidated = (
                close[later_index] < zone_low
                if kind == "DEMAND"
                else close[later_index] > zone_high
            )
            if invalidated:
                status = "INVALIDATED"
                status_known_at = later_index
                break
            touched = low[later_index] <= zone_high and high[later_index] >= zone_low
            if touched and status == "FRESH":
                status = "TESTED"
                status_known_at = later_index

        zones.append(
            SupplyDemandZone(
                kind=kind,
                lower=zone_low,
                upper=zone_high,
                base_start=base_start,
                base_end=displacement_index - 1,
                displacement_index=displacement_index,
                known_at=displacement_index,
                status_known_at=status_known_at,
                status=status,
                displacement_atr=float(candle_ranges[displacement_index] / reference_atr),
            )
        )
    return zones
