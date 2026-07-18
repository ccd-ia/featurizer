# coding: utf-8

"""Sequence φ over the pre-t₀ event series: change-point and periodicity.

Both bridges are **per-entity** (they key by ``fk_col``, not by event row),
so they pair with :meth:`~featurizer.bridge.base.BridgeComputer.materialize_nodes`
for a single snapshot and with the ADR-0014 snapshot loop
(:meth:`~featurizer.bridge.base.BridgeComputer.materialize_snapshots`) for a
backtest cohort — the caller's window slice IS the pre-t₀ guard, re-asserted
per window. Both use ``fit_rows`` (the causal-guarded slice) and ignore the
unsliced ``rows``.

Pure numpy (a hard transitive dependency) — no ``[bridge]`` extra needed.

- :class:`ChangePointBridge` — a binary-segmentation mean-shift statistic
  over a numeric measure series: at every split point compare the two
  segment means, normalized by the overall standard deviation. Emits the
  maximum (``change_score``) and where it occurs as a 0–1 fraction
  (``change_position``). A score of ~0 means "no shift"; a clean step lands
  around 2. This is a scoring heuristic, not a formal hypothesis test.
- :class:`PeriodicityBridge` — FFT peak over the entity's event-count series
  binned on a regular grid: ``period_strength`` is the dominant non-DC
  frequency's share of non-DC power (0–1), ``period_bins`` its period in bin
  units (7 with daily bins and a weekly rhythm).
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import MultiColumnBridge


def _entity_series(
    fit_rows: List[Dict[str, Any]],
    fk_col: str,
    order_col: str,
    value_col: Optional[str],
) -> Dict[Any, List[Tuple[Any, Any]]]:
    """Group ``(order, value)`` pairs per entity, sorted by order."""
    grouped: Dict[Any, List[Tuple[Any, Any]]] = {}
    for row in fit_rows:
        order = row.get(order_col)
        if order is None:
            continue
        value = row.get(value_col) if value_col is not None else None
        grouped.setdefault(row.get(fk_col), []).append((order, value))
    for series in grouped.values():
        series.sort(key=lambda pair: pair[0])
    return grouped


class ChangePointBridge(MultiColumnBridge):
    """Strongest mean shift in an entity's pre-t₀ measure series."""

    VALUE_COLS = ("change_score", "change_position")

    #: Segments shorter than this never anchor a split (guards spurious
    #: single-point "shifts" at the series edges).
    MIN_SEGMENT = 2

    def __init__(
        self,
        *,
        fk_col: str,
        order_col: str,
        measure_col: str,
        name: str = "change_point",
    ) -> None:
        super().__init__(name=name, value_cols=list(self.VALUE_COLS))
        self.fk_col = fk_col
        self.order_col = order_col
        self.measure_col = measure_col

    def _score(self, values: List[float]) -> Dict[str, Optional[float]]:
        n = len(values)
        if n < 2 * self.MIN_SEGMENT:
            return dict.fromkeys(self.VALUE_COLS)
        x = np.asarray(values, dtype=float)
        scale = float(np.std(x))
        if scale == 0.0:  # constant series: no shift, by definition
            return {"change_score": 0.0, "change_position": None}
        best_score, best_k = 0.0, None
        for k in range(self.MIN_SEGMENT, n - self.MIN_SEGMENT + 1):
            score = abs(float(np.mean(x[:k])) - float(np.mean(x[k:]))) / scale
            if score > best_score:
                best_score, best_k = score, k
        return {
            "change_score": best_score,
            "change_position": (best_k / n) if best_k is not None else None,
        }

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Dict[str, Any]]:
        grouped = _entity_series(
            fit_rows, self.fk_col, self.order_col, self.measure_col
        )
        return {
            entity: self._score([float(v) for _, v in series if v is not None])
            for entity, series in grouped.items()
        }


def _order_to_days(value: Any) -> Optional[float]:
    """An event order value as a float day count (dates, datetimes, numbers)."""
    if isinstance(value, datetime.datetime):
        return value.timestamp() / 86400.0
    if isinstance(value, datetime.date):
        return float(value.toordinal())
    if isinstance(value, (int, float)):
        return float(value)
    return None


class PeriodicityBridge(MultiColumnBridge):
    """Dominant cycle in an entity's pre-t₀ event timing (FFT peak)."""

    VALUE_COLS = ("period_strength", "period_bins")

    #: Fewer bins than this cannot support a meaningful spectrum.
    MIN_BINS = 8

    def __init__(
        self,
        *,
        fk_col: str,
        order_col: str,
        bin_days: float = 1.0,
        name: str = "periodicity",
    ) -> None:
        if bin_days <= 0:
            raise ValueError(f"{name}: bin_days must be positive")
        super().__init__(name=name, value_cols=list(self.VALUE_COLS))
        self.fk_col = fk_col
        self.order_col = order_col
        self.bin_days = bin_days

    def _score(self, orders: List[Any]) -> Dict[str, Optional[float]]:
        days = [d for d in (_order_to_days(o) for o in orders) if d is not None]
        if not days:
            return dict.fromkeys(self.VALUE_COLS)
        start = min(days)
        n_bins = int((max(days) - start) / self.bin_days) + 1
        if n_bins < self.MIN_BINS:
            return dict.fromkeys(self.VALUE_COLS)
        counts = np.zeros(n_bins)
        for day in days:
            counts[int((day - start) / self.bin_days)] += 1.0
        power = np.abs(np.fft.rfft(counts - counts.mean())) ** 2
        non_dc = power[1:]
        total = float(non_dc.sum())
        if total == 0.0:  # perfectly uniform series: no cycle
            return {"period_strength": 0.0, "period_bins": None}
        peak = int(np.argmax(non_dc)) + 1
        return {
            "period_strength": float(non_dc[peak - 1]) / total,
            "period_bins": n_bins / peak,
        }

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Dict[str, Any]]:
        grouped = _entity_series(fit_rows, self.fk_col, self.order_col, None)
        return {
            entity: self._score([order for order, _ in series])
            for entity, series in grouped.items()
        }
