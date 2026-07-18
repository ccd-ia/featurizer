"""Unit tests for the sequence-extension φ bridges (no database).

Planted signals: a clean step series must dominate an alternating no-shift
series on the change score, and a strict weekly rhythm in daily bins must put
the FFT peak at period 7. Both bridges key by entity and read only the
causal-guarded ``fit_rows`` — proven by feeding a polluted ``rows`` argument.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from featurizer.bridge import ChangePointBridge, PeriodicityBridge


def _series_rows(values, owner="a"):
    return [{"owner": owner, "ts": i, "value": v} for i, v in enumerate(values)]


# --------------------------------------------------------------------------- #
# Change point
# --------------------------------------------------------------------------- #


def _cp() -> ChangePointBridge:
    return ChangePointBridge(fk_col="owner", order_col="ts", measure_col="value")


def test_planted_step_is_recovered():
    rows = _series_rows([0.0] * 10 + [1.0] * 10)
    phi = _cp().compute([], fit_rows=rows)
    # Half zeros, half ones: std = 0.5, mean shift 1.0 -> score 2, mid split.
    assert phi["a"]["change_score"] == pytest.approx(2.0)
    assert phi["a"]["change_position"] == pytest.approx(0.5)


def test_no_shift_scores_low():
    step = _cp().compute([], fit_rows=_series_rows([0.0] * 10 + [1.0] * 10))
    flat = _cp().compute([], fit_rows=_series_rows([0.0, 1.0] * 10))
    assert step["a"]["change_score"] > 5 * flat["a"]["change_score"]


def test_constant_and_short_series_are_degenerate():
    phi = _cp().compute([], fit_rows=_series_rows([3.0] * 8))
    assert phi["a"]["change_score"] == 0.0
    assert phi["a"]["change_position"] is None
    short = _cp().compute([], fit_rows=_series_rows([1.0, 2.0, 3.0]))
    assert short["a"] == {"change_score": None, "change_position": None}


def test_change_point_reads_only_fit_rows():
    """The unsliced ``rows`` must be ignored — the snapshot loop passes the
    causal window through ``fit_rows``."""
    polluted = _series_rows([0.0] * 10 + [100.0] * 10)
    clean = _series_rows([0.0] * 8)
    phi = _cp().compute(polluted, fit_rows=clean)
    assert phi["a"]["change_score"] == 0.0


def test_entities_are_scored_independently():
    rows = _series_rows([0.0] * 10 + [1.0] * 10, owner="a") + _series_rows(
        [5.0] * 8, owner="b"
    )
    phi = _cp().compute([], fit_rows=rows)
    assert phi["a"]["change_score"] == pytest.approx(2.0)
    assert phi["b"]["change_score"] == 0.0


# --------------------------------------------------------------------------- #
# Periodicity
# --------------------------------------------------------------------------- #


def _weekly_rows(weeks: int, owner="a"):
    start = date(2020, 1, 6)  # a Monday
    return [{"owner": owner, "ts": start + timedelta(weeks=w)} for w in range(weeks)]


def test_weekly_rhythm_peaks_at_period_seven():
    bridge = PeriodicityBridge(fk_col="owner", order_col="ts", bin_days=1.0)
    phi = bridge.compute([], fit_rows=_weekly_rows(8))
    # 50 daily bins, one event every 7th: dominant period 7 bins (within the
    # resolution of a 50-bin spectrum), carrying a dominant share of power.
    assert phi["a"]["period_bins"] == pytest.approx(7.0, rel=0.2)
    assert phi["a"]["period_strength"] > 0.2


def test_too_few_bins_is_null_and_bin_days_validated():
    bridge = PeriodicityBridge(fk_col="owner", order_col="ts", bin_days=1.0)
    phi = bridge.compute(
        [], fit_rows=[{"owner": "a", "ts": date(2020, 1, d)} for d in (1, 3)]
    )
    assert phi["a"] == {"period_strength": None, "period_bins": None}
    with pytest.raises(ValueError, match="bin_days"):
        PeriodicityBridge(fk_col="owner", order_col="ts", bin_days=0)


def test_numeric_orders_are_supported():
    """Integer day numbers work like dates (the snapshot loop's slices may
    carry either)."""
    bridge = PeriodicityBridge(fk_col="owner", order_col="ts", bin_days=1.0)
    phi = bridge.compute([], fit_rows=[{"owner": "a", "ts": 7 * w} for w in range(8)])
    assert phi["a"]["period_bins"] == pytest.approx(7.0, rel=0.2)


def test_snapshot_loop_windows_the_series():
    """compute_snapshots slices per as-of window: the step lives in the later
    window only."""
    rows = [{"owner": "a", "ts": t, "value": 0.0 if t < 10 else 5.0} for t in range(20)]
    bridge = ChangePointBridge(fk_col="owner", order_col="ts", measure_col="value")
    snaps = bridge.compute_snapshots(rows, as_of_dates=[9, 19], causal_col="ts")
    assert snaps[("a", 9)]["change_score"] == 0.0  # pre-step window: constant
    assert snaps[("a", 19)]["change_score"] == pytest.approx(2.0)
