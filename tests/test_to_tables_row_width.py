"""The ``to_tables`` heap-row-width pre-flight (v1.0 hardening).

A heap tuple must fit one 8 KiB page (~8160 usable bytes), a bound the
SELECT/fetch paths never hit. A ~1100-column group of fixed-width feature
values is a perfectly valid *query* (≤1664 target-list entries) yet fails
``create table … as`` with ``row is too big``. These tests pin the estimator,
the downshift decision in ``Featurizer._grouped_for_tables``, and the
fail-loud manifest guard — all DB-free; the CTAS proof lives in
``tests/integration/test_to_tables_row_width.py``.
"""

from __future__ import annotations

import tempfile

import pytest
import yaml

from featurizer import Featurizer
from featurizer.sharding import (
    HEAP_ROW_BUDGET_BYTES,
    estimate_heap_row_width,
    max_heap_safe_columns,
)

# ------------------------------------------------------------------ #
# Config helpers
# ------------------------------------------------------------------ #


def _featurizer(config: dict) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path, validate=False)


def _config(n_vars: int, aggregations: list[str], intervals: list[str]) -> dict:
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": intervals,
        "aggregations": aggregations,
        "transformations": ["identity"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {f"v{i}": {"type": "numeric"} for i in range(n_vars)},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }


def _narrow_config() -> dict:
    return _config(1, ["count", "sum", "mean"], ["P1M"])


def _over_budget_config() -> dict:
    """~1100 feature columns sharing one lineage signature: the default
    partition packs them into a single group — a valid single query whose CTAS
    row (~8 bytes × 1100) exceeds the 8160-byte heap page."""
    return _config(
        30,
        ["count", "sum", "mean", "min", "max"],
        ["P1W", "P2W", "P1M", "P3M", "P6M", "P1Y", "P2Y", "P3Y"],
    )


# ------------------------------------------------------------------ #
# The estimator
# ------------------------------------------------------------------ #


def test_estimate_heap_row_width_model():
    """header (24) + null bitmap (1 bit/column, byte-rounded) + 8 bytes/column."""
    assert estimate_heap_row_width(0) == 24
    assert estimate_heap_row_width(1) == 24 + 1 + 8
    assert estimate_heap_row_width(8) == 24 + 1 + 64
    assert estimate_heap_row_width(9) == 24 + 2 + 72
    assert estimate_heap_row_width(1000) == 24 + 125 + 8000


def test_estimate_is_monotonic():
    widths = [estimate_heap_row_width(n) for n in range(0, 2000, 37)]
    assert widths == sorted(widths)
    assert len(set(widths)) == len(widths)


def test_max_heap_safe_columns_is_tight_against_the_budget():
    """The cap is the exact boundary: cap+keys fits the budget, one more does not."""
    for n_keys in (2, 3, 5):
        cap = max_heap_safe_columns(n_keys)
        assert estimate_heap_row_width(cap + n_keys) <= HEAP_ROW_BUDGET_BYTES
        assert estimate_heap_row_width(cap + n_keys + 1) > HEAP_ROW_BUDGET_BYTES


def test_max_heap_safe_columns_shrinks_with_more_keys():
    assert max_heap_safe_columns(5) < max_heap_safe_columns(2)
    # Ballpark sanity: ~8 KiB / 8 bytes ≈ 1000 columns minus overhead.
    assert 900 < max_heap_safe_columns(2) < 1010


# ------------------------------------------------------------------ #
# The downshift decision
# ------------------------------------------------------------------ #


def test_over_budget_group_is_repartitioned_for_tables():
    """A single 1000+-column group downshifts into several heap-safe groups."""
    f = _featurizer(_over_budget_config())

    # Precondition: the default partition would write ONE over-budget table.
    default_groups = f._sharder().column_groups()
    assert len(default_groups) == 1
    n_all = sum(len(cols) for cols in default_groups.values())
    assert estimate_heap_row_width(n_all + 2) > HEAP_ROW_BUDGET_BYTES

    grouped, groups = f._grouped_for_tables()

    assert len(groups) > 1
    n_keys = len(grouped.key_columns)
    for gid, cols in groups.items():
        assert (
            estimate_heap_row_width(len(cols) + n_keys) <= HEAP_ROW_BUDGET_BYTES
        ), f"{gid} still estimates over the heap budget after the downshift"

    # The rendered queries and the mapping describe the same partition.
    assert list(grouped.queries) == list(groups)

    # Column coverage: nothing lost, nothing duplicated.
    all_columns = [c for cols in groups.values() for c in cols]
    assert len(all_columns) == n_all
    assert len(set(all_columns)) == n_all
    assert set(all_columns) == {c for cols in default_groups.values() for c in cols}


def test_narrow_config_keeps_single_query_shortcut():
    """No downshift for a config that fits: one group, byte-identical query."""
    f = _featurizer(_narrow_config())
    grouped, groups = f._grouped_for_tables()
    assert list(groups) == ["group_000"]
    assert list(grouped.queries) == ["group_000"]
    assert grouped.queries["group_000"] == f.query
    assert grouped.fits_single


def test_manifest_covers_every_downshifted_group_column():
    """Every partitioned column has a manifest row and vice versa (the mapping
    the persisted ``feature_group`` tags are built from)."""
    f = _featurizer(_over_budget_config())
    _, groups = f._grouped_for_tables()
    partitioned = {c.replace('"', "") for cols in groups.values() for c in cols}
    manifest = {e.column for e in f.feature_manifest}
    assert partitioned == manifest


# ------------------------------------------------------------------ #
# The manifest fail-loud guard
# ------------------------------------------------------------------ #


class _RecordingCursor:
    """Just enough cursor to run ``_write_manifest_table`` DB-free."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.rows: list[tuple] = []

    def execute(self, sql, params=None):  # noqa: ANN001
        self.statements.append(str(sql))

    def executemany(self, sql, rows):  # noqa: ANN001
        self.statements.append(str(sql))
        self.rows.extend(rows)


def test_manifest_writer_accepts_the_real_partition():
    f = _featurizer(_narrow_config())
    _, groups = f._grouped_for_tables()
    cur = _RecordingCursor()
    f._write_manifest_table(cur, "s", "stem", groups)
    assert len(cur.rows) == len(f.feature_manifest)
    assert all(row[-1] == "group_000" for row in cur.rows)


def test_manifest_writer_raises_on_orphaned_column():
    """A column absent from the partition raises — never a silent mis-tag."""
    from collections import OrderedDict

    f = _featurizer(_narrow_config())
    _, groups = f._grouped_for_tables()
    dropped = groups["group_000"][0]
    truncated = OrderedDict(
        (gid, [c for c in cols if c != dropped]) for gid, cols in groups.items()
    )
    cur = _RecordingCursor()
    with pytest.raises(RuntimeError) as excinfo:
        f._write_manifest_table(cur, "s", "stem", truncated)
    message = str(excinfo.value)
    assert dropped.replace('"', "") in message
    assert "group_000" in message  # the available groups are named
    assert cur.rows == []  # nothing was inserted


def test_manifest_writer_names_all_orphans_up_to_five():
    from collections import OrderedDict

    f = _featurizer(_narrow_config())
    _, groups = f._grouped_for_tables()
    empty = OrderedDict((gid, []) for gid in groups)
    with pytest.raises(RuntimeError, match="map to no column group"):
        f._write_manifest_table(_RecordingCursor(), "s", "stem", empty)
