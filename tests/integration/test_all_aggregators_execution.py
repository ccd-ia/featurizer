"""Execute EVERY registered aggregator on real PostgreSQL over edge-case data.

The advanced-aggregator tier (everything beyond the curated default-active set)
was, for years, only string-shape tested — no test executed its generated SQL —
so it accumulated runtime bugs (division-by-zero, ``STDDEV(interval)``, nested
aggregates, unbalanced SQL) that string tests structurally cannot catch. This
harness closes that gap: it drives every aggregator the registry exposes against
fixtures engineered to trigger the failure modes, so "every registered
aggregator executes without error" is a tested invariant.

Fixtures (child groups):
  pid=1  single-row group          (stddev / gap / skew undefined -> NULL)
  pid=2  constant value            (var_pop = 0 -> guarded ratios -> NULL)
  pid=3  contains 0 and a negative (1/x, ln(x) domain edges)
  pid=4  normal multi-row group
  pid=5  values averaging exactly 0 (mean-in-denominator edges)
  pid=6  a single repeated category (sequence/distribution edges)
Run against BOTH a ``date`` and a ``timestamp`` temporal column.

Predicate-driven (first_passage_time, cross_type_latency) and spatial
aggregators do not fire without their special config; they are recorded as
explicitly-skipped so there are no silent gaps.
"""

from __future__ import annotations

import pytest

from featurizer import Featurizer
from featurizer.primitives.utils import list_aggregations

from ._harness import create_temp_table, run_featurizer

pytestmark = pytest.mark.integration

_ROWS = [
    (1, "2023-01-01", 5.0, "a"),
    (2, "2023-02-01", 3.0, "x"),
    (2, "2023-02-02", 3.0, "x"),
    (3, "2023-03-01", 0.0, "p"),
    (3, "2023-03-05", -2.0, "q"),
    (4, "2023-04-01", 1.0, "m"),
    (4, "2023-04-03", 4.0, "n"),
    (4, "2023-04-06", 9.0, "m"),
    (4, "2023-04-10", 16.0, "n"),
    (5, "2023-05-01", -4.0, "s"),
    (5, "2023-05-02", 4.0, "s"),
    (6, "2023-06-01", 2.0, "z"),
    (6, "2023-06-02", 2.0, "z"),
    (6, "2023-06-03", 2.0, "z"),
]
_ALL_AGGS = sorted(list_aggregations())
# Fire only with a special config (predicates / spatial_ix) — not exercised here,
# but recorded so the coverage test proves there are no silent gaps.
_NEEDS_SPECIAL_CONFIG = {
    # boolean (no boolean column in the fixture)
    "all",
    "any",
    # spatial (need a spatial_ix)
    "bbox_area",
    "distance_travelled",
    "radius_of_gyration",
    "spatial_std",
    # predicate-driven (need role/predicate config)
    "first_passage_time",
    "cross_type_latency",
    "right_censoring_indicator",
    # distribution-drift (need a two-window / reference config)
    "kl_drift",
    "wasserstein_drift",
}


def _config(agg: str) -> dict:
    return {
        "target": "p",
        "max_depth": 2,
        "intervals": [],
        # `count` companion keeps the aggs CTE non-empty when `agg` fires on only
        # one column type — isolating the aggregator's own SQL.
        "aggregations": [agg, "count"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "p", "table": "p", "id": "pid", "variables": {}},
            {
                "alias": "c",
                "table": "c",
                "id": None,
                "temporal_ix": "ts",
                "variables": {
                    "num": {"type": "numeric"},
                    "cat": {"type": "categorical"},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "p", "key": "pid"},
                "child": {"entity": "c", "key": "pid"},
            }
        ],
    }


def _seed(conn, ts_type: str) -> None:
    create_temp_table(conn, "p", [("pid", "int")], [(i,) for i in range(1, 7)])
    create_temp_table(
        conn,
        "c",
        [("pid", "int"), ("ts", ts_type), ("num", "numeric"), ("cat", "text")],
        _ROWS,
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)])


def _fires(f: Featurizer, agg: str) -> bool:
    key = agg.upper().replace("_", "")
    return any(
        key in feat.label.upper().replace("_", "")
        for feat in f._plan.target_output_features
    )


@pytest.mark.parametrize("ts_type", ["date", "timestamp"])
@pytest.mark.parametrize("agg", _ALL_AGGS)
def test_aggregator_executes_without_error(pg_conn, agg, ts_type):
    """Every firing aggregator produces SQL that runs over the edge-case groups
    without raising (no division-by-zero, interval, or syntax error)."""
    import tempfile

    import yaml

    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(_config(agg), handle)
    handle.close()
    featurizer = Featurizer(handle.name, validate=False)
    if not _fires(featurizer, agg):
        pytest.skip(
            f"{agg} does not fire on the plain fixture (needs predicates/spatial)"
        )
    _seed(pg_conn, ts_type)
    rows = run_featurizer(pg_conn, _config(agg))
    assert len(rows) == 6  # one row per group, all materialized without error


def test_every_registered_aggregator_is_accounted_for():
    """No silent gaps: each registered aggregator either fires on the plain
    fixture (and is executed above) or is a known predicate/spatial primitive."""
    non_firing = []
    for agg in _ALL_AGGS:
        import tempfile

        import yaml

        h = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(_config(agg), h)
        h.close()
        if not _fires(Featurizer(h.name, validate=False), agg):
            non_firing.append(agg)
    # Everything that doesn't fire on the plain fixture must be a known
    # special-config primitive — otherwise a new aggregator slipped coverage.
    unexpected = set(non_firing) - _NEEDS_SPECIAL_CONFIG
    assert (
        not unexpected
    ), f"aggregators with no execution coverage: {sorted(unexpected)}"
