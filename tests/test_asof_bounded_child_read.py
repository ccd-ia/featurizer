# coding: utf-8

"""A child entity is read as of the as-of date (issue #27).

``<child>_synth`` and ``<child>_transform`` used to read the whole child table;
the cut on ``aod.as_of_date`` was applied only later, where the aggregation
reads ``<child>_transform``. A backward-only window (``lag``, a running ``sum``)
never noticed, because a row never looks past itself. A window that spans its
whole partition did: ``percent_rank()`` divides by the partition's size, and
that size counted rows dated after the as-of date. The surviving value carried
the count.

The cut now sits where the child is *read*, so no window can see an unknowable
row. These tests pin where the bound renders and where it does not; the values
are pinned in ``tests/integration/test_asof_bounded_child_read.py``.
"""

from __future__ import annotations

import re
import tempfile

import yaml

from featurizer import Featurizer


def _config(**overrides) -> dict:
    config = {
        "target": "series",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["max"],
        "transformations": ["identity", "percent_rank"],
        "entities": [
            {"alias": "series", "table": "series", "id": "series_id"},
            {
                "alias": "events",
                "table": "events",
                "id": "series_id",
                "temporal_ix": "ts",
                "variables": {"x": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": "series_id"},
                "child": {"entity": "events", "key": "series_id"},
            }
        ],
    }
    config.update(overrides)
    return config


def _featurizer(config: dict, **kwargs) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
    return Featurizer(handle.name, validate=False, **kwargs)


def _cte(query: str, name: str) -> str:
    """The body of one CTE, from ``<name> as (`` to the next CTE's comment."""
    match = re.search(
        rf"\b{name} as \((.*?)(?=\n\s*,\s*\n|\n\s*select \* from)", query, re.S
    )
    assert match, f"no CTE named {name} in the query"
    return match.group(1)


def test_the_child_synth_is_bounded_on_the_as_of_date() -> None:
    synth = _cte(_featurizer(_config()).query, "events_synth")
    assert 'where events."ts" <= aod.as_of_date' in " ".join(synth.split())


def test_the_bound_follows_the_boundary_mode() -> None:
    query = _featurizer(_config(as_of_boundary="exclusive")).query
    synth = " ".join(_cte(query, "events_synth").split())
    assert 'where events."ts" < aod.as_of_date' in synth


def test_a_target_with_a_temporal_index_is_bounded_too() -> None:
    """Issue #49 / ADR-0017. #27 left the target's read alone because its rows
    are the cohort; an event-like target then emitted rows dated after the as-of
    date, and they shaped the whole-partition values of the rows before them."""
    config = _config()
    config["entities"][0]["temporal_ix"] = "opened_at"
    synth = " ".join(_cte(_featurizer(config).query, "series_synth").split())
    assert 'where series."opened_at" <= aod.as_of_date' in synth


def test_a_target_without_a_temporal_index_is_not_bounded() -> None:
    """One row per entity and no timeline: nothing to cut on. Every triage-pg
    config is this shape, and its SQL does not move."""
    synth = _cte(_featurizer(_config()).query, "series_synth")
    assert "aod.as_of_date" not in synth


def test_a_paired_target_carries_both_predicates() -> None:
    config = _config(as_of_dates={"id_column": "cohort_id"})
    config["entities"][0]["temporal_ix"] = "opened_at"
    synth = " ".join(_cte(_featurizer(config).query, "series_synth").split())
    assert "_cohort.as_of_date = aod.as_of_date)" in synth
    assert 'and series."opened_at" <= aod.as_of_date' in synth


def test_a_child_without_a_temporal_index_is_not_bounded() -> None:
    """No temporal index, nothing to cut on — and no window to leak through:
    the whole-partition transformers order by the temporal index."""
    config = _config(transformations=["identity"])
    del config["entities"][1]["temporal_ix"]
    synth = _cte(_featurizer(config).query, "events_synth")
    assert "aod.as_of_date" not in synth


def test_a_temporal_index_that_needs_quoting_is_quoted_in_the_bound() -> None:
    config = _config(transformations=["identity"])
    config["entities"][1]["temporal_ix"] = "Event Date"
    synth = " ".join(_cte(_featurizer(config).query, "events_synth").split())
    assert 'where events."Event Date" <= aod.as_of_date' in synth


def test_cdf_renders_the_function_postgresql_has() -> None:
    """``cum_dist()`` does not exist. It was left misspelled on purpose until
    the child read was bounded, so that the fix could not switch on a third
    transformer that reads the future."""
    query = _featurizer(_config(transformations=["identity", "cdf"])).query
    assert "cume_dist()" in query
    assert "cum_dist()" not in query


def test_output_names_do_not_move() -> None:
    """ADR-0007 / ADR-0015: the bound changes what a child read sees, never
    what a column is called."""
    labels = {entry.label for entry in _featurizer(_config()).feature_manifest}
    assert {
        "MAX(events.x)",
        "MAX(events.PERCENT_RANK(events.x))",
    } <= labels


def test_the_target_cut_is_announced_once(monkeypatch) -> None:
    """ADR-0017 changes which rows a date returns; it must not arrive silently."""
    from loguru import logger

    import featurizer.planner as planner

    monkeypatch.setattr(planner, "_ANNOUNCED_TARGET_CUTS", set())
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        config = _config()
        config["entities"][0]["temporal_ix"] = "opened_at"
        _featurizer(config)
        _featurizer(config)
        _featurizer(_config())  # no temporal index on the target: nothing to say
    finally:
        logger.remove(sink)
    announced = [m for m in messages if "ADR-0017" in m]
    assert len(announced) == 1
    assert "'series'" in announced[0] and "'opened_at'" in announced[0]
