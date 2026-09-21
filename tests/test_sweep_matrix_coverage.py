# coding: utf-8

"""The sweep matrix stays closed (issue #50).

Between 2026-09-07 and 2026-09-20 every defect filed against featurizer came
out of a combination of features that no test executed: a primitive x a name
shape, a whole-partition window x a row after the as-of date, the TEMP-table
path x two as-of dates. Each class now has a registry-wide executing sweep. This
file is what keeps them registry-wide: a sweep that parametrizes over "the
primitives I thought of" is how the holes got there.

Two checks, both DB-free, so they fail in the fast tier:

- every sweep covers the WHOLE registry, minus exclusions that are written
  down here with their reason. Registering a primitive that a sweep skips —
  silently, because its input type has no sample, as ``cum_count`` was skipped
  by the quoting sweep until this file existed — fails here.
- every constructor parameter of the config's spec classes is classified as
  naming a column or not, and each one that does has a role in the identifier
  sweep. A new config key cannot land unclassified.

The rule for contributors is in CONTRIBUTING.md, "A new primitive, key or path
lands with its row in the matrix".
"""

from __future__ import annotations

import inspect

import pytest

import tests.integration.test_all_aggregators_execution as aggregators_execute
import tests.integration.test_asof_bounded_child_read as transformer_future_row
import tests.integration.test_cohort_child_read_sweeps as narrowed_child
import tests.integration.test_future_row_aggregations as aggregation_future_row
import tests.integration.test_transformer_selection as transformer_selection
import tests.test_aggregation_quoting as aggregation_quoting
import tests.test_identifier_roles_quoting as identifier_roles
import tests.test_transformer_quoting as transformer_quoting
from featurizer.primitives import abstractions
from featurizer.primitives.utils import list_aggregations, list_transformations

# Not configurable as a bare ``transformations:`` entry: ``identity`` is what a
# sweep adds beside the transformer under test, ``in_array`` needs an argument.
NOT_STANDALONE = {"identity", "in_array"}


def _first_values(params) -> set:
    return {param.values[0] for param in params}


def _parametrized_over(test_function) -> set:
    """The values of the first ``parametrize`` mark on a test function."""
    (mark,) = [m for m in test_function.pytestmark if m.name == "parametrize"]
    return set(mark.args[1])


TRANSFORMER_SWEEPS = {
    "executes over a non-bare variable": lambda: _first_values(
        transformer_quoting._sweep_cases()
    ),
    "a row after the as-of date moves nothing": lambda: _first_values(
        transformer_future_row._sweep_cases()
    ),
    "runs selected alone, without identity": lambda: _first_values(
        transformer_selection._cases()
    ),
    "a child narrowed by a paired cohort gives the dense value": lambda: _first_values(
        narrowed_child._sweep_cases()
    ),
}

AGGREGATION_SWEEPS = {
    "executes": lambda: set(aggregators_execute._ALL_AGGS),
    "executes over a non-bare variable or temporal index": lambda: (
        _first_values(aggregation_quoting._column_cases())
        | _first_values(aggregation_quoting._index_cases())
    ),
    "a row after the as-of date moves nothing": lambda: _parametrized_over(
        aggregation_future_row.test_a_row_after_the_as_of_date_moves_no_aggregation
    ),
    "a child narrowed by a paired cohort gives the dense value": lambda: (
        _parametrized_over(
            narrowed_child.test_every_aggregation_over_a_narrowed_child_gives_the_dense_value
        )
    ),
}


@pytest.mark.parametrize("sweep", sorted(TRANSFORMER_SWEEPS))
def test_a_transformer_sweep_covers_the_whole_registry(sweep: str) -> None:
    covered = TRANSFORMER_SWEEPS[sweep]()
    registry = set(list_transformations())
    assert registry - covered == NOT_STANDALONE, (
        f"'{sweep}' skips {sorted(registry - covered - NOT_STANDALONE)}: give the "
        "sweep a sample for their input type, or record why they are excluded"
    )
    assert covered <= registry


@pytest.mark.parametrize("sweep", sorted(AGGREGATION_SWEEPS))
def test_an_aggregation_sweep_covers_the_whole_registry(sweep: str) -> None:
    covered = AGGREGATION_SWEEPS[sweep]()
    registry = set(list_aggregations())
    assert covered == registry, (
        f"'{sweep}' skips {sorted(registry - covered)} "
        f"and names unknown {sorted(covered - registry)}"
    )


# --------------------------------------------------------------------------- #
# Identifier roles. For every parameter of the classes a config is parsed into:
# the role(s) of tests/test_identifier_roles_quoting.py that put a non-bare name
# there, or None with the reason it does not name a column.
# --------------------------------------------------------------------------- #

NAMES_A_COLUMN: dict[str, dict[str, object]] = {
    "Entity": {
        "alias": None,  # a featurizer-side name, never SQL-quoted data
        "table": None,  # a possibly schema-qualified relation; left as written
        "id": ["target id", "child id", "grandchild id", "node id"],
        "temporal_ix": ["child temporal index", "target temporal index"],
        "spatial_ix": ["left latitude", "right latitude"],  # parsed into SpatialIx
        "variables": ["carried index variable", "looked-up variable"],
        "edge": ["edge source"],  # parsed into EdgeSpec
        "peer_groups": ["peer-group categorical"],  # parsed into PeerGroupSpec
        "peer_group": ["peer-group categorical"],  # the singular spelling
    },
    "Relationship": {
        "parent": None,  # an entity alias
        "child": None,
        "parent_key": ["target id", "lookup parent key"],
        "child_key": ["child foreign key", "lookup child key"],
        "temporal_mode": None,
        "temporal_grace": None,  # an interval literal
        "temporal_child_field": ["lookup child_timestamp"],
        "name": None,  # a featurizer-side name
    },
    "EdgeSpec": {
        "entity": None,
        "node": None,  # an entity alias
        "source": ["edge source"],
        "target": ["edge target"],
        "weight": ["edge weight"],
        "timestamp": ["edge timestamp"],
        "features": None,  # family names
    },
    "PeerGroupSpec": {
        "entity": None,
        "by": ["peer-group categorical"],
        "measures": ["peer-group measure"],
    },
    "SpatialIx": {
        "entity": None,
        "lat": ["left latitude", "right latitude"],
        "lon": ["left longitude", "right longitude"],
        # A PostGIS geometry column: parsed, and no pass reads it yet (the
        # spatial pass warns and skips). It gets a role the day one does.
        "geom": None,
        "srid": None,  # a number
    },
    "SpatialRelationshipSpec": {
        "name": None,
        "left": None,  # entity aliases
        "right": None,
        "within_m": None,
        "bandwidth_m": None,
        "features": None,
    },
    "GraphRelationshipSpec": {
        "name": None,
        "left": None,
        "edges": ["edge source", "edge target", "edge timestamp"],
        "right": None,
        "directed": None,
        "measures": ["neighbour measure"],
        "shares": ["neighbour share"],
        "features": None,
    },
}

SWEPT_ROLES = (
    set(identifier_roles.PLAIN)
    | set(identifier_roles.PASS_PLAIN)
    | set(identifier_roles.EDGE_PLAIN)
)


@pytest.mark.parametrize("class_name", sorted(NAMES_A_COLUMN))
def test_every_config_parameter_is_classified(class_name: str) -> None:
    """A new parameter is a new config key. It may name a column."""
    parameters = set(
        inspect.signature(getattr(abstractions, class_name).__init__).parameters
    ) - {"self"}
    classified = set(NAMES_A_COLUMN[class_name])
    assert parameters == classified, (
        f"{class_name}: unclassified {sorted(parameters - classified)}, "
        f"stale {sorted(classified - parameters)}. Say in NAMES_A_COLUMN whether "
        "it names a column; if it does, give it a role in "
        "tests/test_identifier_roles_quoting.py"
    )


def test_every_column_naming_parameter_has_a_swept_role() -> None:
    for class_name, parameters in NAMES_A_COLUMN.items():
        for parameter, roles in parameters.items():
            for role in roles or []:
                assert role in SWEPT_ROLES, (
                    f"{class_name}.{parameter} names a column through the role "
                    f"{role!r}, which the identifier sweep does not have"
                )
