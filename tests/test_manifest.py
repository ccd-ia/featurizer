"""DB-free tests for the feature manifest (column <-> full intended name)."""

from __future__ import annotations

import tempfile
from typing import Any

import yaml

from featurizer import Featurizer


def _featurizer(config: dict) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path)


def _config(facility_type_def: dict[str, Any]) -> dict:
    return {
        "target": "facilities",
        "max_depth": 1,
        "intervals": ["P1Y"],
        "aggregations": [],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "facilities",
                "id": "license_no",
                "table": "dirtyduck.facilities",
                "temporal_ix": "first_seen",
                "variables": {
                    "facility_type": facility_type_def,
                    "risk_score": {"type": "numeric"},
                },
            }
        ],
        "relationships": [],
    }


def test_manifest_kinds_cover_variable_and_one_hot() -> None:
    featurizer = _featurizer(
        _config(
            {"type": "categorical", "role": "categorical", "vocabulary": ["A", "B"]}
        )
    )
    by_column = {e.column: e for e in featurizer.feature_manifest}

    # A plain numeric passthrough is a ``variable``.
    assert by_column["risk_score"].kind == "variable"
    assert by_column["risk_score"].label == "risk_score"
    assert by_column["risk_score"].truncated is False

    # The one-hot columns are present with their source/value.
    one_hots = {c: e for c, e in by_column.items() if e.kind == "one_hot"}
    assert set(one_hots) == {
        "facilities.facility_type=A",
        "facilities.facility_type=B",
    }
    assert one_hots["facilities.facility_type=A"].value == "A"
    assert one_hots["facilities.facility_type=A"].source_column == "facility_type"


def test_manifest_recovers_truncated_long_name() -> None:
    long_value = (
        "Mobile Food Dispenser With An Extremely Long Descriptive Category Name Indeed"
    )
    featurizer = _featurizer(
        _config(
            {
                "type": "categorical",
                "role": "categorical",
                "vocabulary": [long_value, "Short"],
            }
        )
    )
    entries = {e.value: e for e in featurizer.feature_manifest if e.kind == "one_hot"}

    long_entry = entries[long_value]
    # The rendered column name is capped at PostgreSQL's 63-byte identifier limit.
    assert len(long_entry.column) <= 63
    assert long_entry.truncated is True
    # ...but the full, human-readable intended name is preserved as the label.
    assert long_entry.label == f"facilities.facility_type={long_value}"

    short_entry = entries["Short"]
    assert short_entry.truncated is False
    assert short_entry.column == short_entry.label


def test_manifest_dataframe_shape() -> None:
    featurizer = _featurizer(
        _config({"type": "categorical", "role": "categorical", "vocabulary": ["A"]})
    )
    frame = featurizer.manifest_dataframe()
    assert list(frame.columns) == [
        "column",
        "label",
        "truncated",
        "kind",
        "entity",
        "source_column",
        "value",
        "definition",
    ]
    row = frame[frame["column"] == "facilities.facility_type=A"].iloc[0]
    assert row["kind"] == "one_hot"
    assert row["value"] == "A"
