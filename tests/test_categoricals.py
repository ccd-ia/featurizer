"""DB-free tests for direct-categorical one-hot encoding and variable roles.

These assert on the *shape* of the generated SQL and on the resolved feature
set, so they hold without a PostgreSQL (the introspection/end-to-end paths are
covered by the integration tier in
``tests/integration/test_direct_categoricals.py``).
"""

from __future__ import annotations

import contextlib
import re
import tempfile
from typing import Any, Dict, Iterator, List

import pytest
import yaml
from loguru import logger

from featurizer import Featurizer


@contextlib.contextmanager
def capture_warnings() -> Iterator[List[str]]:
    """Collect loguru WARNING messages emitted within the block."""
    messages: List[str] = []
    sink_id = logger.add(
        lambda m: messages.append(m.record["message"]), level="WARNING"
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _facilities_config(facility_type_def: Dict[str, Any], **extra_vars: Any) -> dict:
    variables: Dict[str, Any] = {"facility_type": facility_type_def}
    variables.update(extra_vars)
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
                "variables": variables,
            }
        ],
        "relationships": [],
    }


def _featurizer(config: dict, **kwargs: Any) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path, **kwargs)


def _no_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("DATABASE_URL", "PGHOST", "PGDATABASE", "PGPORT", "PGUSER"):
        monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
# role: identifier
# --------------------------------------------------------------------------- #


def test_identifier_role_excluded_and_logged() -> None:
    config = _facilities_config(
        {"type": "categorical", "role": "categorical", "vocabulary": ["A", "B"]},
        name={"type": "text", "role": "identifier"},
        risk_score={"type": "numeric"},
    )
    with capture_warnings() as warnings:
        featurizer = _featurizer(config)
    query = featurizer.query

    # The identifier column is neither projected nor in the manifest.
    columns = {entry.column for entry in featurizer.feature_manifest}
    assert "name" not in columns
    assert not re.search(r"(?<![A-Za-z0-9_])name(?![A-Za-z0-9_])", query)

    # ...and the omission is loud, naming the column.
    assert any("identifier" in m and "facilities.name" in m for m in warnings)


# --------------------------------------------------------------------------- #
# role: categorical with a declared vocabulary
# --------------------------------------------------------------------------- #


def test_declared_vocabulary_one_hot_columns_sorted() -> None:
    config = _facilities_config(
        {
            "type": "categorical",
            "role": "categorical",
            # deliberately unsorted to prove we sort
            "vocabulary": ["Restaurant", "Grocery Store", "School"],
        }
    )
    featurizer = _featurizer(config)
    query = featurizer.query

    one_hot_cols = [
        e.column for e in featurizer.feature_manifest if e.kind == "one_hot"
    ]
    assert one_hot_cols == [
        "facilities.facility_type=Grocery Store",
        "facilities.facility_type=Restaurant",
        "facilities.facility_type=School",
    ]

    # Each value yields a deterministic 0/1 indicator; ``else 0`` (not NULL) is
    # what makes a NULL or out-of-vocabulary value an all-zero row, not a crash.
    for value in ("Restaurant", "Grocery Store", "School"):
        fragment = f"case when facility_type::text = '{value}' then 1 else 0 end"
        assert fragment in query
    assert "else null" not in query.lower()

    # The raw categorical string never reaches the output.
    assert "facility_type as facility_type" not in query


def test_high_cardinality_vocabulary_warns_but_still_encodes() -> None:
    # A 30-value declared vocabulary (e.g. a per-zipcode encoding) exceeds the
    # one-hot cardinality threshold: it should WARN but still encode every value
    # (split-blind, no silent data loss).
    vocab = [f"v{i:02d}" for i in range(30)]
    config = _facilities_config(
        {"type": "categorical", "role": "categorical", "vocabulary": vocab}
    )
    with capture_warnings() as warnings:
        featurizer = _featurizer(config)

    one_hot_cols = [
        e.column for e in featurizer.feature_manifest if e.kind == "one_hot"
    ]
    assert len(one_hot_cols) == 30  # every value still encoded
    assert any(
        "high-cardinality" in m.lower() and "facilities.facility_type" in m
        for m in warnings
    ), warnings


def test_small_vocabulary_does_not_warn() -> None:
    # A deliberately-capped top-N vocabulary (like triage's zip_code=12) is fine.
    config = _facilities_config(
        {
            "type": "categorical",
            "role": "categorical",
            "vocabulary": [f"z{i}" for i in range(12)],
        }
    )
    with capture_warnings() as warnings:
        _featurizer(config)
    assert not any("high-cardinality" in m.lower() for m in warnings), warnings


def test_declared_vocabulary_escapes_single_quotes() -> None:
    config = _facilities_config(
        {"type": "categorical", "role": "categorical", "vocabulary": ["O'Hare"]}
    )
    query = _featurizer(config).query
    assert "case when facility_type::text = 'O''Hare' then 1 else 0 end" in query


def test_one_hot_manifest_records_source_and_value() -> None:
    config = _facilities_config(
        {"type": "categorical", "role": "categorical", "vocabulary": ["Restaurant"]}
    )
    entries = [e for e in _featurizer(config).feature_manifest if e.kind == "one_hot"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.column == "facilities.facility_type=Restaurant"
    assert entry.label == "facilities.facility_type=Restaurant"
    assert entry.truncated is False
    assert entry.source_column == "facility_type"
    assert entry.value == "Restaurant"


# --------------------------------------------------------------------------- #
# fail-loud: no vocabulary and no database
# --------------------------------------------------------------------------- #


def test_categorical_without_vocabulary_or_db_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_db_env(monkeypatch)
    config = _facilities_config({"type": "categorical", "role": "categorical"})
    with pytest.raises(ValueError, match="has no .*vocabulary|declare"):
        _featurizer(config)


# --------------------------------------------------------------------------- #
# no role: warn before passing a raw string column through unencoded
# --------------------------------------------------------------------------- #


def test_unencoded_text_passthrough_warns() -> None:
    config = _facilities_config(
        {"type": "categorical", "role": "categorical", "vocabulary": ["A"]},
        category={"type": "text"},  # no role -> footgun
    )
    with capture_warnings() as warnings:
        _featurizer(config)
    assert any(
        "raw string column" in m and "facilities.category" in m for m in warnings
    )
