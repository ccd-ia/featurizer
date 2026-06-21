# coding: utf-8

"""Feature manifest: a column ↔ full-intended-name map for the output matrix.

A generated feature name longer than PostgreSQL's 63-byte identifier limit is
capped with a stable hash suffix (see
:func:`~featurizer.primitives.abstractions.pg_identifier`), which erases the
readable tail. The manifest records, for every output column, both the rendered
``column`` name (exactly as it appears in the SQL / Arrow / Parquet output) and
the full untruncated ``label`` — so humans, agents, partner-facing tables, and
plot legends can recover the intended name, and one-hot columns expose their
``source_column`` and ``value``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .categoricals import OneHotFeature
from .primitives import Variable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from .primitives import Feature


@dataclass(frozen=True)
class ManifestEntry:
    """One row of the feature manifest."""

    column: str
    """Rendered output column name (possibly hash-truncated)."""
    label: str
    """Full, untruncated, human-readable intended name."""
    truncated: bool
    """Whether ``column`` differs from ``label`` (the readable tail was capped)."""
    kind: str
    """``one_hot`` | ``variable`` | ``derived``."""
    entity: Optional[str]
    source_column: Optional[str]
    """For ``one_hot`` columns: the categorical column they encode."""
    value: Optional[str]
    """For ``one_hot`` columns: the category value this column indicates."""
    definition: Optional[str]
    """The SQL expression that computes the column."""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _strip_quotes(name: str) -> str:
    return name.replace('"', "")


def _classify(feature: "Feature") -> str:
    if isinstance(feature, OneHotFeature):
        return "one_hot"
    if isinstance(feature, Variable):
        return "variable"
    return "derived"


def build_feature_manifest(
    output_features: List["Feature"],
) -> List[ManifestEntry]:
    """Build the manifest entries for a target's output feature columns."""
    entries: List[ManifestEntry] = []
    for feature in output_features:
        column = _strip_quotes(feature.name)
        label = _strip_quotes(feature.label)
        kind = _classify(feature)
        entries.append(
            ManifestEntry(
                column=column,
                label=label,
                truncated=column != label,
                kind=kind,
                entity=feature.entity.alias if feature.entity is not None else None,
                source_column=(
                    feature.source_column
                    if isinstance(feature, OneHotFeature)
                    else None
                ),
                value=feature.value if isinstance(feature, OneHotFeature) else None,
                definition=feature.definition,
            )
        )
    return entries


def manifest_dataframe(entries: List[ManifestEntry]) -> "pd.DataFrame":
    """Render manifest entries as a pandas DataFrame (for tables / plots / joins)."""
    import pandas as pd

    columns = [
        "column",
        "label",
        "truncated",
        "kind",
        "entity",
        "source_column",
        "value",
        "definition",
    ]
    return pd.DataFrame([e.as_dict() for e in entries], columns=pd.Index(columns))
