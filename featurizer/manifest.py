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

Since v0.5.0 each entry also carries lineage (``depth``, immediate ``parents``,
the ``source_alias`` stream a derived feature was computed over, the outermost
``interval`` window) and a mechanically generated human ``description``
templated from the primitive documentation — and ``Featurizer.to_tables``
persists the whole manifest as a ``"<schema>"."<stem>_manifest"`` table beside
the feature-group tables.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
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
    depth: int = 0
    """Derivation depth (``Feature.stack_depth``): 0 = base variable."""
    parents: List[str] = field(default_factory=list)
    """Immediate parent feature labels (the derivation chain, one level up)."""
    source_alias: Optional[str] = None
    """The stream alias a derived feature was computed over — the relationship
    naming alias for aggregations over a named relationship, else the source
    entity alias. None for plain variables."""
    interval: Optional[str] = None
    """The outermost ``|interval=...`` window, when the feature is windowed."""
    description: str = ""
    """Mechanically generated human description (see ``_describe``)."""

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


def _iter_parents(feature: "Feature") -> List["Feature"]:
    """Normalize ``Feature.parents`` (assigned both bare Features and lists)."""
    parents = getattr(feature, "parents", None)
    if parents is None:
        return []
    if isinstance(parents, (list, tuple, set)):
        return [p for p in parents if p is not None]
    return [parents]


# ``OP(alias.rest`` — the operator token and the stream alias it reads.
_DERIVED_HEAD = re.compile(r"^([A-Za-z0-9_]+)\(([A-Za-z0-9_]+)\.(.*)\)$", re.DOTALL)
# ``alias.column`` with no operator — a qualified direct transfer.
_QUALIFIED_DIRECT = re.compile(r"^([A-Za-z0-9_]+)\.([^()=|]+)$")
_INTERVAL = re.compile(r"\|interval=([^)|]+)")


def _parse_label(label: str) -> Dict[str, Optional[str]]:
    """Extract (op, source_alias, argument, interval) from a feature label.

    The label grammar is the naming scheme the primitives emit:
    ``OP(alias.argument|interval=P1M)`` with arbitrary nesting inside
    ``argument``. The OUTERMOST interval is the last ``|interval=`` at the top
    nesting level; for simplicity the last occurrence overall is used — for
    every name the engine emits today they coincide.
    """
    op = source_alias = argument = interval = None
    match = _DERIVED_HEAD.match(label)
    if match:
        op, source_alias, rest = match.groups()
        intervals = _INTERVAL.findall(label)
        interval = intervals[-1] if intervals else None
        argument = f"{source_alias}." + _INTERVAL.sub("", rest)
    else:
        direct = _QUALIFIED_DIRECT.match(label)
        if direct:
            source_alias = direct.group(1)
            argument = label
    return {
        "op": op,
        "source_alias": source_alias,
        "argument": argument,
        "interval": interval,
    }


def _primitive_docs() -> Dict[str, str]:
    """Primitive name -> short description, from the CLI documentation dicts.

    Imported lazily (function level) so manifest.py never creates a module
    import cycle with cli.py, which imports validation/primitives itself.
    """
    from .cli import AGGREGATION_DOCS, TRANSFORMATION_DOCS

    docs: Dict[str, str] = {}
    for name, info in {**TRANSFORMATION_DOCS, **AGGREGATION_DOCS}.items():
        description = info.get("description")
        if isinstance(description, str) and description:
            docs[name.lower()] = description.rstrip(".")
    return docs


def _describe(
    kind: str,
    label: str,
    entity: Optional[str],
    parsed: Dict[str, Optional[str]],
    feature: "Feature",
    docs: Dict[str, str],
) -> str:
    """A deterministic, DB-free human description for one output column."""
    if kind == "one_hot":
        source = getattr(feature, "source_column", None)
        value = getattr(feature, "value", None)
        return (
            f"1 when {entity}.{source} = '{value}', else 0 "
            "(fixed-vocabulary one-hot; NULL/out-of-vocabulary rows are all-zero)"
        )
    if kind == "variable":
        declared = feature.description
        if declared and declared != "a feature":
            return declared
        return f"Direct column {entity}.{_strip_quotes(feature.name)}"
    op, argument, interval = parsed["op"], parsed["argument"], parsed["interval"]
    if op is not None:
        base = docs.get(op.lower())
        if base is not None:
            text = f"{base}, applied to {argument}"
            if interval:
                text += f", over the trailing {interval} window"
            return text
    if parsed["source_alias"] is not None and op is None:
        return f"Column {argument} transferred from the '{parsed['source_alias']}' relationship"
    return f"Derived feature: {label}"


def build_feature_manifest(
    output_features: List["Feature"],
) -> List[ManifestEntry]:
    """Build the manifest entries for a target's output feature columns."""
    docs = _primitive_docs()
    entries: List[ManifestEntry] = []
    for feature in output_features:
        column = _strip_quotes(feature.name)
        label = _strip_quotes(feature.label)
        kind = _classify(feature)
        parsed = _parse_label(label)
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
                depth=feature.stack_depth,
                parents=[
                    _strip_quotes(parent.label or parent.name)
                    for parent in _iter_parents(feature)
                ],
                source_alias=parsed["source_alias"],
                interval=parsed["interval"],
                description=_describe(
                    kind,
                    label,
                    feature.entity.alias if feature.entity is not None else None,
                    parsed,
                    feature,
                    docs,
                ),
            )
        )
    return entries


def manifest_dataframe(entries: List[ManifestEntry]) -> "pd.DataFrame":
    """Render manifest entries as a pandas DataFrame (for tables / plots / joins).

    ``parents`` (a list) is rendered comma-joined for tabular consumption.
    """
    import pandas as pd

    columns = [
        "column",
        "label",
        "truncated",
        "kind",
        "entity",
        "source_alias",
        "depth",
        "parents",
        "interval",
        "source_column",
        "value",
        "description",
        "definition",
    ]
    rows = []
    for entry in entries:
        record = entry.as_dict()
        record["parents"] = ", ".join(record["parents"])
        rows.append(record)
    return pd.DataFrame(rows, columns=pd.Index(columns))
