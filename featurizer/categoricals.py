# coding: utf-8

"""Fixed-vocabulary one-hot encoding for direct (target-entity) categoricals.

Featurizer is **split-blind and fit-free**: it may encode a categorical only
when the vocabulary comes from a *declared* source — a ``vocabulary: [...]`` list
in the config, or the column's PostgreSQL ``ENUM`` labels. It must **never learn
a vocabulary by scanning the data** for distinct values: that statistic would be
fit on whatever rows happen to be present, leaking the test-period vocabulary
into what a downstream consumer treats as a train-only transform.

By vocabulary source:

- **Declared / ENUM** vocabulary -> fit-free, deterministic -> *featurizer owns this*.
- **Learned** vocabulary (from data) -> a fitted, split-sensitive transform ->
  the *consumer's* job (train-only), never featurizer's.

The encoder emits one deterministic 0/1 column per vocabulary value, named
``"<entity_alias>.<column>=<value>"`` (a quoted PostgreSQL identifier; long names
are capped by :func:`~featurizer.primitives.abstractions.pg_identifier`). A NULL
or out-of-vocabulary value yields an all-zero row (never a crash).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from loguru import logger

from .primitives import Feature, pg_identifier

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .primitives import Entity, Variable

#: Per-variable ``role`` values controlling direct-variable handling.
ROLE_IDENTIFIER = "identifier"
ROLE_CATEGORICAL = "categorical"
ROLE_NUMERIC = "numeric"


def _sql_string_literal(value: Any) -> str:
    """A single-quoted SQL string literal with embedded quotes doubled."""
    return "'" + str(value).replace("'", "''") + "'"


def _split_schema_table(table: str) -> tuple[Optional[str], str]:
    """Split a (possibly schema-qualified) table name into (schema, table).

    ``food_inspections.facilities`` -> ``("food_inspections", "facilities")``;
    a bare ``facilities`` -> ``(None, "facilities")`` (resolved against the
    session ``current_schema()`` during introspection). Surrounding double
    quotes, if any, are stripped.
    """
    schema, _, name = table.rpartition(".")
    schema = schema.strip('"') or None
    return schema, name.strip('"')


class OneHotFeature(Feature):
    """One deterministic 0/1 indicator column for a single categorical value.

    ``definition`` casts the source column to ``text`` before comparison so the
    same expression is valid for both ``ENUM`` and ``text``/``varchar`` columns
    and an out-of-vocabulary data value can never raise an invalid-enum-input
    error — it simply matches no ``when`` branch and yields ``0``. A NULL source
    likewise falls through to ``0`` (the ``when`` predicate is unknown).
    """

    def __init__(self, *, entity: "Entity", source_column: str, value: str) -> None:
        intended = f"{entity.alias}.{source_column}={value}"
        super().__init__(
            name=pg_identifier(intended),
            type="numeric",
            definition=(
                f"case when {source_column}::text = {_sql_string_literal(value)} "
                "then 1 else 0 end"
            ),
            entity=entity,
            label=intended,
            description=f"one-hot indicator: {intended}",
        )
        self.source_column: str = source_column
        self.value: str = value


def introspect_enum_labels(
    connection: Any, table: str, column: str
) -> Optional[List[str]]:
    """Return a column's PostgreSQL ``ENUM`` labels, or ``None`` if not an enum.

    Reads ``pg_enum`` for the type of ``table.column`` (schema-qualified or
    resolved against ``current_schema()``), in declared ``enumsortorder``. The
    caller sorts for determinism. Returns ``None`` when the column exists but is
    not an enum (so the caller can fail loud), and ``None`` for an empty result.
    """
    schema, name = _split_schema_table(table)
    sql = """
        select e.enumlabel
        from pg_catalog.pg_attribute a
        join pg_catalog.pg_class      c on c.oid = a.attrelid
        join pg_catalog.pg_namespace  n on n.oid = c.relnamespace
        join pg_catalog.pg_type       t on t.oid = a.atttypid
        join pg_catalog.pg_enum       e on e.enumtypid = t.oid
        where c.relname = %s
          and a.attname = %s
          and n.nspname = coalesce(%s, current_schema())
        order by e.enumsortorder
    """
    with connection.cursor() as cur:
        cur.execute(sql, (name, column, schema))
        labels = [row[0] for row in cur.fetchall()]
    return labels or None


#: Above this many vocabulary values, one-hot encoding a direct categorical is
#: warned as high-cardinality: it emits one sparse 0/1 column per value, which is
#: wide and weak. featurizer is split-blind, so it cannot frequency- or
#: target-encode instead (those are fitted, train-only transforms); the fix is to
#: declare a top-N ``vocabulary:`` and let the long tail fall into the all-zero
#: "other". Chosen above the largest vocabulary in practice (15) so it never fires
#: on a deliberately-capped config, only on a genuinely large declared list or an
#: un-curated ENUM.
ONE_HOT_CARDINALITY_WARN = 25


def _warn_if_high_cardinality(
    vocab: List[str], entity: "Entity", variable: "Variable", source: str
) -> List[str]:
    if len(vocab) > ONE_HOT_CARDINALITY_WARN:
        logger.warning(
            "Categorical {}.{} has a {}-value {} vocabulary → {} one-hot columns "
            '("{}.{}=<value>"). High-cardinality one-hot is sparse and wide; '
            "declare a top-N `vocabulary:` (the long tail becomes an all-zero "
            '"other") rather than encoding every value. featurizer is split-blind '
            "and cannot frequency/target-encode (those are fitted, train-only).",
            entity.alias,
            variable.name,
            len(vocab),
            source,
            len(vocab),
            entity.alias,
            variable.name,
        )
    return vocab


def resolve_vocabulary(
    variable: "Variable", entity: "Entity", connection: Any
) -> List[str]:
    """Resolve a categorical's fixed vocabulary, sorted for determinism.

    Priority: (a) the declared ``variable.vocabulary``; (b) the column's
    PostgreSQL ``ENUM`` labels (requires ``connection``); (c) otherwise **raise**.
    Never scans the data for distinct values — that would be the forbidden
    learned vocabulary.
    """
    if variable.vocabulary:
        return _warn_if_high_cardinality(
            sorted(str(v) for v in variable.vocabulary), entity, variable, "declared"
        )

    if connection is not None:
        labels = introspect_enum_labels(connection, entity.table, variable.name)
        if labels:
            logger.info(
                "Resolved vocabulary for {}.{} from its PostgreSQL ENUM ({} labels).",
                entity.alias,
                variable.name,
                len(labels),
            )
            return _warn_if_high_cardinality(
                sorted(labels), entity, variable, "PostgreSQL ENUM"
            )

    raise ValueError(
        f"role: categorical variable '{entity.alias}.{variable.name}' has no "
        "vocabulary. Declare a fixed `vocabulary: [...]` in the config, or type "
        f"the column ({entity.table}.{variable.name}) as a PostgreSQL ENUM so its "
        "labels can be introspected (a database connection is required for that). "
        "Featurizer is split-blind and will not scan the data for distinct values "
        "— a data-derived vocabulary is a fitted, train-only transform and belongs "
        "to the consumer, not to featurizer."
    )


def build_one_hot_features(
    entity: "Entity", variable: "Variable"
) -> List[OneHotFeature]:
    """Build the sorted one-hot indicator columns for a resolved categorical.

    Requires ``variable.vocabulary`` to already be resolved (see
    :func:`resolve_vocabulary`, called once at ``Featurizer`` construction so the
    planner stays DB-free). Raises if it is missing.
    """
    if not variable.vocabulary:
        raise ValueError(
            f"Cannot one-hot encode '{entity.alias}.{variable.name}': its "
            "vocabulary was not resolved. This is resolved at Featurizer "
            "construction (declared list or introspected ENUM); see "
            "resolve_vocabulary."
        )
    return [
        OneHotFeature(entity=entity, source_column=variable.name, value=value)
        for value in variable.vocabulary
    ]
