"""Bug #8, transformer family: transformer output names must survive
PostgreSQL's 63-byte identifier cap the same way aggregation names do.

Before this fix, ``Transformer`` (and the window / rolling / lag / EMA /
Holt-Winters / diff / population families) built their output name inline as a
raw quoted string that never routed through :func:`pg_identifier`. A long
transformer-wrapped name — e.g. ``ABS(patients.MEAN(visits.ABS(...)|...))`` at
68 bytes — was emitted verbatim and *silently truncated by PostgreSQL* to 63
bytes, so two variants sharing a 63-byte prefix collapsed into one ambiguous
column, and nothing carried the full intended name for the manifest.

Now every transformer routes its name through ``pg_identifier`` (deterministic
hash suffix past 63 bytes) AND carries a full untruncated ``label`` so the
manifest maps column -> intended name. Mirrors
``tests/primitives/test_long_feature_names.py`` for aggregations.
"""

from featurizer.manifest import build_feature_manifest
from featurizer.primitives.abstractions import Entity
from featurizer.primitives.utils import get_transformers

# A 58-char numeric column; ``ABS(e.<col>)`` is 6 + 58 + 1 = 65 bytes > 63.
LONG_COL = "a_really_quite_long_numeric_measurement_column_name_forcap"
SHORT_COL = "amount"


def _entity(col: str) -> Entity:
    return Entity(
        alias="e",
        table="t",
        id="id",
        temporal_ix="ts",
        variables={col: {"type": "numeric"}},
    )


def _feature(entity: Entity, col: str):
    return next(ft for ft in entity.features if ft.name == col)


def _apply(name: str, col: str):
    entity = _entity(col)
    transformer = get_transformers([name])[name]
    return transformer(entity, _feature(entity, col))


def test_long_transformer_name_fits_identifier_limit():
    # One representative from each affected code path.
    for tname in ("abs", "cum_sum", "lag_1", "rolling_mean_7", "ema_7", "cusum"):
        result = _apply(tname, LONG_COL)
        quoted = result.name
        assert quoted.startswith('"') and quoted.endswith('"'), tname
        assert len(quoted.strip('"').encode()) <= 63, (tname, quoted)
        # ...and the hash suffix marks that the readable tail was capped.
        assert "~" in quoted, (tname, quoted)


def test_long_transformer_carries_full_label():
    result = _apply("abs", LONG_COL)
    # The label is the full, untruncated intended name...
    assert result.label == f"ABS(e.{LONG_COL})"
    assert len(result.label.encode()) > 63
    # ...and it differs from the (capped) rendered column name.
    assert result.label != result.name.strip('"')


def test_short_transformer_names_are_byte_identical():
    # The ADR-0007 name-stability contract: names that already fit are unchanged.
    result = _apply("abs", SHORT_COL)
    assert result.name == '"ABS(e.amount)"'
    assert result.label == "ABS(e.amount)"


def test_long_transformer_variants_do_not_collide():
    # Two distinct long columns must stay distinct within the first 63 bytes —
    # what PostgreSQL actually compares after its own truncation.
    a = _apply("abs", LONG_COL + "_alpha")
    b = _apply("abs", LONG_COL + "_omega")
    assert a.name != b.name
    truncated = {a.name.strip('"').encode()[:63], b.name.strip('"').encode()[:63]}
    assert len(truncated) == 2


def test_manifest_maps_capped_transformer_column_to_full_label():
    result = _apply("abs", LONG_COL)
    (entry,) = build_feature_manifest([result])
    assert entry.column == result.name.strip('"')
    assert entry.label == f"ABS(e.{LONG_COL})"
    assert entry.truncated is True
    # The label grammar still parses, so lineage/description are populated.
    assert entry.depth == 1
    assert entry.description  # non-empty generated description


def test_transformer_label_uses_parent_label_not_capped_name():
    """A transformer over an already-capped parent uses the parent's full
    label, so the readable chain survives at any nesting depth."""
    entity = _entity(LONG_COL)
    inner = _apply("abs", LONG_COL)  # already capped in .name, full in .label
    # Re-home the intermediate onto the same entity and wrap it again.
    inner.entity = entity
    outer = get_transformers(["abs"])["abs"](entity, inner)
    # The outer label nests the parent's *label*, not its hash-capped name.
    assert f"ABS(e.{LONG_COL})" in outer.label
    assert "~" not in outer.label  # readable chain, no hash fragments
    assert len(outer.name.strip('"').encode()) <= 63  # still capped for PG
