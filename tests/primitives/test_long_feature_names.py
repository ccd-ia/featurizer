"""Bug #8 guard: feature names must survive PostgreSQL's 63-byte identifier cap.

PostgreSQL silently truncates identifiers to 63 bytes, so two long feature
names sharing a 63-byte prefix (e.g. the P6M and P1Y interval variants of
``MARKOV_CONDITIONAL_ENTROPY(inspections.inspection_type|...)``) collapse
into one ambiguous column. ``Aggregator._build_name`` caps long names with a
stable hash suffix instead. Found by the realistic food-inspections dataset.
"""

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.utils import get_aggregations


def _setup():
    parent = Entity(alias="facilities", table="f", id="license_no")
    child = Entity(
        alias="inspections",
        table="i",
        id="inspection_id",
        temporal_ix="inspection_date",
        variables={"inspection_type": {"type": "categorical"}},
    )
    rel = Relationship(
        parent=parent, child=child, parent_key="license_no", child_key="license_no"
    )
    feature = next(f for f in child.features if f.name == "inspection_type")
    return parent, child, rel, feature


def _name(interval):
    parent, child, rel, feature = _setup()
    agg = get_aggregations(["markov_conditional_entropy"])["markov_conditional_entropy"]
    return agg(parent, child, feature, interval=interval, relationship=rel).name


def test_long_names_fit_postgres_identifier_limit():
    for interval in (None, "P6M", "P1Y"):
        quoted = _name(interval)
        assert quoted.startswith('"') and quoted.endswith('"')
        assert len(quoted.strip('"').encode()) <= 63, quoted


def test_long_interval_variants_do_not_collide():
    names = {_name("P6M"), _name("P1Y"), _name(None)}
    assert len(names) == 3
    # ...and not merely distinct: distinct within the first 63 bytes, which is
    # what PostgreSQL actually compares after truncation.
    truncated = {name.strip('"').encode()[:63] for name in names}
    assert len(truncated) == 3


def test_short_names_are_untouched():
    parent, child, rel, feature = _setup()
    agg = get_aggregations(["count"])["count"]
    result = agg(parent, child, feature, relationship=rel)
    assert result.name == '"COUNT(inspections.inspection_type)"'
