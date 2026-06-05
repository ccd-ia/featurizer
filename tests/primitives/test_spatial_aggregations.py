"""Tests for the spatial substrate (Phase 9): SpatialIx + spatial aggregators."""

from featurizer.primitives.abstractions import Entity, Id, Relationship, SpatialIx
from featurizer.primitives.utils import get_aggregations, list_aggregations

SPATIAL = ["distance_travelled", "radius_of_gyration", "spatial_std", "bbox_area"]


def _setup(spatial=True, temporal=True):
    parent = Entity(alias="users", table="u", id="user_id")
    kwargs = dict(
        alias="pings",
        table="p",
        id="ping_id",
        variables={"amount": {"type": "numeric"}},
    )
    if temporal:
        kwargs["temporal_ix"] = "ts"
    if spatial:
        kwargs["spatial_ix"] = {"lat": "latitude", "lon": "longitude"}
    child = Entity(**kwargs)
    rel = Relationship(
        parent=parent, child=child, parent_key="user_id", child_key="user_id"
    )
    return parent, child, rel


# --------------------------------------------------------------------------- #
# SpatialIx parsing
# --------------------------------------------------------------------------- #


def test_spatial_ix_dict_parsed():
    _, child, _ = _setup()
    sx = child.spatial_ix
    assert isinstance(sx, SpatialIx)
    assert sx.lat == "latitude" and sx.lon == "longitude"
    assert sx.backend == "plain"
    assert [c.name for c in sx.columns] == ["latitude", "longitude"]


def test_spatial_columns_flow_into_indexes():
    _, child, _ = _setup()
    names = [ix.name for ix in child.indexes]
    assert "latitude" in names and "longitude" in names


def test_spatial_ix_string_is_backward_compatible():
    e = Entity(alias="geo", table="g", id="gid", spatial_ix="coordinates")
    assert isinstance(e.spatial_ix, Id)
    assert e.spatial_ix.name == "coordinates"
    assert "coordinates" in [ix.name for ix in e.indexes]


# --------------------------------------------------------------------------- #
# Spatial aggregators
# --------------------------------------------------------------------------- #


def test_registered():
    available = set(list_aggregations())
    for name in SPATIAL:
        assert name in available


def test_fires_and_bounded():
    parent, child, rel = _setup()
    for name in SPATIAL:
        agg = get_aggregations([name])[name]
        result = agg(parent, child, child.temporal_ix, relationship=rel)
        assert result is not None, name
        assert "latitude" in result.definition and "longitude" in result.definition
        assert "<= aod.as_of_date" in result.definition  # causal bound


def test_distance_uses_haversine_and_lag():
    parent, child, rel = _setup()
    agg = get_aggregations(["distance_travelled"])["distance_travelled"]
    result = agg(parent, child, child.temporal_ix, relationship=rel)
    assert "asin(sqrt(" in result.definition
    assert "LAG(sub.latitude)" in result.definition


def test_spatial_std_formula():
    parent, child, rel = _setup()
    agg = get_aggregations(["spatial_std"])["spatial_std"]
    result = agg(parent, child, child.temporal_ix, relationship=rel)
    assert "var_samp(sub.latitude) + var_samp(sub.longitude)" in result.definition


def test_requires_spatial_ix():
    parent, child, rel = _setup(spatial=False)
    for name in SPATIAL:
        agg = get_aggregations([name])[name]
        assert agg(parent, child, child.temporal_ix, relationship=rel) is None


def test_requires_temporal_ix():
    parent, child, rel = _setup(temporal=False)
    # no temporal_ix → no feature to fire on, and the guard returns None
    for name in SPATIAL:
        agg = get_aggregations([name])[name]
        # pass the amount feature; spatial aggs only fire on temporal_ix
        amount = next(f for f in child.features if f.name == "amount")
        assert agg(parent, child, amount, relationship=rel) is None
