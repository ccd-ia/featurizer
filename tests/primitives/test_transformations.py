"""Transformer tests cover object copying semantics and composite outputs."""

from featurizer.primitives.abstractions import Entity
from featurizer.primitives.utils import get_transformers


def _make_numeric_entity() -> Entity:
    return Entity(
        alias="visits",
        table="analytics.visits",
        id="visit_id",
        temporal_ix="visited_at",
        variables={"duration_minutes": {"type": "numeric"}},
    )


def _make_temporal_entity() -> Entity:
    return Entity(
        alias="observations",
        table="analytics.observations",
        id="observation_id",
        temporal_ix="observed_at",
        variables={"observed_at": {"type": "timestamp"}},
    )


def _get_feature(entity: Entity, name: str):
    return next(ft for ft in entity.features if ft.name == name)


def test_abs_transformer_creates_distinct_feature():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["abs"])["abs"]

    result = transformer(entity, feature)

    assert result is not feature
    assert result.definition.strip() == "abs(duration_minutes)"


def test_identity_transformer_preserves_instance():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["identity"])["identity"]

    result = transformer(entity, feature)

    assert result is feature
    assert len({result, feature}) == 1  # Hashability regression guard.


def test_cyclical_transformer_returns_pair():
    entity = _make_temporal_entity()
    feature = _get_feature(entity, "observed_at")
    transformer = get_transformers(["cyclic_month"])["cyclic_month"]

    result = transformer(entity, feature)

    assert isinstance(result, list)
    assert len(result) == 2
    for sub_feature in result:
        assert sub_feature.entity is entity


def test_cum_sum_transformer_creates_window_feature():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["cum_sum"])["cum_sum"]

    result = transformer(entity, feature)

    assert result is not feature
    assert "over (partition by" in result.definition.lower()


def test_lag_transformer_builds_temporal_window():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["lag_3"])["lag_3"]

    result = transformer(entity, feature)

    assert result is not None
    assert f"lag({feature.name}, 3)" in result.definition
    assert "order by visited_at" in result.definition.lower()


def test_rolling_mean_transformer_uses_frame():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["rolling_mean_3"])["rolling_mean_3"]

    result = transformer(entity, feature)

    assert result is not None
    assert "avg" in result.definition.lower()
    assert "rows between 2 preceding and current row" in result.definition.lower()


def _make_idless_entity() -> Entity:
    """An event entity with no primary id (``id: ~`` in YAML)."""
    return Entity(
        alias="measurements",
        table="analytics.measurements",
        temporal_ix="measured_at",
        variables={"reading": {"type": "numeric"}},
    )


def test_window_transformers_skip_entity_without_id():
    """Window functions need a partition key (the entity id); an id-less entity
    must make them return None (skip), not raise AttributeError.

    Regression for the ``parent.id.name`` dereference that crashed before the
    None-check in both ``WindowFunctionTransformer`` and
    ``DistributionTransformer``.
    """
    entity = _make_idless_entity()
    assert entity.id is None
    feature = _get_feature(entity, "reading")

    # cum_sum -> WindowFunctionTransformer; percent_rank -> DistributionTransformer.
    for name in ("cum_sum", "percent_rank"):
        transformer = get_transformers([name])[name]
        assert transformer(entity, feature) is None


def test_rolling_median_transformer_uses_percentile():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["rolling_median_7"])["rolling_median_7"]

    result = transformer(entity, feature)

    assert result is not None
    definition = result.definition.lower()
    assert "percentile_cont(0.5)" in definition
    # PostgreSQL forbids OVER on percentile_cont, so the rolling window is a
    # correlated subquery: the 7 most-recent rows per ego, by the temporal index.
    assert "limit 7" in definition
    assert "_ego." in definition


def test_rolling_iqr_transformer_differs_percentiles():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["rolling_iqr_7"])["rolling_iqr_7"]

    result = transformer(entity, feature)

    assert result is not None
    assert "percentile_cont(0.75)" in result.definition.lower()
    assert "percentile_cont(0.25)" in result.definition.lower()


def test_ema_transformer_builds_weighted_window():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["ema_7"])["ema_7"]

    result = transformer(entity, feature)

    assert result is not None
    assert "sum(" in result.definition.lower()
    assert "exp(" in result.definition.lower()
    assert "nullif" in result.definition.lower()


def test_holt_winters_trend_uses_regression():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["holt_winters_trend_7"])["holt_winters_trend_7"]

    result = transformer(entity, feature)

    assert result is not None
    assert "regr_slope" in result.definition.lower()
    assert "rows between 6 preceding and current row" in result.definition.lower()


def test_pct_change_transformer_computes_ratio():
    entity = _make_numeric_entity()
    feature = _get_feature(entity, "duration_minutes")
    transformer = get_transformers(["pct_change_1"])["pct_change_1"]

    result = transformer(entity, feature)

    assert result is not None
    assert "case" in result.definition.lower()
    assert f"({feature.name} - lag" in result.definition
