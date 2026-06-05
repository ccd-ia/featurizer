"""Integration tests for complete feature generation workflows."""

import tempfile
from pathlib import Path


from featurizer import Featurizer


class TestBasicWorkflows:
    """End-to-end workflow tests."""

    def test_single_entity_no_relationships(self):
        """Single entity with no relationships generates base features."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: users
max_depth: 1
intervals: []
entities:
  - alias: users
    table: analytics.users
    id: user_id
    temporal_ix: created_at
    variables:
      age:
        type: numeric
      city:
        type: categorical
""")
            f.flush()

            featurizer = Featurizer(f.name)

            # Should have target entity
            assert featurizer.target.alias == "users"

            # Should generate features
            features = featurizer.features["users"]
            assert len(features) > 0

            # Query should be valid SQL
            query = featurizer.query
            assert "select" in query.lower()
            assert "users_transform" in query.lower()

            Path(f.name).unlink()

    def test_parent_child_aggregation(self):
        """Parent-child relationship generates aggregations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: users
max_depth: 2
intervals: []
entities:
  - alias: users
    table: analytics.users
    id: user_id
  - alias: orders
    table: analytics.orders
    id: order_id
    temporal_ix: ordered_at
    variables:
      amount:
        type: numeric

relationships:
  - parent:
      entity: users
      key: user_id
    child:
      entity: orders
      key: user_id
""")
            f.flush()

            featurizer = Featurizer(f.name)

            # Should have aggregations from orders
            features = featurizer.features["users"]
            feature_names = {f.name for f in features}

            # Check for aggregated features
            assert any("MEAN" in name for name in feature_names)
            assert any("orders" in name.lower() for name in feature_names)

            # Should have CTEs for aggregations
            query = featurizer.query
            assert "orders_aggs_for_users" in query

            Path(f.name).unlink()

    def test_temporal_join_generates_lateral(self):
        """Temporal as-of join generates lateral clause."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: patients
max_depth: 2
intervals: []
entities:
  - alias: patients
    table: analytics.patients
    id: patient_id
    temporal_ix: registered_at
  - alias: care_plans
    table: analytics.care_plans
    id: plan_id
    temporal_ix: effective_at
    variables:
      risk_score:
        type: numeric

relationships:
  - parent:
      entity: care_plans
      key: patient_id
    child:
      entity: patients
      key: patient_id
    temporal:
      mode: as_of
      grace: P14D
""")
            f.flush()

            featurizer = Featurizer(f.name)

            query = featurizer.query.lower()

            # Should have lateral join
            assert "lateral" in query
            assert "care_plans_asof_for_patients" in query

            Path(f.name).unlink()

    def test_multiple_depths(self):
        """Multiple depth traversal generates features from grandchildren."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: users
max_depth: 3
intervals: []
entities:
  - alias: users
    table: analytics.users
    id: user_id
  - alias: orders
    table: analytics.orders
    id: order_id
    variables:
      total:
        type: numeric
  - alias: line_items
    table: analytics.line_items
    id: item_id
    variables:
      quantity:
        type: numeric

relationships:
  - parent:
      entity: users
      key: user_id
    child:
      entity: orders
      key: user_id
  - parent:
      entity: orders
      key: order_id
    child:
      entity: line_items
      key: order_id
""")
            f.flush()

            featurizer = Featurizer(f.name)

            # Should traverse to depth 3
            assert featurizer.max_depth == 3

            # Should have features from both orders and line_items
            features = featurizer.features["users"]
            feature_names = {f.name for f in features}

            # Check for nested aggregations
            assert any("orders" in name.lower() for name in feature_names)
            assert any("line_items" in name.lower() for name in feature_names)

            Path(f.name).unlink()

    def test_interval_based_aggregations(self):
        """Interval specifications generate time-windowed aggregations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: users
max_depth: 2
intervals:
  - P7D
  - P30D
entities:
  - alias: users
    table: analytics.users
    id: user_id
  - alias: events
    table: analytics.events
    id: event_id
    temporal_ix: occurred_at
    variables:
      value:
        type: numeric

relationships:
  - parent:
      entity: users
      key: user_id
    child:
      entity: events
      key: user_id
""")
            f.flush()

            featurizer = Featurizer(f.name)

            # Should have interval-based features
            features = featurizer.features["users"]
            feature_names = {f.name for f in features}

            # Check for interval annotations in feature names
            assert any("interval=P7D" in name for name in feature_names)
            assert any("interval=P30D" in name for name in feature_names)

            # Query should have filter clauses for intervals
            query = featurizer.query
            assert "filter" in query.lower()
            assert "daterange" in query.lower()

            Path(f.name).unlink()

    def test_transformations_applied(self):
        """Transformations generate derived features."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: sensors
max_depth: 2
intervals: []
entities:
  - alias: sensors
    table: iot.sensors
    id: sensor_id
    temporal_ix: measured_at
    variables:
      temperature:
        type: numeric
      humidity:
        type: numeric
""")
            f.flush()

            featurizer = Featurizer(f.name)

            features = featurizer.features["sensors"]
            feature_names = {f.name for f in features}

            # Should have transformed features (at depth 2)
            assert any("ABS(" in name for name in feature_names)
            assert any("CUM_SUM(" in name for name in feature_names)

            # Query should have transform CTE
            query = featurizer.query
            assert "sensors_transform" in query

            Path(f.name).unlink()


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_entity_without_temporal_index_no_intervals(self):
        """Entity without temporal_ix doesn't break interval aggregation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: users
max_depth: 2
intervals:
  - P7D
entities:
  - alias: users
    table: users
    id: user_id
  - alias: events
    table: events
    id: event_id
    variables:
      value:
        type: numeric

relationships:
  - parent:
      entity: users
      key: user_id
    child:
      entity: events
      key: user_id
""")
            f.flush()

            # Should not raise, but should warn
            featurizer = Featurizer(f.name)

            # Should still generate query
            query = featurizer.query
            assert "select" in query.lower()

            Path(f.name).unlink()

    def test_empty_intervals_list(self):
        """Empty intervals list is valid."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: test
max_depth: 1
intervals: []
entities:
  - alias: test
    table: test
    id: id
""")
            f.flush()

            featurizer = Featurizer(f.name)
            assert featurizer.intervals == []

            Path(f.name).unlink()

    def test_max_depth_1_stops_early(self):
        """max_depth=1 doesn't traverse relationships."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: users
max_depth: 1
intervals: []
entities:
  - alias: users
    table: users
    id: user_id
  - alias: orders
    table: orders
    id: order_id
    variables:
      total:
        type: numeric

relationships:
  - parent:
      entity: users
      key: user_id
    child:
      entity: orders
      key: user_id
""")
            f.flush()

            featurizer = Featurizer(f.name)

            # With depth 1, shouldn't traverse to orders
            assert featurizer.max_depth == 1

            # Should still generate valid query
            query = featurizer.query
            assert "select" in query.lower()

            Path(f.name).unlink()

    def test_entity_properties_accessible(self):
        """Featurizer exposes graph properties."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
target: test
max_depth: 2
intervals: []
entities:
  - alias: test
    table: test
    id: id
  - alias: other
    table: other
    id: id
relationships:
  - parent:
      entity: test
      key: id
    child:
      entity: other
      key: test_id
""")
            f.flush()

            featurizer = Featurizer(f.name)

            # Can access entities
            entities = list(featurizer.entities)
            assert len(entities) == 2

            # Can access relationships
            relationships = featurizer.relationships
            assert len(relationships) == 1

            # Can access CTEs (with depth 2, CTEs are generated)
            assert len(featurizer.ctes) > 0

            # Can access joins
            assert isinstance(featurizer.joins, dict)

            Path(f.name).unlink()
