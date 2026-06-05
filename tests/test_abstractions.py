"""Tests for core abstractions: Entity, Relationship, Feature, ERGraph."""

from featurizer.primitives.abstractions import (
    Entity,
    ERGraph,
    Feature,
    Id,
    Key,
    Relationship,
    Variable,
)


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation_minimal(self):
        """Entity can be created with minimal args."""
        entity = Entity(alias="test", table="test.table", id="test_id")

        assert entity.alias == "test"
        assert entity.table == "test.table"
        assert entity.id is not None
        assert entity.id.name == "test_id"
        assert entity.spatial_ix is None
        assert entity.temporal_ix is None
        assert len(entity.keys) == 0

    def test_entity_with_no_id(self):
        """Entity can have no ID (id=None)."""
        entity = Entity(alias="test", table="test.table", id=None)

        assert entity.id is None
        assert len(entity.indexes) == 0

    def test_entity_with_temporal_index(self):
        """Entity with temporal_ix creates Id feature."""
        entity = Entity(
            alias="events",
            table="analytics.events",
            id="event_id",
            temporal_ix="occurred_at",
        )

        assert entity.temporal_ix is not None
        assert entity.temporal_ix.name == "occurred_at"
        assert entity.temporal_ix in entity.features

    def test_entity_with_spatial_index(self):
        """Entity with spatial_ix creates Id feature."""
        entity = Entity(
            alias="locations",
            table="geo.locations",
            id="loc_id",
            spatial_ix="coordinates",
        )

        assert entity.spatial_ix is not None
        assert entity.spatial_ix.name == "coordinates"
        assert entity.spatial_ix in entity.features

    def test_entity_with_variables(self):
        """Entity variables become Variable features."""
        entity = Entity(
            alias="users",
            table="users",
            id="user_id",
            variables={"age": {"type": "numeric"}, "name": {"type": "categorical"}},
        )

        variable_names = {f.name for f in entity.features if isinstance(f, Variable)}
        assert "age" in variable_names
        assert "name" in variable_names

    def test_entity_indexes_property(self):
        """indexes property returns all non-None indexes."""
        entity = Entity(
            alias="events", table="events", id="id", temporal_ix="ts", spatial_ix="loc"
        )

        indexes = entity.indexes
        assert len(indexes) == 3
        assert all(isinstance(idx, Id) for idx in indexes)

    def test_entity_repr(self):
        """Entity repr shows alias."""
        entity = Entity(alias="test", table="test.table", id="test_id")
        assert repr(entity) == "Entity(test)"

    def test_entity_info(self):
        """info() returns formatted string with variables."""
        entity = Entity(
            alias="users",
            table="users",
            id="user_id",
            variables={"age": {"type": "numeric"}},
        )

        info = entity.info()
        assert "Users" in info
        assert "users" in info
        assert "age" in info

    def test_add_key(self):
        """add_key appends unique keys."""
        entity = Entity(alias="child", table="child", id="id")
        key1 = Key(name="parent_id", entity=entity)
        key2 = Key(name="other_id", entity=entity)

        entity.add_key(key1)
        assert key1 in entity.keys

        entity.add_key(key2)
        assert key2 in entity.keys
        assert len(entity.keys) == 2

        # Adding duplicate doesn't increase count
        entity.add_key(key1)
        assert len(entity.keys) == 2

    def test_add_features(self):
        """add_features appends unique features."""
        entity = Entity(alias="test", table="test", id="id")
        feat1 = Feature(name="f1", type="numeric", entity=entity)
        feat2 = Feature(name="f2", type="numeric", entity=entity)

        initial_count = len(entity.features)

        entity.add_features([feat1, feat2])
        assert feat1 in entity.features
        assert feat2 in entity.features
        assert len(entity.features) == initial_count + 2

        # Adding duplicates doesn't increase count
        entity.add_features([feat1])
        assert len(entity.features) == initial_count + 2


class TestRelationship:
    """Tests for Relationship class."""

    def test_relationship_creation(self):
        """Relationship links parent and child entities."""
        parent = Entity(alias="parent", table="parent", id="id")
        child = Entity(alias="child", table="child", id="id")

        rel = Relationship(
            parent=parent, child=child, parent_key="id", child_key="parent_id"
        )

        assert rel.parent is parent
        assert rel.child is child
        assert rel.parent_key == "id"
        assert rel.child_key == "parent_id"

    def test_relationship_with_temporal_mode(self):
        """Relationship can have temporal settings."""
        parent = Entity(alias="p", table="p", id="id", temporal_ix="ts")
        child = Entity(alias="c", table="c", id="id", temporal_ix="ts")

        rel = Relationship(
            parent=parent,
            child=child,
            parent_key="id",
            child_key="p_id",
            temporal_mode="as_of",
            temporal_grace="P7D",
            temporal_child_field="custom_ts",
        )

        assert rel.temporal_mode == "as_of"
        assert rel.temporal_grace == "P7D"
        assert rel.temporal_child_field == "custom_ts"

    def test_relationship_repr(self):
        """Relationship repr shows connection."""
        parent = Entity(alias="parent", table="parent", id="id")
        child = Entity(alias="child", table="child", id="id")

        rel = Relationship(
            parent=parent, child=child, parent_key="pid", child_key="cid"
        )

        repr_str = repr(rel)
        assert "parent" in repr_str
        assert "child" in repr_str
        assert "pid" in repr_str
        assert "cid" in repr_str
        assert "->" in repr_str

    def test_relationship_equality(self):
        """Relationships are equal if all fields match."""
        p1 = Entity(alias="p", table="p", id="id")
        c1 = Entity(alias="c", table="c", id="id")

        rel1 = Relationship(p1, c1, "pk", "ck", "as_of", "P1D", "ts")
        rel2 = Relationship(p1, c1, "pk", "ck", "as_of", "P1D", "ts")

        assert rel1 == rel2

    def test_relationship_inequality(self):
        """Relationships are not equal if fields differ."""
        p1 = Entity(alias="p", table="p", id="id")
        c1 = Entity(alias="c", table="c", id="id")

        rel1 = Relationship(p1, c1, "pk", "ck")
        rel2 = Relationship(p1, c1, "pk", "different")

        assert rel1 != rel2

    def test_relationship_hash(self):
        """Relationships are hashable."""
        parent = Entity(alias="parent", table="parent", id="id")
        child = Entity(alias="child", table="child", id="id")

        rel = Relationship(parent, child, "pk", "ck")

        # Can be added to set
        rel_set = {rel}
        assert rel in rel_set

    def test_relationship_contains_entity(self):
        """__contains__ checks if entity is parent or child."""
        parent = Entity(alias="parent", table="parent", id="id")
        child = Entity(alias="child", table="child", id="id")
        other = Entity(alias="other", table="other", id="id")

        rel = Relationship(parent, child, "pk", "ck")

        assert parent in rel
        assert child in rel
        assert other not in rel

    def test_relationship_is_backward(self):
        """is_backward checks parent->child direction."""
        parent = Entity(alias="parent", table="parent", id="id")
        child = Entity(alias="child", table="child", id="id")

        rel = Relationship(parent, child, "pk", "ck")

        assert rel.is_backward(parent, child) is True
        assert rel.is_backward(child, parent) is False

    def test_relationship_is_forward(self):
        """is_forward checks child->parent direction."""
        parent = Entity(alias="parent", table="parent", id="id")
        child = Entity(alias="child", table="child", id="id")

        rel = Relationship(parent, child, "pk", "ck")

        assert rel.is_forward(child, parent) is True
        assert rel.is_forward(parent, child) is False


class TestFeature:
    """Tests for Feature class."""

    def test_feature_creation(self):
        """Feature can be created with name and type."""
        feat = Feature(name="test_feature", type="numeric")

        assert feat.name == "test_feature"
        assert feat.type == "numeric"

    def test_feature_with_definition(self):
        """Feature can have SQL definition."""
        feat = Feature(name="derived", type="numeric", definition="original * 2")

        assert feat.definition == "original * 2"

    def test_feature_query_property(self):
        """query property formats as SQL."""
        feat = Feature(name="test", type="numeric", definition="value")

        query = feat.query
        assert "test" in query
        assert "value" in query
        assert " as " in query

    def test_feature_short_name_under_limit(self):
        """short_name returns name if under 63 chars."""
        feat = Feature(name="short", type="numeric")
        assert feat.short_name == "short"

    def test_feature_short_name_over_limit(self):
        """short_name returns hash if over 63 chars."""
        long_name = "a" * 64
        feat = Feature(name=long_name, type="numeric")

        assert feat.short_name != long_name
        assert isinstance(feat.short_name, int)

    def test_feature_equality_by_name(self):
        """Features are equal if names match."""
        f1 = Feature(name="test", type="numeric")
        f2 = Feature(name="test", type="categorical")

        assert f1 == f2

    def test_feature_inequality(self):
        """Features are not equal if names differ."""
        f1 = Feature(name="test1", type="numeric")
        f2 = Feature(name="test2", type="numeric")

        assert f1 != f2

    def test_feature_repr(self):
        """Feature repr shows name without quotes."""
        feat = Feature(name='"quoted"', type="numeric")
        assert "quoted" in repr(feat)
        assert '"' not in repr(feat)

    def test_feature_hash(self):
        """Features are hashable."""
        feat = Feature(name="test", type="numeric")

        # Can be added to set
        feat_set = {feat}
        assert feat in feat_set

    def test_variable_is_feature(self):
        """Variable is a Feature subclass."""
        entity = Entity(alias="test", table="test", id="id")
        var = Variable(name="age", type="numeric", entity=entity)

        assert isinstance(var, Feature)
        assert var.name == "age"
        assert var.type == "numeric"
        assert var.definition == "age"
        assert var.stack_depth == 0

    def test_id_is_feature(self):
        """Id is a Feature subclass."""
        entity = Entity(alias="test", table="test", id="test_id")
        id_feature = Id(name="test_id", entity=entity)

        assert isinstance(id_feature, Feature)
        assert id_feature.type == "index"
        assert id_feature.definition == "test_id"

    def test_key_is_feature(self):
        """Key is a Feature subclass."""
        entity = Entity(alias="test", table="test", id="id")
        key = Key(name="parent_id", entity=entity)

        assert isinstance(key, Feature)
        assert key.type == "key"
        assert key.definition == "parent_id"


class TestERGraph:
    """Tests for ERGraph class."""

    def test_ergraph_creation(self):
        """ERGraph constructs from entity and relationship dicts."""
        entities = [
            {"alias": "parent", "table": "parent", "id": "id"},
            {"alias": "child", "table": "child", "id": "id"},
        ]
        relationships = [
            {
                "parent": {"entity": "parent", "key": "id"},
                "child": {"entity": "child", "key": "parent_id"},
            }
        ]

        graph = ERGraph(entities, relationships)

        assert "parent" in graph.entities
        assert "child" in graph.entities
        assert len(graph.relationships) == 1

    def test_ergraph_with_no_relationships(self):
        """ERGraph handles None relationships."""
        entities = [{"alias": "single", "table": "single", "id": "id"}]

        graph = ERGraph(entities, None)

        assert graph.relationships == []

    def test_ergraph_adds_keys_to_children(self):
        """ERGraph automatically adds foreign keys to child entities."""
        entities = [
            {"alias": "parent", "table": "parent", "id": "id"},
            {"alias": "child", "table": "child", "id": "id"},
        ]
        relationships = [
            {
                "parent": {"entity": "parent", "key": "id"},
                "child": {"entity": "child", "key": "parent_id"},
            }
        ]

        graph = ERGraph(entities, relationships)

        child = graph.entities["child"]
        key_names = {k.name for k in child.keys}
        assert "parent_id" in key_names

    def test_get_backward_entities(self):
        """get_backward_entities returns children of an entity."""
        entities = [
            {"alias": "parent", "table": "parent", "id": "id"},
            {"alias": "child", "table": "child", "id": "id"},
        ]
        relationships = [
            {
                "parent": {"entity": "parent", "key": "id"},
                "child": {"entity": "child", "key": "parent_id"},
            }
        ]

        graph = ERGraph(entities, relationships)
        parent = graph.entities["parent"]

        backward = graph.get_backward_entities(parent)
        backward_aliases = {e.alias for e in backward}

        assert "child" in backward_aliases

    def test_get_forward_entities(self):
        """get_forward_entities returns parents of an entity."""
        entities = [
            {"alias": "parent", "table": "parent", "id": "id"},
            {"alias": "child", "table": "child", "id": "id"},
        ]
        relationships = [
            {
                "parent": {"entity": "parent", "key": "id"},
                "child": {"entity": "child", "key": "parent_id"},
            }
        ]

        graph = ERGraph(entities, relationships)
        child = graph.entities["child"]

        forward = graph.get_forward_entities(child)
        forward_aliases = {e.alias for e in forward}

        assert "parent" in forward_aliases

    def test_get_backward_relationships(self):
        """get_backward_relationships returns relationships where entity is parent."""
        entities = [
            {"alias": "parent", "table": "parent", "id": "id"},
            {"alias": "child", "table": "child", "id": "id"},
        ]
        relationships = [
            {
                "parent": {"entity": "parent", "key": "id"},
                "child": {"entity": "child", "key": "parent_id"},
            }
        ]

        graph = ERGraph(entities, relationships)
        parent = graph.entities["parent"]

        backward_rels = graph.get_backward_relationships(parent)

        assert len(backward_rels) == 1
        rel = list(backward_rels)[0]
        assert rel.parent is parent

    def test_get_forward_relationships(self):
        """get_forward_relationships returns relationships where entity is child."""
        entities = [
            {"alias": "parent", "table": "parent", "id": "id"},
            {"alias": "child", "table": "child", "id": "id"},
        ]
        relationships = [
            {
                "parent": {"entity": "parent", "key": "id"},
                "child": {"entity": "child", "key": "parent_id"},
            }
        ]

        graph = ERGraph(entities, relationships)
        child = graph.entities["child"]

        forward_rels = graph.get_forward_relationships(child)

        assert len(forward_rels) == 1
        rel = list(forward_rels)[0]
        assert rel.child is child
