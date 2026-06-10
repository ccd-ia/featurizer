"""Tests for the Text Path-1 lexical transformers (pure-SQL over text columns)."""

from featurizer.primitives.abstractions import Entity
from featurizer.primitives.utils import get_transformers, list_transformations

LEXICAL = [
    "num_words",
    "num_sentences",
    "avg_word_length",
    "caps_ratio",
    "digit_ratio",
    "punct_ratio",
    "exclamation_count",
    "question_count",
    "unique_word_ratio",
]


def _text_entity() -> Entity:
    return Entity(
        alias="posts",
        table="posts",
        id="post_id",
        temporal_ix="created_at",
        variables={"body": {"type": "text"}},
    )


def _numeric_entity() -> Entity:
    return Entity(
        alias="visits",
        table="visits",
        id="visit_id",
        temporal_ix="visited_at",
        variables={"duration_minutes": {"type": "numeric"}},
    )


def _feature(entity: Entity, name: str):
    return next(ft for ft in entity.features if ft.name == name)


def test_lexical_transformers_are_registered():
    available = set(list_transformations())
    for name in LEXICAL:
        assert name in available, f"{name} should be registered"


def test_lexical_transformers_emit_numeric_feature_over_text():
    entity = _text_entity()
    body = _feature(entity, "body")
    transformers = get_transformers(LEXICAL)
    for name, transformer in transformers.items():
        result = transformer(entity, body)
        assert result is not body, f"{name} must return a new feature"
        assert result.type == "numeric", f"{name} must be numeric"
        assert "body" in result.definition, f"{name} must reference the column"
        assert result.name == f'"{name.upper()}(posts.body)"'


def test_lexical_transformers_skip_non_text_features():
    """Applied to a numeric column, a text transformer passes the input through."""
    entity = _numeric_entity()
    duration = _feature(entity, "duration_minutes")
    for name, transformer in get_transformers(LEXICAL).items():
        result = transformer(entity, duration)
        assert result is duration, f"{name} must not transform a numeric column"


def test_num_chars_uses_char_length_not_typo():
    """Regression: num_chars previously emitted the invalid `char_lenght`."""
    entity = _text_entity()
    body = _feature(entity, "body")
    result = get_transformers(["num_chars"])["num_chars"](entity, body)
    assert "char_length(body)" in result.definition
    assert "char_lenght" not in result.definition


def test_word_based_transformers_are_null_safe():
    """Word/ratio transformers wrap the column in coalesce so NULL text is safe."""
    entity = _text_entity()
    body = _feature(entity, "body")
    for name in ("num_words", "avg_word_length", "unique_word_ratio", "caps_ratio"):
        result = get_transformers([name])[name](entity, body)
        assert "coalesce(body" in result.definition
