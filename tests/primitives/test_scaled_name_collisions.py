"""Scaled name-collision guard for the transformer-label fix (DB-free).

Stress-testing against triage-pg's donorschoose schema generated 52,225 features
with 44,844 hash-capped names and ZERO post-63-byte-cap collisions. This test
reproduces the essential shape — a multi-entity graph with several numeric child
measures under the full default aggregation set plus a broad transform set — and
asserts, without a database, that every rendered identifier is ≤ 63 bytes and no
two features collide after PostgreSQL's 63-byte cap. A regression in either the
aggregation or transformer name/label wiring (bug #8) trips this immediately.
"""

import tempfile

import yaml

from featurizer import Featurizer
from featurizer.primitives.abstractions import _truncate_identifier
from featurizer.primitives.aggregations import DEFAULT_AGGREGATIONS

# A broad, planner-safe transform set that forces deep nesting + long names
# (excludes ``in_array``, which needs an argument the planner can't supply).
WIDE_TX = [
    "identity",
    "abs",
    "ln",
    "sqrt",
    "cum_sum",
    "cum_count",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "ema_7",
    "diff",
    "pct_change_1",
    "cusum",
]


def _wide_config() -> dict:
    def child(alias, key):
        return {
            "alias": alias,
            "table": f"s.{alias}",
            "id": None,
            "temporal_ix": "ts",
            "variables": {
                "measurement_value_reading": {"type": "numeric"},
                "secondary_amount_column": {"type": "numeric"},
            },
        }

    return {
        "target": "ego",
        "max_depth": 2,
        "intervals": ["P1M", "P6M", "P12M"],
        "aggregations": list(DEFAULT_AGGREGATIONS),
        "transformations": WIDE_TX,
        "entities": [
            {
                "alias": "ego",
                "table": "s.ego",
                "id": "entity_id",
                "variables": {"a_long_numeric_attribute_name": {"type": "numeric"}},
            },
            child("first_child_stream", "entity_id"),
            child("second_child_stream", "entity_id"),
        ],
        "relationships": [
            {
                "parent": {"entity": "ego", "key": "entity_id"},
                "child": {"entity": "first_child_stream", "key": "entity_id"},
            },
            {
                "parent": {"entity": "ego", "key": "entity_id"},
                "child": {"entity": "second_child_stream", "key": "entity_id"},
            },
        ],
    }


def _write(cfg: dict) -> str:
    h = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, h)
    h.close()
    return h.name


def test_wide_config_has_no_capped_name_collisions():
    f = Featurizer(_write(_wide_config()), validate=False)
    feats = f._plan.target_output_features
    # Sanity: this really is a wide, deeply-nested plan that exercises capping.
    assert len(feats) > 1000, len(feats)
    caps = [_truncate_identifier(x.name.replace('"', "")) for x in feats]
    assert any("~" in c for c in caps), "expected some hash-capped names"

    # The two invariants the label fix guarantees:
    over = [c for c in caps if len(c.encode()) > 63]
    assert not over, f"{len(over)} identifiers exceed 63 bytes, e.g. {over[:3]}"
    collisions = len(caps) - len(set(caps))
    assert collisions == 0, f"{collisions} post-cap short-name collisions"


def test_capped_features_carry_full_labels():
    f = Featurizer(_write(_wide_config()), validate=False)
    capped = [
        x
        for x in f._plan.target_output_features
        if "~" in _truncate_identifier(x.name.replace('"', ""))
    ]
    assert capped, "expected capped features in a wide config"
    for feat in capped[:50]:
        # The manifest maps the capped column back to a longer, readable label.
        assert len(feat.label.encode()) >= len(feat.name.replace('"', "").encode())
