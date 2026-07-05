"""Lineage, source-alias, interval, and generated-description contract of the
feature manifest (v0.5.0 additions). DB-free."""

import tempfile

import yaml

from featurizer import Featurizer


def _featurizer(config: dict) -> Featurizer:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return Featurizer(handle.name)


def _config() -> dict:
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": ["P1M"],
        "aggregations": ["sum"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "customers",
                "table": "customers",
                "id": "customer_id",
                "temporal_ix": "created_at",
                "variables": {
                    "score": {"type": "numeric"},
                    "segment": {
                        "type": "categorical",
                        "role": "categorical",
                        "vocabulary": ["basic", "premium"],
                    },
                },
            },
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {"amount": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }


def _by_column(featurizer: Featurizer) -> dict:
    return {entry.column: entry for entry in featurizer.feature_manifest}


class TestLineageFields:
    def test_variable_entry(self):
        entry = _by_column(_featurizer(_config()))["score"]
        assert entry.kind == "variable"
        assert entry.depth == 0
        assert entry.parents == []
        assert entry.source_alias is None
        assert entry.interval is None
        assert "score" in entry.description

    def test_windowed_aggregation_entry(self):
        entries = _by_column(_featurizer(_config()))
        entry = entries["SUM(orders.amount|interval=P1M)"]
        assert entry.kind == "derived"
        assert entry.depth >= 1
        assert entry.parents == ["amount"]
        assert entry.source_alias == "orders"
        assert entry.interval == "P1M"

    def test_unwindowed_aggregation_has_no_interval(self):
        entry = _by_column(_featurizer(_config()))["SUM(orders.amount)"]
        assert entry.interval is None
        assert entry.source_alias == "orders"

    def test_one_hot_entry(self):
        entries = _by_column(_featurizer(_config()))
        entry = entries["customers.segment=premium"]
        assert entry.kind == "one_hot"
        assert entry.source_column == "segment"
        assert entry.value == "premium"
        assert "premium" in entry.description
        assert "one-hot" in entry.description

    def test_named_relationship_source_alias(self):
        config = _config()
        config["relationships"] = [
            {
                "name": "purchases",
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "buyer_id"},
            },
            {
                "name": "sales",
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "seller_id"},
            },
        ]
        entries = _by_column(_featurizer(config))
        assert entries["SUM(purchases.amount|interval=P1M)"].source_alias == "purchases"
        assert entries["SUM(sales.amount|interval=P1M)"].source_alias == "sales"


class TestGeneratedDescriptions:
    def test_aggregation_description_mentions_argument_and_window(self):
        entries = _by_column(_featurizer(_config()))
        description = entries["SUM(orders.amount|interval=P1M)"].description
        assert "orders.amount" in description
        assert "P1M" in description
        # Comes from the primitive docs, not the generic fallback.
        assert not description.startswith("Derived feature:")

    def test_unknown_op_falls_back_loudly(self):
        from featurizer.manifest import _describe, _parse_label

        label = "FRQZ(orders.amount)"
        parsed = _parse_label(label)

        class _Fake:
            name = label
            entity = None
            description = "a feature"
            stack_depth = 1

        text = _describe("derived", label, "customers", parsed, _Fake(), {})
        assert text == f"Derived feature: {label}"


class TestTruncatedNamesKeepFullLabel:
    def test_long_name_truncated_column_full_label(self):
        config = _config()
        long_column = "a_very_long_column_name_that_overflows_postgres" + "_x" * 20
        config["entities"][1]["variables"] = {long_column: {"type": "numeric"}}
        entries = _featurizer(config).feature_manifest
        truncated = [e for e in entries if e.truncated]
        assert truncated, "expected at least one truncated entry"
        for entry in truncated:
            assert len(entry.column.encode()) <= 63
            assert long_column in entry.label


class TestManifestDataFrame:
    def test_dataframe_columns_and_parents_rendering(self):
        df = _featurizer(_config()).manifest_dataframe()
        assert list(df.columns) == [
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
        row = df[df["column"] == "SUM(orders.amount|interval=P1M)"].iloc[0]
        assert row["parents"] == "amount"  # list rendered comma-joined
