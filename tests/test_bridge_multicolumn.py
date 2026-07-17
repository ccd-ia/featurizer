"""Contract tests for the ADR-0014 harness extensions (no database).

Covers the MultiColumnBridge shape (compute → materialize DDL/payload →
emit_yaml), the temporal snapshot-sequence keying and its per-window causal
guard, the persist option, and the model-vintage strict check. A fake
connection records the SQL so materialization is provable without PostgreSQL;
the real DB path is covered per family in the integration tier.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pytest

from featurizer.bridge import BridgeComputer, MultiColumnBridge


class FakeCursor:
    """Answers one SELECT from canned rows; records every other statement."""

    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.conn.statements.append(sql)
        if sql.lstrip().lower().startswith("select"):
            self.description = [
                type("D", (), {"name": n})() for n in self.conn.select_names
            ]
            self._rows = self.conn.select_rows

    def executemany(self, sql: str, payload: List[tuple]) -> None:
        self.conn.statements.append(sql)
        self.conn.payloads.append(list(payload))

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, names: List[str], rows: List[tuple]) -> None:
        self.select_names = names
        self.select_rows = rows
        self.statements: List[str] = []
        self.payloads: List[List[tuple]] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


class CountsBridge(MultiColumnBridge):
    """Stub multi-column φ: per-row char counts split into two columns."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(name="counts", value_cols=["vowels", "consonants"], **kw)

    def compute(self, rows, *, fit_rows):
        out: Dict[Any, Dict[str, Any]] = {}
        for row in rows:
            text = str(row.get("body") or "")
            vowels = sum(c in "aeiou" for c in text)
            out[row["doc_id"]] = {
                "vowels": float(vowels),
                "consonants": float(sum(c.isalpha() for c in text) - vowels),
            }
        return out


class PerEntityCountBridge(BridgeComputer):
    """Stub per-entity φ: number of events per owner in the fit slice."""

    def __init__(self) -> None:
        super().__init__(name="evt_count", value_col="evt_count")

    def compute(self, rows, *, fit_rows):
        counts: Dict[Any, float] = {}
        for row in fit_rows:
            counts[row["owner"]] = counts.get(row["owner"], 0.0) + 1.0
        return counts


# --------------------------------------------------------------------------- #
# MultiColumnBridge shape
# --------------------------------------------------------------------------- #


def test_multicolumn_requires_at_least_one_column():
    class Empty(MultiColumnBridge):
        def compute(self, rows, *, fit_rows):
            return {}

    with pytest.raises(ValueError, match="at least one column"):
        Empty(name="empty", value_cols=[])


def test_multicolumn_compute_and_value_tuple():
    bridge = CountsBridge()
    values = bridge.compute([{"doc_id": 1, "body": "abc"}], fit_rows=[])
    assert values == {1: {"vowels": 1.0, "consonants": 2.0}}
    assert bridge._value_tuple(values[1]) == (1.0, 2.0)
    # Missing key / missing column -> NULLs, never KeyError.
    assert bridge._value_tuple(None) == (None, None)
    assert bridge._value_tuple({"vowels": 3.0}) == (3.0, None)


def test_multicolumn_materialize_ddl_and_payload():
    conn = FakeConn(
        names=["doc_id", "owner", "ts", "body"],
        rows=[
            (1, 10, date(2020, 1, 1), "ab"),
            (2, 10, date(2020, 2, 1), "e"),
        ],
    )
    bridge = CountsBridge()
    bridge.materialize(
        conn,
        source_table="docs",
        pk="doc_id",
        carry_cols=["owner", "ts"],
        content_cols=["body"],
        output_table="counts_out",
        causal_col="ts",
        fit_before=date(2020, 6, 1),
    )
    create = next(s for s in conn.statements if s.startswith("create"))
    # Default stays session-temporary; one double-precision column per value_col.
    assert create.startswith("create temp table counts_out")
    assert create.endswith("on commit drop")
    assert "vowels double precision" in create
    assert "consonants double precision" in create
    assert conn.payloads[0] == [
        (1, 10, date(2020, 1, 1), 1.0, 1.0),
        (2, 10, date(2020, 2, 1), 1.0, 0.0),
    ]


def test_multicolumn_emit_yaml_declares_one_variable_per_column():
    bridge = CountsBridge()
    fragment = bridge.emit_yaml(
        output_table="counts_out",
        pk="doc_id",
        parent_alias="owners",
        parent_key="owner",
        fk="owner",
        temporal_ix="ts",
    )
    assert fragment["entity"]["variables"] == {
        "vowels": {"type": "numeric"},
        "consonants": {"type": "numeric"},
    }


def test_multicolumn_categorical_type_flows_to_yaml_and_ddl():
    class MembershipBridge(MultiColumnBridge):
        def __init__(self):
            super().__init__(
                name="community",
                value_cols=["community_id", "modularity"],
                value_types={"community_id": "categorical"},
            )

        def compute(self, rows, *, fit_rows):
            return {r["node"]: {"community_id": "c1", "modularity": 0.5} for r in rows}

    bridge = MembershipBridge()
    fragment = bridge.emit_yaml(
        output_table="t", pk="node", parent_alias="p", parent_key="node", fk="node"
    )
    assert fragment["entity"]["variables"]["community_id"] == {"type": "categorical"}
    assert fragment["entity"]["variables"]["modularity"] == {"type": "numeric"}
    values = bridge.compute([{"node": 1}], fit_rows=[])
    assert bridge._value_ddl_types(values) == ["text", "double precision"]


# --------------------------------------------------------------------------- #
# Regression: the single-column contract is unchanged (additive guarantee)
# --------------------------------------------------------------------------- #


def test_single_column_emit_yaml_unchanged():
    bridge = PerEntityCountBridge()
    fragment = bridge.emit_yaml(
        output_table="bridge_events",
        pk="event_id",
        parent_alias="owners",
        parent_key="owner_id",
        fk="owner_id",
        temporal_ix="ts",
    )
    assert fragment == {
        "entity": {
            "alias": "evt_count",
            "table": "bridge_events",
            "id": "event_id",
            "variables": {"evt_count": {"type": "numeric"}},
            "temporal_ix": "ts",
        },
        "relationship": {
            "parent": {"entity": "owners", "key": "owner_id"},
            "child": {"entity": "evt_count", "key": "owner_id"},
        },
    }


def test_single_column_materialize_ddl_unchanged():
    conn = FakeConn(
        names=["event_id", "owner", "ts"],
        rows=[(1, 10, date(2020, 1, 1))],
    )
    PerEntityCountBridge().materialize(
        conn,
        source_table="events",
        pk="event_id",
        carry_cols=["owner", "ts"],
        output_table="out",
    )
    create = next(s for s in conn.statements if s.startswith("create"))
    assert create == (
        "create temp table out (event_id bigint, owner bigint, ts date, "
        "evt_count double precision) on commit drop"
    )


# --------------------------------------------------------------------------- #
# Temporal snapshot sequences
# --------------------------------------------------------------------------- #

EVENT_ROWS = [
    {"owner": 1, "ts": date(2020, 1, 1)},
    {"owner": 1, "ts": date(2020, 3, 1)},
    {"owner": 2, "ts": date(2020, 3, 1)},
    {"owner": 1, "ts": date(2020, 9, 1)},  # knowable only at the later window
]


def test_compute_snapshots_keys_by_entity_and_as_of():
    bridge = PerEntityCountBridge()
    snaps = bridge.compute_snapshots(
        EVENT_ROWS,
        as_of_dates=[date(2020, 6, 1), date(2020, 12, 1)],
        causal_col="ts",
    )
    # Per-window rebuild: the September event exists only in the later window.
    assert snaps[(1, date(2020, 6, 1))] == 2.0
    assert snaps[(1, date(2020, 12, 1))] == 3.0
    assert snaps[(2, date(2020, 6, 1))] == 1.0
    assert snaps[(2, date(2020, 12, 1))] == 1.0


def test_compute_snapshots_never_sees_the_future():
    class RecordingBridge(PerEntityCountBridge):
        seen: List[List[Dict[str, Any]]] = []

        def compute(self, rows, *, fit_rows):
            self.seen.append(fit_rows)
            return super().compute(rows, fit_rows=fit_rows)

    bridge = RecordingBridge()
    bridge.compute_snapshots(
        EVENT_ROWS, as_of_dates=[date(2020, 6, 1)], causal_col="ts"
    )
    (window,) = bridge.seen
    assert all(r["ts"] <= date(2020, 6, 1) for r in window)


def test_materialize_snapshots_emits_event_stream():
    conn = FakeConn(
        names=["owner", "ts"],
        rows=[(r["owner"], r["ts"]) for r in EVENT_ROWS],
    )
    bridge = PerEntityCountBridge()
    bridge.materialize_snapshots(
        conn,
        source_table="events",
        output_table="snap_out",
        as_of_dates=[date(2020, 6, 1), date(2020, 12, 1)],
        causal_col="ts",
        content_cols=["owner"],
        entity_col="node_id",
        as_of_col="as_of_date",
    )
    create = next(s for s in conn.statements if s.startswith("create"))
    assert create == (
        "create temp table snap_out (node_id bigint, as_of_date date, "
        "evt_count double precision) on commit drop"
    )
    assert conn.payloads[0] == [
        (1, date(2020, 6, 1), 2.0),
        (2, date(2020, 6, 1), 1.0),
        (1, date(2020, 12, 1), 3.0),
        (2, date(2020, 12, 1), 1.0),
    ]


# --------------------------------------------------------------------------- #
# Persist option
# --------------------------------------------------------------------------- #


def test_persist_writes_a_real_table():
    conn = FakeConn(names=["event_id", "owner"], rows=[(1, 10)])
    PerEntityCountBridge().materialize(
        conn,
        source_table="events",
        pk="event_id",
        carry_cols=["owner"],
        output_table="asset_out",
        persist=True,
    )
    create = next(s for s in conn.statements if s.startswith("create"))
    assert create.startswith("create table asset_out (")
    assert "temp" not in create and "on commit drop" not in create


# --------------------------------------------------------------------------- #
# Model vintage (pretrained-model leakage metadata)
# --------------------------------------------------------------------------- #


def test_model_vintage_defaults_to_unknown_and_surfaces_in_metadata():
    bridge = PerEntityCountBridge()
    assert bridge.model_vintage is None
    assert bridge.metadata == {
        "name": "evt_count",
        "value_cols": ["evt_count"],
        "model_vintage": None,
    }


def test_assert_model_vintage_strict_check():
    bridge = PerEntityCountBridge()
    with pytest.raises(ValueError, match="unknown"):
        bridge.assert_model_vintage(date(2020, 6, 1))

    bridge.model_vintage = date(2020, 1, 1)
    bridge.assert_model_vintage(date(2020, 6, 1))  # vintage <= as_of: fine
    with pytest.raises(ValueError, match="trained on data not knowable"):
        bridge.assert_model_vintage(date(2019, 6, 1))
