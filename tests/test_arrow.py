"""ArrowExporter's batch decode without a database (#95).

The integration tier (``tests/integration/test_arrow_export.py``) runs the
same path through real ``COPY``; these pin the batching rules on decoded rows.
"""

from __future__ import annotations

import pytest

from featurizer.arrow import ArrowExporter, _TableBuilder

pa = pytest.importorskip("pyarrow")

_ROWS = [(1, 0.5, None), (2, None, "a"), (None, 2.5, "b")]
_OIDS = [20, 701, 25]  # int8, float8, text


def test_batch_values_must_be_positive():
    with pytest.raises(ValueError, match="batch_values"):
        ArrowExporter(batch_values=0)


def test_a_batch_holds_at_least_one_row():
    exporter = ArrowExporter(batch_values=5)
    assert exporter._batch_rows(2) == 2
    assert exporter._batch_rows(600) == 1


def test_batches_become_chunks_of_one_table():
    per_row = _TableBuilder(["a", "b", "c"], _OIDS, pa, numeric_as_float=True)
    for row in _ROWS:
        per_row.add([row])
    whole = _TableBuilder(["a", "b", "c"], _OIDS, pa, numeric_as_float=True)
    whole.add(_ROWS)

    t1, t2 = per_row.table(), whole.table()
    assert t1.column("a").num_chunks == 3 and t2.column("a").num_chunks == 1
    assert t1.equals(t2)
    assert t1.to_pydict() == {
        "a": [1, 2, None],
        "b": [0.5, None, 2.5],
        "c": [None, "a", "b"],
    }


def test_a_value_typed_column_is_not_streamable():
    """numeric kept exact waits for every value; numeric as float does not."""
    assert _TableBuilder(["n"], [1700], pa, numeric_as_float=True).streamable
    assert not _TableBuilder(["n"], [1700], pa, numeric_as_float=False).streamable
    assert not _TableBuilder(["i"], [1186], pa, numeric_as_float=True).streamable
