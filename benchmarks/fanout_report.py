"""DB-free fan-out report for the child-stream pre-aggregation tier (plan P3).

The full-aggregator config emits one companion CTE per ``(family, column,
interval)``, each an independent scan of the child ``<child>_transform`` stream,
all merge-joined. That fan-out — not any single slow primitive — dominates
full-cohort materialization on real data (see
``specs/reduce-child-stream-fanout-p3.html``). This module measures it from the
generated SQL alone (no database): the number of companion-CTE definitions and
the total references to any child ``*_transform`` stream.

Run: ``python -m benchmarks.fanout_report`` (or ``just bench-fanout``). It reports
a repo-internal synthetic config that reproduces the fan-out shape, so the number
is reproducible without the external triage datasets. Phase 3's signature-merge
must drive ``companion_ctes`` down; the guard in ``tests/test_fanout_budget.py``
pins the current value so a regression trips loudly.
"""

from __future__ import annotations

import re
import tempfile
from typing import Any, Dict, List

import yaml

from featurizer import Featurizer
from featurizer.primitives.aggregations import DEFAULT_AGGREGATIONS


def synthetic_config(
    *,
    n_categorical: int = 3,
    n_numeric: int = 1,
    intervals: List[str] | None = None,
) -> Dict[str, Any]:
    """A config that reproduces the fan-out shape of a real multi-categorical
    event stream under the full default aggregator set — no external data."""
    intervals = intervals or ["P1M", "P3M", "P6M"]
    cat_cols = {f"cat_{i}": {"type": "categorical"} for i in range(n_categorical)}
    num_cols = {f"num_{i}": {"type": "numeric"} for i in range(n_numeric)}
    return {
        "target": "entities",
        "max_depth": 2,
        "intervals": intervals,
        "aggregations": list(DEFAULT_AGGREGATIONS),
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "entities",
                "id": "entity_id",
                "table": "entities",
                "variables": {},
            },
            {
                "alias": "events",
                "id": None,
                "table": "events",
                "temporal_ix": "ts",
                "variables": {**cat_cols, **num_cols},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "entities", "key": "entity_id"},
                "child": {"entity": "events", "key": "entity_id"},
                "temporal": {"mode": "as_of"},
            }
        ],
    }


def fanout(config: Dict[str, Any]) -> Dict[str, int]:
    """Fan-out metrics for one config, from the generated SQL only (no DB)."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(config, handle)
    handle.close()
    featurizer = Featurizer(handle.name, validate=False)
    sql = featurizer.query
    transforms = {
        t
        for t in set(re.findall(r"(\w+_transform)\b", sql))
        if not t.startswith("as_of")
    }
    return {
        "features": len(featurizer._plan.target_output_features),
        "total_ctes": sql.count(" as ("),
        "companion_ctes": len(re.findall(r"_preaggs_for_\w+ as \(", sql)),
        "child_stream_refs": sum(sql.count(t) for t in transforms),
    }


def main() -> None:
    config = synthetic_config()
    m = fanout(config)
    print(
        "fan-out report — synthetic 3-categorical / 1-numeric child, 3 intervals, all-agg"
    )
    print("-" * 68)
    print(f"  output features       : {m['features']}")
    print(f"  total CTEs            : {m['total_ctes']}")
    print(f"  companion CTEs (P3 ↓) : {m['companion_ctes']}")
    print(f"  child-stream refs     : {m['child_stream_refs']}")
    print("-" * 68)
    print(
        "companion CTEs is the fan-out P3 collapses (one per (family,column,interval) today)."
    )


if __name__ == "__main__":
    main()
