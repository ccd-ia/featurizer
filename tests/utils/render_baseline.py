"""Configs and digests for the "absent means byte-identical" guarantee (issue #10).

``as_of_dates: {id_column: …}`` is additive under ADR-0015: a config without
the block must render exactly the SQL it rendered before the key existed. "Exactly"
is checked, not asserted — ``tests/fixtures/render_baseline_pre_cohort.json``
holds SHA-256 digests and ``tests/test_cohort_pairs.py`` compares against them
byte for byte.

Lineage of the digests, because a digest cannot show what moved it. They were
captured from master at ``3dbbfd8``, the commit before the change. Master at
``de03142`` still reproduced every one of them. Issue #29 then quoted the column
each aggregation wraps, which moves the rendered bytes of every config that
aggregates a declared column, so they were captured again from ``de03142`` plus
that fix (``requoted_by`` in the file). The move is quoting only: with every
double quote stripped, ``de03142`` and the fix render identical text for every
part of every case, and ``n_groups`` did not change.

Three renderers read the target's base relation, so all three are digested: the
monolithic query, the column-group queries, and the temp-table preamble.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
BASELINE = REPO / "tests" / "fixtures" / "render_baseline_pre_cohort.json"

# Shipped configs that no open change rewrites. Example 02 is left out on
# purpose: its config is being re-oriented (issue #9), so its digest would move
# for a reason that has nothing to do with this guarantee. Example 04 names
# primitives its tutorial registers at runtime.
SHIPPED = [
    "examples/01-basic-aggregations/config.yaml",
    "examples/03-deep-nesting/config.yaml",
    "examples/05-categoricals-output/config.yaml",
    "examples/06-graph-text-bridge/config.yaml",
    "featurizer/featurizer.yaml",
    "tests/fixtures/sample_config.yaml",
    "tests/fixtures/sample_config_snapshot.yaml",
]


def wide_config() -> Dict[str, Any]:
    """A config wide enough to shard into column groups and to push a child CTE
    past the materialization threshold, so the sharder and the temp-table
    preamble are both exercised."""
    return {
        "target": "accounts",
        "max_depth": 2,
        "intervals": ["P30D", "P90D"],
        "entities": [
            {
                "alias": "accounts",
                "id": "account_id",
                "table": "accounts",
                "variables": {"balance": {"type": "numeric"}},
            },
            {
                "alias": "payments",
                "id": "payment_id",
                "table": "payments",
                "temporal_ix": "paid_at",
                "variables": {f"m{i:02d}": {"type": "numeric"} for i in range(12)},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "accounts", "key": "account_id"},
                "child": {"entity": "payments", "key": "account_id"},
            }
        ],
    }


def chain_config() -> Dict[str, Any]:
    """stores <- orders <- items. With ``materialize_threshold=1`` the child
    CTEs are materialized into TEMP shards, which is the only path that reads
    ``as_of_dates`` outside the query itself (the as-of preamble cross-joins it).
    """
    return {
        "target": "stores",
        "max_depth": 3,
        "intervals": ["P30D"],
        "aggregations": ["count", "sum", "mean"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "stores",
                "id": "store_id",
                "table": "stores",
                "variables": {"area": {"type": "numeric"}},
            },
            {
                "alias": "orders",
                "id": "order_id",
                "table": "orders",
                "temporal_ix": "ordered_at",
                "variables": {"total": {"type": "numeric"}},
            },
            {
                "alias": "items",
                "id": "item_id",
                "table": "items",
                "temporal_ix": "added_at",
                "variables": {"price": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "stores", "key": "store_id"},
                "child": {"entity": "orders", "key": "store_id"},
            },
            {
                "parent": {"entity": "orders", "key": "order_id"},
                "child": {"entity": "items", "key": "order_id"},
            },
        ],
    }


def cases(tmp_dir: Path) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """``(name, config path, Featurizer kwargs)`` for every baseline case."""
    for rel in SHIPPED:
        yield rel, str(REPO / rel), {}
    for name, config, kwargs in (
        ("generated/wide", wide_config(), {"materialize_threshold": 400}),
        ("generated/chain", chain_config(), {"materialize_threshold": 1}),
    ):
        path = tmp_dir / f"{name.split('/')[1]}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        yield name, str(path), kwargs


def digests(config_path: str, kwargs: Dict[str, Any]) -> Dict[str, str]:
    from featurizer import Featurizer

    f = Featurizer(config_path, **kwargs)
    groups = f.query_groups
    parts = {
        "groups": "\n-- next group --\n".join(f"{k}\n{v}" for k, v in groups.items()),
        "ddl": "\n-- next statement --\n".join(f.materialization_ddl),
        "n_groups": str(len(groups)),
    }
    if len(groups) == 1 and not f.materialization_ddl:
        # Past the column limit, or once a child is materialized, the
        # monolithic query is not a usable artifact and ``.query`` raises.
        parts["query"] = f.query
    return {
        key: (
            value
            if key == "n_groups"
            else hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        for key, value in parts.items()
    }
