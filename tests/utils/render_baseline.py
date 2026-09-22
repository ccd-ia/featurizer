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

Issue #27 then cut every non-target read on the as-of date, so they were
captured a third time, from ``0f1feb9`` (master with #29 in) plus that fix
(``bounded_by`` in the file). Read as text against ``0f1feb9``: in eight of the
nine cases the only difference in ``query`` and ``groups`` is one added
``where <table>."<temporal_ix>" <= aod.as_of_date`` per temporal child synth,
and ``ddl`` does not move. ``generated/chain``, the one case that materializes,
moves in ``ddl`` and ``groups`` by design: its as-of shards are now built one
as-of date at a time (issue #36). ``n_groups`` did not change anywhere.

Issue #46 then quoted every declared identifier that is not a variable — entity
ids, relationship keys, temporal indexes, the columns the planner passes read —
so they were captured a fourth time, from ``b72da03`` plus that fix
(``quoted_identifiers_by``). The move is quoting only, checked as #29's was:
15 parts moved, and for each of them master's text and the fix's are identical
once every double quote is stripped; ``n_groups`` did not change.

Issue #37 then replaced the ``select *`` with which a materialized CTE's shards
were re-joined by the columns the reader names, so they were captured a fifth
time, from ``7cd6326`` plus that fix (``pruned_rejoin_by``). Only
``generated/chain`` moves, the one case that materializes: four lines in all
(three in ``ddl``, one in ``groups``), each the same line with an explicit
select list where ``select *`` was. ``n_groups`` did not change.

Issue #49 then cut the target's read on the as-of date too, when the target
declares a temporal index (ADR-0017), so they were captured a sixth time, from
``da99fd4`` plus that fix (``target_bounded_by``). Six of the nine cases have
such a target. Read as text against ``da99fd4``: each moved part gains exactly
one line, ``where <table>."<temporal_ix>" <= aod.as_of_date`` on the target's
synth, and nothing is removed. ``ddl`` and ``n_groups`` did not change. Unlike
the earlier moves this one changes what a config returns, which is the point of
the fix; the baseline only records that nothing ELSE in the SQL moved.

Issue #66 then gave every entity one row order (ADR-0018): after the temporal
index a window orders by the entity's other identifier columns and then by its
declared variables, the rolling percentiles' "up to the current row" became a
row comparison over the same columns, and the as-of lookup's ``limit 1`` takes
the last row in that order. So they were captured a seventh time, from
``18299ac`` plus that fix (``ordered_by``). Four of the nine cases move, in
five parts: the three whose config selects a window transformer, in ``groups``;
and the snapshot config, in ``query`` and ``groups``, whose one as-of lookup is
all that moves in it. Read as text against ``18299ac`` with the
gained columns stripped from every ``order by`` and row comparison, what is
left differs in nine lines: a column-group synth gaining the one declared
target variable its ``order by`` now reads (``gender`` in four groups of
``featurizer.yaml``, ``age`` in five of ``sample_config.yaml``); the literal-name
rule keeps what a window names. ``ddl`` and ``n_groups`` did not move, and
neither did any transform's select list, so no output column moves.

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
