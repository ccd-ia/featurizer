"""Paired cohorts: ``as_of_dates: {id_column: …}`` (issue #10).

Two properties matter and they pull in opposite directions. The key must narrow
the matrix to the declared ``(as_of_date, id)`` pairs, and its *absence* must
change nothing at all. The second protects every existing consumer, so it gets
the strictest test here: a byte comparison against digests captured from master
before the key existed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from featurizer import Featurizer
from featurizer.validation import ConfigValidator
from tests.utils.render_baseline import BASELINE, cases, chain_config, digests

CONFIG = {
    "target": "customers",
    "max_depth": 2,
    "intervals": ["P30D"],
    "aggregations": ["count", "sum"],
    "transformations": ["identity"],
    "entities": [
        {
            "alias": "customers",
            "id": "customer_id",
            "table": "customers",
            "variables": {"age": {"type": "numeric"}},
        },
        {
            "alias": "orders",
            "id": "order_id",
            "table": "orders",
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


def _paired(config: dict, id_column: str = "cohort_id") -> dict:
    config = copy.deepcopy(config)
    config["as_of_dates"] = {"id_column": id_column}
    return config


def _featurizer(tmp_path: Path, config: dict, **kwargs) -> Featurizer:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return Featurizer(str(path), **kwargs)


# ------------------------------------------------- absent means byte-identical


def test_the_baseline_was_captured_before_the_change() -> None:
    baseline = json.loads(BASELINE.read_text())
    assert baseline["captured_from"] == "3e1790d"
    # One case must exercise the temp-table preamble, the only reader of
    # as_of_dates outside the query itself.
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert any(case["ddl"] != empty for case in baseline["cases"].values())
    assert any(int(case["n_groups"]) > 1 for case in baseline["cases"].values())


def test_without_the_block_every_renderer_is_byte_identical(tmp_path) -> None:
    baseline = json.loads(BASELINE.read_text())["cases"]
    seen = set()
    for name, path, kwargs in cases(tmp_path):
        assert digests(path, kwargs) == baseline[name], name
        seen.add(name)
    assert seen == set(baseline)


# ------------------------------------------------------------- validation


def test_a_well_formed_block_validates() -> None:
    result = ConfigValidator().validate(_paired(CONFIG))
    assert result.is_valid, [e.message for e in result.errors]


@pytest.mark.parametrize(
    "block,location",
    [
        ("cohort_id", "as_of_dates"),
        (["cohort_id"], "as_of_dates"),
        ({}, "as_of_dates.id_column"),
        ({"id_column": ""}, "as_of_dates.id_column"),
        ({"id_column": "   "}, "as_of_dates.id_column"),
        ({"id_column": 7}, "as_of_dates.id_column"),
        ({"id_column": "cohort_id", "date_column": "d"}, "as_of_dates.date_column"),
    ],
)
def test_a_malformed_block_is_an_error(block, location) -> None:
    config = copy.deepcopy(CONFIG)
    config["as_of_dates"] = block
    result = ConfigValidator().validate(config)

    assert not result.is_valid
    assert location in [e.location for e in result.errors]


def test_a_misspelled_key_gets_a_suggestion() -> None:
    config = copy.deepcopy(CONFIG)
    config["as_of_dates"] = {"id_colum": "cohort_id"}
    errors = ConfigValidator().validate(config).errors

    unknown = [e for e in errors if e.location == "as_of_dates.id_colum"]
    assert len(unknown) == 1
    assert "id_column" in (unknown[0].suggestion or "")


def test_pairing_needs_a_target_id() -> None:
    config = _paired(CONFIG)
    config["entities"][0]["id"] = None
    result = ConfigValidator().validate(config)

    assert "as_of_dates.id_column" in [e.location for e in result.errors]


def test_chain_config_is_a_valid_config() -> None:
    """The baseline's generated configs must be real configs, not render-only."""
    assert ConfigValidator().validate(chain_config()).is_valid


# ------------------------------------------------------------- rendering

SPINE = "from (select distinct as_of_date from as_of_dates) as aod"


def _cut(target_id: str, id_column: str = '"cohort_id"') -> str:
    return (
        f"where {target_id} in (select _cohort.{id_column} from as_of_dates "
        "_cohort where _cohort.as_of_date = aod.as_of_date)"
    )


def _population(config: dict) -> dict:
    config = copy.deepcopy(config)
    config["transformations"] = ["identity", "cross_entity_zscore"]
    return config


def test_aod_ranges_over_the_distinct_dates(tmp_path) -> None:
    """One lateral evaluation per DATE, as in the dense query. With aod ranging
    over the pairs a 65-aggregation config had not finished after 1,520 s where
    the dense query took 72 s; on the dates it stays within 8% of dense."""
    query = _featurizer(tmp_path, _paired(CONFIG)).query

    assert query.count(SPINE) == 1
    assert "from as_of_dates as aod" not in query


def test_the_predicate_sits_in_the_targets_base_read(tmp_path) -> None:
    """In the synth CTE, after its joins — not as a filter on the lateral's output."""
    query = _featurizer(tmp_path, _paired(CONFIG)).query

    synth = query[
        query.index("customers_synth as (") : query.index("-- transform customers")
    ]
    predicate = _cut('customers."customer_id"') + "\n        )"
    assert predicate in synth
    # After the base table and after its joins: a WHERE cannot precede them.
    assert synth.index("from customers") < synth.index("left join")
    assert synth.index("left join") < synth.index(predicate)
    assert query.count("_cohort.as_of_date = aod.as_of_date") == 1
    assert "select * from customers_transform\n" in query
    # The child is not paired: it keeps its causal cut and gains nothing.
    child = query[: query.index("customers_synth as (")]
    assert "_cohort" not in child


def test_the_id_column_is_quoted_byte_for_byte(tmp_path) -> None:
    query = _featurizer(tmp_path, _paired(CONFIG, "Cohort Id")).query
    assert '_cohort."Cohort Id"' in query


def test_every_column_group_carries_the_spine_and_the_predicate(tmp_path) -> None:
    from tests.utils.render_baseline import wide_config

    f = _featurizer(tmp_path, _paired(wide_config()), materialize_threshold=400)
    groups = f.query_groups

    assert len(groups) > 1
    for name, sql in groups.items():
        assert sql.count(SPINE) == 1, name
        assert sql.count(_cut('accounts."account_id"')) == 1, name


def test_the_temp_table_preamble_reads_distinct_dates(tmp_path) -> None:
    """It cross-joins as_of_dates and groups by date; a pair table repeats each
    date once per id, which would multiply every count and sum."""
    paired = _featurizer(tmp_path, _paired(chain_config()), materialize_threshold=1)
    ddl = "\n".join(paired.materialization_ddl)

    assert "cross join" in ddl
    assert "from as_of_dates aod" not in ddl
    assert "from (select distinct as_of_date from as_of_dates) aod" in ddl


def test_a_population_level_transformer_moves_the_cut_after_the_transform(
    tmp_path,
) -> None:
    """``cross_entity_zscore`` is ``avg(x) over ()``: it compares a target row
    with the other target rows. Narrowing the base read shrinks that population
    to the cohort — measured, every z-score turned NULL — so the cut is applied
    to the final select instead and nothing upstream is narrowed."""
    query = _featurizer(tmp_path, _paired(_population(CONFIG))).query

    assert query.count("_cohort.as_of_date = aod.as_of_date") == 1
    synth = query[
        query.index("customers_synth as (") : query.index("-- transform customers")
    ]
    assert "_cohort" not in synth
    post_filter = "select * from customers_transform " + _cut('"customer_id"')
    assert post_filter in query
    assert query.count(SPINE) == 1


def test_the_population_flag_is_what_the_planner_reads() -> None:
    from featurizer.primitives.utils import get_transformers

    flagged = {
        name
        for name, t in get_transformers(None).items()
        if getattr(t, "population_level", False)
    }
    assert flagged == {"cross_entity_zscore", "cross_entity_percentile"}


# ------------------------------------------------------------- execution

DATES = ["2024-02-01", "2024-03-01", "2024-04-01"]
# Several pairs per date on purpose: that is what would expose a multiplied
# aggregate in the temp-table preamble.
PAIRS = [
    ("2024-02-01", 1),
    ("2024-02-01", 2),
    ("2024-03-01", 2),
    ("2024-03-01", 3),
    ("2024-04-01", 1),
    ("2024-04-01", 2),
    ("2024-04-01", 4),
]


def _load_flat(cur) -> None:
    cur.execute(
        "create temp table customers (customer_id int, age numeric, segment text)"
    )
    cur.execute(
        "insert into customers values "
        "(1, 30, 'a'), (2, 40, 'a'), (3, 50, 'a'), (4, 60, 'b')"
    )
    cur.execute(
        "create temp table orders (order_id int, customer_id int, "
        "ordered_at date, amount numeric)"
    )
    cur.execute(
        "insert into orders values "
        "(1, 1, '2024-01-10', 10), (2, 1, '2024-03-15', 20), "
        "(3, 2, '2024-01-20', 5), (4, 2, '2024-02-20', 7), (5, 2, '2024-03-25', 9), "
        "(6, 3, '2024-02-25', 100), (7, 4, '2024-05-01', 1)"
    )


def _load_chain(cur) -> None:
    cur.execute("create temp table stores (store_id int, area numeric)")
    cur.execute("insert into stores values (1, 100), (2, 200), (3, 300), (4, 400)")
    cur.execute(
        "create temp table orders (order_id int, store_id int, "
        "ordered_at date, total numeric)"
    )
    cur.execute(
        "insert into orders values "
        "(1, 1, '2024-01-10', 10), (2, 1, '2024-03-15', 20), "
        "(3, 2, '2024-01-20', 5), (4, 2, '2024-02-20', 7), "
        "(5, 3, '2024-02-25', 100), (6, 4, '2024-05-01', 1)"
    )
    cur.execute(
        "create temp table items (item_id int, order_id int, "
        "added_at date, price numeric)"
    )
    cur.execute(
        "insert into items values "
        "(1, 1, '2024-01-10', 4), (2, 1, '2024-01-11', 6), (3, 2, '2024-03-15', 20), "
        "(4, 3, '2024-01-20', 5), (5, 4, '2024-02-20', 3), (6, 4, '2024-02-21', 4), "
        "(7, 5, '2024-02-25', 100), (8, 6, '2024-05-01', 1)"
    )


def _peer_groups(config: dict) -> dict:
    """Peer features compare an entity with its peers: a planner pass that reads
    the base table in its own CTE, so narrowing the synth must not reach it."""
    config = copy.deepcopy(config)
    target = config["entities"][0]
    target["variables"]["segment"] = {"type": "categorical", "role": "identifier"}
    target["peer_groups"] = [{"by": "segment", "measures": ["age"]}]
    return config


def _rows(frame, id_name: str) -> dict:
    frame = frame.reset_index()
    columns = [c for c in frame.columns if c not in ("as_of_date", id_name)]
    return {
        (str(r["as_of_date"]), int(r[id_name])): tuple(
            None if r[c] is None or r[c] != r[c] else float(r[c]) for c in columns
        )
        for _, r in frame.iterrows()
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    "config,load,id_name,kwargs",
    [
        (CONFIG, _load_flat, "customer_id", {}),
        (chain_config(), _load_chain, "store_id", {"materialize_threshold": 1}),
        (_population(CONFIG), _load_flat, "customer_id", {}),
        (_peer_groups(CONFIG), _load_flat, "customer_id", {}),
    ],
    ids=["one-query", "temp-table-preamble", "population-level", "peer-groups"],
)
def test_the_paired_matrix_equals_the_dense_one_on_the_pairs(
    tmp_path, config, load, id_name, kwargs
) -> None:
    """The correctness gate: a faster wrong answer is the failure mode here."""
    import os

    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")

    dense_f = _featurizer(tmp_path, config, **kwargs)
    paired_dir = tmp_path / "paired"
    paired_dir.mkdir()
    paired_f = _featurizer(paired_dir, _paired(config), **kwargs)

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            load(cur)
            cur.execute("create temp table as_of_dates (as_of_date date)")
            cur.executemany(
                "insert into as_of_dates values (%s)", [(d,) for d in DATES]
            )
        dense = _rows(dense_f.to_dataframe(connection=conn), id_name)
        conn.rollback()

        with conn.cursor() as cur:
            load(cur)
            cur.execute(
                "create temp table as_of_dates (as_of_date date, cohort_id int)"
            )
            # One pair is inserted twice: a semi-join must not repeat its row.
            cur.executemany(
                "insert into as_of_dates values (%s, %s)", PAIRS + PAIRS[:1]
            )
        frame = paired_f.to_dataframe(connection=conn)
        assert len(frame) == len(PAIRS)
        paired = _rows(frame, id_name)
        conn.rollback()

    assert len(dense) == len(DATES) * 4
    assert set(paired) == set(PAIRS)
    assert paired == {pair: dense[pair] for pair in PAIRS}
    # The oracle must be able to fail: the dense values differ across pairs.
    assert len(set(paired.values())) > 1


# A target with several rows per id and a temporal index, so the window
# transformers have a timeline to walk and a partition with more than one row.
SWEEP_CONFIG = """target: visits
max_depth: 1
intervals: []
aggregations: []
transformations: [{name}]
entities:
  - alias: visits
    id: patient_id
    table: visits
    temporal_ix: seen_at
    variables:
      x:
        type: {vtype}
{block}"""
SWEEP_SQL_TYPE = {"numeric": "numeric", "date": "date", "text": "text"}
SWEEP_VALUES = {
    "numeric": ["10", "20", "5.5", "7", "3", "41"],
    "date": [f"date '2024-0{m}-1{m}'" for m in range(1, 7)],
    "text": ["'alpha beta'", "'Gamma!'", "'delta'", "'alpha'", "'eps, zeta.'", "'eta'"],
}
# (patient_id, seen_at) for the six rows: three patients, two visits each.
SWEEP_KEYS = [
    (1, "2024-01-05"),
    (1, "2024-02-05"),
    (2, "2024-01-15"),
    (2, "2024-03-05"),
    (3, "2024-02-10"),
    (3, "2024-02-20"),
]
SWEEP_PAIRS = [("2024-06-01", 1), ("2024-06-01", 3), ("2024-07-01", 2)]


def _sweep_run(conn, query: str, vtype: str, paired: bool) -> dict:
    rows = ", ".join(
        f"({pid}, date '{seen}', {value})"
        for (pid, seen), value in zip(SWEEP_KEYS, SWEEP_VALUES[vtype])
    )
    with conn.cursor() as cur:
        cur.execute(
            "create temp table visits (patient_id int, seen_at date, "
            f"x {SWEEP_SQL_TYPE[vtype]})"
        )
        cur.execute(f"insert into visits values {rows}")
        if paired:
            cur.execute(
                "create temp table as_of_dates (as_of_date date, cohort_id int)"
            )
            cur.executemany("insert into as_of_dates values (%s, %s)", SWEEP_PAIRS)
        else:
            cur.execute("create temp table as_of_dates (as_of_date date)")
            cur.executemany(
                "insert into as_of_dates values (%s)",
                sorted({(d,) for d, _ in SWEEP_PAIRS}),
            )
        try:
            cur.execute(query)
            names = [c.name for c in cur.description]
            out = {}
            for row in cur.fetchall():
                record = dict(zip(names, row))
                key = (
                    str(record.pop("as_of_date")),
                    record.pop("patient_id"),
                    str(record.pop("seen_at")),
                )
                out[key] = record
            return out
        finally:
            conn.rollback()


@pytest.mark.integration
def test_every_transformer_gives_the_dense_value_on_the_pairs(tmp_path) -> None:
    """The net under ``population_level``: a transformer that reads other target
    rows and does not say so would differ here."""
    import os

    import psycopg

    from featurizer.primitives.utils import get_transformers, list_transformations

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")

    block = "as_of_dates:\n  id_column: cohort_id\n"
    wanted = {(d, pid) for d, pid in SWEEP_PAIRS}
    compared, broken_either_way, differing = [], [], []
    with psycopg.connect(url) as conn:
        for name in sorted(list_transformations()):
            if name in {"identity", "in_array"}:  # not a bare transformations: entry
                continue
            types = getattr(get_transformers([name])[name], "input_types", None)
            vtype = next((t for t in types or ["numeric"] if t in SWEEP_SQL_TYPE), None)
            if vtype is None:
                continue
            queries = {}
            for paired in (False, True):
                path = tmp_path / f"{name}-{paired}.yaml"
                path.write_text(
                    SWEEP_CONFIG.format(
                        name=name, vtype=vtype, block=block if paired else ""
                    )
                )
                queries[paired] = Featurizer(str(path)).query
            try:
                dense = _sweep_run(conn, queries[False], vtype, paired=False)
            except psycopg.Error:
                # Fails with no cohort at all (issue #23's list): nothing to compare.
                broken_either_way.append(name)
                continue
            paired_rows = _sweep_run(conn, queries[True], vtype, paired=True)
            expected = {k: v for k, v in dense.items() if (k[0], k[1]) in wanted}
            compared.append(name)
            if paired_rows != expected:
                differing.append(name)

    assert differing == []
    assert len(compared) >= 70, (len(compared), broken_either_way)
    assert {"cross_entity_zscore", "cross_entity_percentile"} <= set(compared)
