---
title: "Bridge cookbook"
description: "How to wire and extend the φ-bridge feature families — text, graph, embeddings, sequence."
---

The SQL spine synthesizes anything expressible as point-in-time-correct SQL.
Some feature families are not — NER counts, graph centrality, sentence
embeddings, fitted sequence models. The **φ-bridge**
([`featurizer/bridge/`](https://github.com/ccd-ia/featurizer/tree/master/featurizer/bridge))
is the precompute companion for exactly those: heavy Python computes a value φ,
materializes it back into PostgreSQL as an ordinary column, and emits a config
fragment declaring it as a `Variable`. The spine then aggregates it with its
normal `<= as_of_date` causal bound — **no second feature engine**.

Every bridge follows one lifecycle:

```text
compute  →  materialize  →  emit_yaml  →  splice into your config  →  spine
```

The hard invariant is the causal boundary
([ADR-0001](/featurizer/engineering/adr/0001-phi-bridge-precompute-causal-boundary/)):
any *fitted* model trains only on rows knowable as-of the cutoff, enforced
fail-fast by `assert_pre_t0`. The 0.9.0 contract extensions — multi-column
output, temporal snapshot sequences, `persist=`, `model_vintage` — are
recorded in
[ADR-0014](/featurizer/engineering/adr/0014-multi-column-bridge-and-temporal-snapshots/).

## Worked example — text (Path 1: reduce → aggregate)

Collapse each document to per-event scalars; the spine then gives you
sentiment slope, recency-of-negative, volatility for free. The sentiment,
readability, and language-id bridges are **dependency-free**; defaults are
Spanish/multilingual per the taxonomy's rule — English is a one-line override,
never the silent default.

```python
from featurizer.bridge import SentimentBridge

bridge = SentimentBridge(pk_col="doc_id", text_col="body")   # language="es"
bridge.materialize(
    conn,
    source_table="complaints",
    pk="doc_id",
    carry_cols=["facility_id", "filed_at"],   # FK + temporal_ix pass through
    content_cols=["body"],
    output_table="bridge_sentiment",
)
fragment = bridge.emit_yaml(
    output_table="bridge_sentiment",
    pk="doc_id",
    parent_alias="facilities",
    parent_key="facility_id",
    fk="facility_id",
    temporal_ix="filed_at",
)
# config["entities"].append(fragment["entity"])
# config["relationships"].append(fragment["relationship"])
```

`NERCountsBridge` is the multi-column case: **one spaCy parse emits five
columns** (`persons`, `orgs`, `locations`, `money`, `dates`) via the ADR-0014
`MultiColumnBridge` contract. It wraps a *pretrained* model, which
`assert_pre_t0` cannot see — declare `model_vintage=` (the model's training
cutoff) and call `assert_model_vintage(as_of)` in strict backtests.

```python
from featurizer.bridge import NERCountsBridge

bridge = NERCountsBridge(
    pk_col="doc_id", text_col="body",
    language="es",                      # default "xx" = multilingual model
    model_vintage=date(2023, 4, 1),     # pin what the model could have known
)
```

## Worked example — graph (snapshot sequences)

Centrality is **non-local**: one future edge changes every node's score, so a
backtest cohort must rebuild the graph *per as-of window* from strictly pre-t₀
edges — never compute one full-history graph and slice it. That is what
`materialize_snapshots` does; the cost is O(windows × build), which is why the
**cheap tier is the default** (degree/in/out/weighted, coreness, clustering)
and every heavier metric (betweenness, eigenvector, closeness) is opt-in via
`include_heavy=True`.

```python
from featurizer.bridge import CentralityBridge

bridge = CentralityBridge(source_col="src", target_col="dst", directed=False)
bridge.materialize_snapshots(
    conn,
    source_table="contact_edges",
    output_table="bridge_centrality",
    as_of_dates=cohort_dates,           # one pre-t₀ rebuild per window
    causal_col="contacted_at",
    content_cols=["src", "dst"],
    entity_col="node_id",
    as_of_col="as_of_date",
)
fragment = bridge.emit_yaml(
    output_table="bridge_centrality",
    pk="node_id", parent_alias="facilities",
    parent_key="facility_id", fk="node_id",
    temporal_ix="as_of_date",           # the snapshot stream is an event stream
)
```

The output is keyed `(node, as_of_date)` — an ordinary event stream, so the
spine trends "centrality over time" like any other metric.

`CommunityBridge` (Louvain) emits membership as a **categorical** column — it
flows through the existing fixed-vocabulary one-hot path
([ADR-0007](/featurizer/engineering/adr/0007-direct-categorical-fixed-vocabulary/))
— plus the partition's modularity. Labels are per-partition names, not stable
identities across snapshots.

## The native alternative: `graph_relationships` (no Python at all)

The cheapest graph tier — as-of degree and 1-hop neighbour mean/share — needs
no bridge: it is a **planner pass** generating pure SQL, declared next to
`peer_groups` / `spatial_relationships`:

```yaml
graph_relationships:
  - name: contacts
    left: facilities            # features attach here
    right: states               # neighbour-state entity (default: left)
    edges:
      table: contact_edges
      source: src_id
      target: dst_id
      timestamp: contacted_at   # required — the pass is as-of by construction
    directed: true
    # measures: [risk_score]    # default: right's numeric variables
    # shares:   [flagged]       # default: right's boolean variables
```

This yields `DEGREE(contacts)` (plus one windowed variant per configured
interval) and `NEIGHBOUR_MEAN` / `NEIGHBOUR_SHARE` columns, bounded by **both**
the edge timestamp and the neighbour state's `temporal_ix` — pre-t₀ edges *and*
pre-t₀ neighbour states. It is deliberately **1-hop only**: 2-hop aggregation
pulls neighbours' future labels (the canonical temporal-GNN leakage) and is not
offered.

## Worked example — text induces the graph (Path 2, two-stage)

Documents can *create* edges — the copy-paste signature of coordination. The
edge builders emit an `(src, dst, ts)` table that is the **input** to the
graph bridges (or the native `graph_relationships` block); causality lives on
the edge timestamp, which for a near-duplicate pair is the *later* document's
— the pair does not exist until the copy appears.

```python
from featurizer.bridge import CentralityBridge, NearDuplicateEdgeBridge

# Stage 1: text -> induced edge table (MinHash/LSH, datasketch).
NearDuplicateEdgeBridge(
    pk_col="post_id", entity_col="author_id",
    text_col="body", ts_col="posted_at",
).materialize_edges(
    conn,
    source_table="posts",
    output_table="text_edges",
    content_cols=["post_id", "author_id", "posted_at", "body"],
)

# Stage 2: edge table -> per-(node, as_of) centrality snapshots -> spine.
CentralityBridge(source_col="src", target_col="dst", directed=False)
    .materialize_snapshots(
        conn, source_table="text_edges", output_table="text_centrality",
        as_of_dates=cohort_dates, causal_col="ts",
        content_cols=["src", "dst"],
    )
```

`CoMentionEdgeBridge` induces edges between names mentioned together in one
document (its default extractor is a deliberately naive capitalized-sequence
heuristic — pass `extract=` for an NER-based one).

## Trajectory & sequence extensions (0.9.1)

`EmbeddingTrajectoryBridge` scores each event against the entity's **own
strictly-prior** embedding history: `novelty` (1 − max cosine — "out of
character?"), `drift` (distance to the history centroid), `volatility` (step
distance to the previous event). It reads a precomputed embedding column —
`SentenceEmbeddingBridge` output works directly — and needs only numpy. The
first event is NULL: no history is not the same as maximal novelty.

`ChangePointBridge` (strongest mean shift in a measure series, with its 0–1
position) and `PeriodicityBridge` (FFT-peak strength and period of the
event-count series) are **per-entity** scores over the pre-t₀ series — pair
them with `materialize_nodes` for one snapshot or `materialize_snapshots` for
a backtest cohort, exactly like the graph bridges.

## Writing your own bridge

Subclass `BridgeComputer` (one value column) or `MultiColumnBridge`
(`compute() → {pk: {col: val}}`), implement `compute(rows, *, fit_rows)`, and
guard optional imports with the install-hint pattern. Fit only on `fit_rows` —
they are the causal-guarded pre-t₀ slice. Everything else (loading, DDL,
INSERT, YAML fragment, snapshot loop) is inherited.

```python
from featurizer.bridge import MultiColumnBridge

class UrgencyBridge(MultiColumnBridge):
    def __init__(self, *, pk_col, text_col):
        super().__init__(name="urgency", value_cols=["urgency", "hedging"])
        self.pk_col, self.text_col = pk_col, text_col

    def compute(self, rows, *, fit_rows):
        return {
            r[self.pk_col]: {"urgency": ..., "hedging": ...}
            for r in rows
        }
```

Rules of the road:

- **Optional deps stay optional**
  ([ADR-0003](/featurizer/engineering/adr/0003-bridge-orchestration-boundary/)):
  the SQL spine never imports bridge deps; add yours to the `[bridge]` extra.
- **`persist=True`** turns the default session-temporary output into a real
  table — the shape you want when the bridge runs as a Dagster/Snakemake asset
  upstream of the SQL run. The bridge is a library, not a scheduler.
- **Snapshot mode is opt-in cost**: use it only for non-local φ (graph
  metrics, per-entity models); per-row content bridges gain nothing from
  per-window recomputation.
- **Pretrained models**: `assert_pre_t0` guards *fitted* models only. A
  pretrained snapshot trained on post-t₀ data is silent leakage — declare
  `model_vintage` and assert it in strict backtests (ADR-0014).

## Dependency matrix

| Family | Bridge | Dependency |
| --- | --- | --- |
| Sentiment, readability, language id | `SentimentBridge`, `ReadabilityBridge`, `LanguageIdBridge` | none |
| Markov surprisal | `MarkovSurprisalBridge` | none |
| NER counts | `NERCountsBridge` | spaCy + a downloaded model |
| TF-IDF topic share | `TfidfTopicShareBridge` | scikit-learn |
| PageRank, multi-metric centrality | `PageRankBridge`, `CentralityBridge` | networkx |
| Community (Louvain) | `CommunityBridge` | python-louvain |
| Sentence embeddings | `SentenceEmbeddingBridge` | sentence-transformers + pgvector |
| Embedding trajectory (novelty/drift/volatility) | `EmbeddingTrajectoryBridge` | none (numpy) |
| Change point, periodicity | `ChangePointBridge`, `PeriodicityBridge` | none (numpy) |
| Near-duplicate edges (MinHash/LSH) | `NearDuplicateEdgeBridge` | datasketch |
| Co-mention edges | `CoMentionEdgeBridge` | none |
| 1-hop degree / neighbour mean / share | *(native `graph_relationships`)* | none — pure SQL |

Install the heavy set with `pip install 'featurizer[bridge]'`; spaCy models
are separate downloads (`python -m spacy download xx_ent_wiki_sm`). SBM /
MDL-surprise is deliberately deferred: graph-tool is not pip-installable.
