# Example 6: Text → Edges → Centrality → Spine (φ-bridges, two-stage)

**Level**: Advanced
**Scenario**: Coordination detection on social posts

Six authors post short Spanish messages. Three of them paste the *same*
campaign text at staggered dates — the copy-paste signature of coordination —
while the others post distinct organic messages. This example runs the full
Path-2 pipeline from the temporal-feature taxonomy:

```
posts (text)
  ├─ SentimentBridge          → bridge_sentiment   (Path 1: reduce → aggregate)
  └─ NearDuplicateEdgeBridge  → text_edges          (Path 2: text induces edges)
       └─ CentralityBridge    → post_centrality     (per-window snapshots)
            └─ SQL spine      → feature matrix      (as-of, causally bounded)
```

## What it demonstrates

- **The φ-bridge lifecycle** (ADR-0001/ADR-0014): `compute → materialize →
  emit_yaml → spine`, with `persist=True` writing real tables — the
  orchestrated-asset flow you would wire into Dagster/Snakemake.
- **Text induces the graph** (Path 2): `NearDuplicateEdgeBridge` (MinHash/LSH)
  emits an `(src, dst, ts)` edge table between the *authors* of
  near-duplicate posts. Each edge is knowable at the **later** post of the
  pair — causality lives on the edge timestamp.
- **Temporal snapshot sequences**: `CentralityBridge.materialize_snapshots`
  rebuilds the graph *per as-of window* from strictly pre-t₀ edges (never
  slicing a full-history graph) and emits `(node_id, as_of_date)` rows — an
  ordinary event stream the spine trends.
- **Drift-proof config**: `run_example.py` asserts that the entities declared
  in `config.yaml` equal the bridges' `emit_yaml()` fragments.
- **The signal**: at the first as-of only the first pair exists (degree 1,
  clustering 0); at the second the triangle has closed (degree 2, clustering
  1.0). Organic authors never enter the induced graph (NULL centrality) and
  the campaign text carries its negative sentiment (−0.72) everywhere it was
  pasted.

## Files

- `create_data.py` — seeds `example_06`: authors, posts (the planted
  copy-paste cluster + organic posts), `as_of_dates`.
- `config.yaml` — the spine config; the two bridge output tables are declared
  as ordinary child entities.
- `run_example.py` — runs the bridges, proves the config matches
  `emit_yaml()`, then executes the spine.
- `tutorial.ipynb` — the narrated walkthrough (committed with executed
  outputs; the docs site renders from them).

## Usage

```bash
just db-up            # throwaway PostgreSQL (sets DATABASE_URL in-recipe)
just example 06       # seed + run end to end

# or directly (DATABASE_URL / PG* must point at PostgreSQL):
python create_data.py
python run_example.py --execute
python run_example.py --show-sql      # no database needed
```

## Key concepts

| Concept | Where |
|---|---|
| Edge knowable at the later document | `NearDuplicateEdgeBridge.compute_edges` |
| Per-window pre-t₀ graph rebuild | `CentralityBridge.materialize_snapshots` |
| Snapshot stream = ordinary event stream | `config.yaml` (`temporal_ix: as_of_date`) |
| Persisted bridge assets | `persist=True` in `run_example.py` |
| Cheap/heavy centrality tiers | `CentralityBridge(include_heavy=…)` |

For the native, no-Python alternative for cheap graph features (as-of degree +
1-hop neighbour aggregates in pure SQL) see the `graph_relationships` config
block in the
[configuration reference](https://ccd-ia.github.io/featurizer/reference/configuration/);
the full pattern catalog lives in the
[bridge cookbook](https://ccd-ia.github.io/featurizer/engineering/bridge-cookbook/).
