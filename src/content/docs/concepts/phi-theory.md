---
title: "φ: the theory of feature creation"
description: >-
  Features as functions over time-restricted event streams — the formalism
  behind Deep Feature Synthesis, why point-in-time correctness holds by
  construction, and how the φ-bridge, peer-group and spatial families fit.
sidebar:
  order: 1
  label: The φ formalism
---

Every page on this site shows *how* to make featurizer produce features. This
one explains *what a feature is* — the small formal idea that makes the whole
library cohere, and the reason leakage is impossible by construction rather
than by discipline.

## The intuition

A feature is **a question you ask about an entity's past**: *how many orders
did this customer place in the last 7 days? What was their average basket in
the last month?* Two things hide inside every such question:

1. **An entity and a moment.** The question is not "how many orders" — it is
   "how many orders *as of June 1st*". The same customer gives a different
   answer on June 2nd.
2. **A restriction.** Only events that had already happened by that moment may
   count. Anything later is the future, and the future must be invisible —
   otherwise a model trained on these features cheats.

So a feature is a function of two arguments — an entity and an as-of date —
evaluated over the events visible at that date. featurizer calls this function
**φ**. Everything else — YAML, planners, CTEs — is machinery for composing φ
from small pieces and compiling it to SQL.

**Play with the idea before reading the formalism** — drag the as-of date and
watch φ recompute; step through the depth-stacking panel to watch a nested
feature name assemble itself:

<iframe src="/featurizer/explorables/phi-dfs.html" title="Interactive: φ and DFS composition" style="width:100%;height:1080px;border:1px solid var(--sl-color-hairline);border-radius:8px"></iframe>

<a href="/featurizer/explorables/phi-dfs.html">Open the explorable full-page</a>.

## The formalism

**Setup.** Let $E$ be an entity (one row per member: a customer, a facility).
A child entity $C$ related to $E$ is an **event stream**: rows
$(\mathrm{key}, \tau, x_1, \dots, x_k)$ where $\tau$ is the event timestamp —
the column you declare as `temporal_ix` — and $x_1, \dots, x_k$ are the
declared variables. Let $T$ be the set of as-of dates (your `as_of_dates`
table).

**The restriction operator.** For an entity $e$ and time $t$, the visible
history is

$$
H(e, t) =
\begin{cases}
\{\, (\tau, x) \in \mathrm{events}(e) : \tau \le t \,\} & \text{inclusive boundary}\\[4pt]
\{\, (\tau, x) \in \mathrm{events}(e) : \tau < t \,\} & \text{exclusive boundary}
\end{cases}
$$

— the `as_of_boundary` config key chooses the comparison. Every construct
below consumes $H(e, t)$ and nothing else.

**A feature** is any function

$$
\varphi : E \times T \to \mathbb{R} \cup \{\text{NULL}\}
\qquad\text{with}\qquad
\varphi(e, t) = f\bigl(H(e, t)\bigr)
$$

$\text{NULL}$ is a first-class value: an empty window means "no data", which
is signal, not zero.

**The two primitive families.** featurizer composes φ from exactly two kinds
of pieces:

- **Transformations** $g$ act *within* an entity, row by row (or over the
  entity's own ordered history — lags, rolling stats, cumulative sums):
  $g(x)$ per event, timestamps untouched. In SQL: an expression in the
  entity's `_transform` CTE.
- **Aggregations** $a$ act *across* a relationship, collapsing a multiset of
  child values to a scalar. The windowed variant restricts further to an
  interval $w$ ending at $t$:

$$
a_w(e, t) \;=\; a\bigl(\{\, g(x) : (\tau, x) \in H(e, t),\; \tau > t - w \,\}\bigr)
$$

  In SQL: an aggregate with a `FILTER (WHERE daterange(t − w, t) @> τ)`
  clause in the relationship's aggregation CTE.

**Deep Feature Synthesis** is closure under composition. With entities
$E \leftarrow C \leftarrow D$ and `max_depth` $= k$, the feature space is
every well-typed stack

$$
\varphi = g' \circ a_w \circ g \;\;\text{(depth 2)}
\qquad
\varphi = g'' \circ a_w \circ g' \circ a_v \circ g \;\;\text{(depth 3)}
\quad\dots
$$

up to depth $k$ — each $g$ drawn from the 83 transformers, each $a$ from the
67 aggregations, each $w$ from your `intervals`. That mechanical enumeration
of compositions is the core idea of the DFS paper: J. M. Kanter &
K. Veeramachaneni,
[*Deep Feature Synthesis: Towards Automating Data Science Endeavors*](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf)
(IEEE DSAA 2015; see also the
[project page](https://dai.lids.mit.edu/projects/deep-feature-synthesis/)).
featurizer's contribution is making the composition *temporal* (every layer
respects `H(e, t)`) and compiling it to a single PostgreSQL query instead of
in-memory dataframes.

**Point-in-time correctness, by construction.** The only data-access
primitive in the algebra is $H(e, t)$. Transformations preserve timestamps;
aggregations only ever consume restricted histories; composition cannot
un-restrict. Therefore *every* $\varphi$ featurizer can express satisfies

$$
\varphi(e, t) \text{ depends only on events with } \tau \le t
$$

There is no discipline to maintain and no review checklist — a leaky feature
is not expressible in the algebra. In the rendered SQL you can point at the
guarantee: the `where τ ≤ aod.as_of_date` guard plus the interval `FILTER`
clauses ([see the query skeleton](/featurizer/walkthrough/#4-render-the-sql--no-database-needed)).

**Names are serialized $\varphi$.** The column name is the composition,
written inside-out:

```text
"MEAN(orders.ABS(orders.amount)|interval=P30D)"
```

$$
= \;\operatorname{mean} \circ \operatorname{abs}
\text{ over } \texttt{orders.amount},\; w = \text{P30D}
$$

which is why the [feature manifest](/featurizer/walkthrough/#6-read-your-features)
can reconstruct lineage (depth, parents, source column, interval) for every
column mechanically.

## Three φ variants beyond the registry

Three feature families are **planner passes** with their own config blocks
rather than registry primitives — but they are the same shape: functions of
restricted histories.

**The φ-bridge** ([ADR-0001](/featurizer/engineering/adr/0001-phi-bridge-precompute-causal-boundary/)).
Some φ need heavy Python — an embedding, a graph statistic — that SQL should
not recompute. The bridge computes a value **per source row** offline,
materializes it back as an ordinary column *with the row's own timestamp*,
and the value re-enters the algebra as a plain variable subject to the same
`τ ≤ t` bound. The causal boundary survives because the precomputed value is
itself an event: it becomes visible when its row does, never earlier. (This
is why per-row φ is the supported shape — a value that genuinely depends on
`(e, t)` jointly cannot be materialized once per row.)

**Peer groups** (`peer_groups:` on an entity). These compare an entity
against the *distribution of the same $\varphi$ over its peers* — entities
sharing a categorical:

$$
\mathrm{PEER\_ZSCORE}(e, t) \;=\;
\frac{\varphi(e, t) - \operatorname{mean}\{\, \varphi(p, t) : p \in \mathrm{peers}(e) \,\}}
     {\operatorname{std}\{\, \varphi(p, t) : p \in \mathrm{peers}(e) \,\}}
$$

Still a function of restricted histories only — just of *several* entities'
histories at the same $t$. Synthesized columns: `PEER_GROUP_SIZE`,
`PEER_EVENT_RATE`, and per measure `PEER_MEAN` / `PEER_ZSCORE` /
`PEER_PCTILE` / `EGO_MINUS_PEER_MEAN`.

**Spatial relationships** (`spatial_relationships:`). Here the "history" is a
second table's geometry: φ asks how entity `e`'s location relates to another
entity set's locations — `COLOCATION_COUNT` within a radius,
`DISTANCE_TO_NEAREST`, `KDE_INTENSITY` under a bandwidth. When the second
table is temporal, the same restriction applies to it; the spatial predicate
(`within_m`, `bandwidth_m`) simply replaces the interval window as the
"neighborhood" being aggregated.

## Where to go next

- [The walkthrough](/featurizer/walkthrough/) — see φ compiled to SQL on real
  data.
- [Primitives reference](/featurizer/reference/primitives/) — the full
  vocabulary of `g` and `a`.
- [Performance internals](/featurizer/engineering/internals/) — how the
  compiled query stays fast when the composition space gets wide.
