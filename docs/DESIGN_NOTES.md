# Design notes: decisions and trade-offs

Short answers to the "why did you…" questions this project tends to raise.

## Data

**Why rebuild segments from `ppth`?**
Raw trips jump stop indices (e.g. S5 Jannowitzbrücke → Hackescher Markt, index 7 → 9) because some station
records, such as the Alexanderplatz S-Bahn platforms, are missing from the dataset. Each stop's planned path
still names the upcoming stations, so the segment is split through them and the travel time is interpolated.
42% of all segment hops come from this repair. A gap that cannot be fully named (2,254 cases over six weeks) is
**dropped**, because keeping it would invent express links. An earlier version did keep those gaps and
produced fake edges such as Hbf → Westkreuz. I caught this by diffing edge sets between builds.

**Why does "no change record" mean "on time"?**
The DB change feed only reports deviations. This is a stated assumption, listed in the README's limitations.

**Why punctuality < 6 min?**
That is Deutsche Bahn's official punctuality definition, which makes the numbers comparable to published statistics.

**Memory:** the first six-week build was OOM-killed at 6 GB. The fixes were tuple rows instead of dicts,
categorical dtypes, keeping only the first 12 planned-path stations, and running ingestion in a spawned child
process so its memory is released before graph building.

## Graph algorithms

**State-expanded Dijkstra:** a plain station graph cannot charge for changing trains, so the search state is
(station, line). Moving along the same line is free; switching lines adds `transfer_penalty`. The complexity is
the same as Dijkstra, but on a graph roughly |lines| times larger.

**Reliable routing:** edge weight = minutes + max(0, mean delay gained on the segment) + P(cancel) × 20 min.
This is a simple expected-cost model. Fastest ≤ reliable in total time is a property test.

**Closure what-if:** for every (predecessor, successor) pair of a station whose fastest route used it,
re-route with the station removed. Articulation points on the undirected graph flag single points of failure.

**Betweenness** is weighted by travel minutes (fastest paths), and **PageRank** is weighted by train frequency.

## GraphRAG

Following Microsoft's GraphRAG idea: detect communities (Louvain, weighted by trips), write a summary per
community ("global" questions), plus station and line profiles ("local" questions). Retrieved station
documents are expanded with their community summary. Summaries are templated rather than LLM-written, so
they are deterministic and free, and every number in them is correct by construction.

## Agent and evaluation

- LangGraph instead of a single prompt: tool calls are explicit, inspectable and limited by a recursion limit.
- The offline rule-based model implements the LangChain chat-model interface, so CI tests the whole graph
  (agent → ToolNode → agent) without API keys, and it doubles as the **baseline** the LLM agents must beat.
- Eval ground truth is computed from the tools, not hand-written, so it updates automatically when the graph
  is rebuilt. The baseline was deliberately **not** tuned on the eval cases, to avoid test-set leakage.

## Known limitations / next steps

Time-dependent (earliest-arrival) routing, U-Bahn/tram data (GTFS), a GNN for delay propagation, and a Neo4j export.
