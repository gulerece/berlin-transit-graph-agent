"""GraphRAG layer: turn the graph into retrievable text, then search it.

Following the GraphRAG idea (Edge et al., 2024), the graph is partitioned into
communities (Louvain on the train-frequency-weighted network) and every
community gets a summary. Retrieval works on three document types:

* ``community`` – "global" summaries of a part of the network
* ``station``   – "local" profiles of one station and its neighbourhood
* ``line``      – the stops and reliability of one line

Station hits are expanded with their community summary (graph-guided context
expansion), which is what makes this graph retrieval rather than plain RAG.

Two retrieval backends are supported:
* TF-IDF (default, offline, deterministic – used in tests and CI)
* any LangChain ``Embeddings`` (e.g. OllamaEmbeddings, OpenAIEmbeddings)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


@dataclass
class Doc:
    id: str
    kind: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def detect_communities(G: nx.DiGraph, seed: int = 42, resolution: float = 1.0) -> list[set[str]]:
    und = nx.Graph()
    und.add_nodes_from(G)
    for u, v, d in G.edges(data=True):
        w = und.edges[u, v]["weight"] + d["trips"] if und.has_edge(u, v) else d["trips"]
        und.add_edge(u, v, weight=w)
    comms = nx.community.louvain_communities(und, weight="weight", seed=seed, resolution=resolution)
    return sorted(comms, key=len, reverse=True)


def build_documents(G: nx.DiGraph, betweenness: dict[str, float]) -> tuple[list[Doc], dict[str, str]]:
    name = lambda n: G.nodes[n].get("name", n)  # noqa: E731
    rank = {n: i + 1 for i, n in enumerate(sorted(betweenness, key=betweenness.get, reverse=True))}
    comms = detect_communities(G)
    community_of: dict[str, str] = {}
    docs: list[Doc] = []

    for i, members in enumerate(comms):
        hubs = sorted(members, key=lambda n: betweenness.get(n, 0), reverse=True)
        cid = f"community-{i}"
        title = f"Network area around {name(hubs[0])}" + (f" and {name(hubs[1])}" if len(hubs) > 1 else "")
        for n in members:
            community_of[n] = cid
        internal = [(u, v, d) for u, v, d in G.edges(data=True) if u in members and v in members]
        line_count = Counter(ln for *_, d in internal for ln in d["lines"])
        measured = [n for n in members if G.nodes[n].get("punctuality") is not None]
        punct = np.mean([G.nodes[n]["punctuality"] for n in measured]) if measured else None
        worst = min(measured, key=lambda n: G.nodes[n]["punctuality"]) if measured else None
        bridges = Counter()
        for u, v, d in G.edges(data=True):
            if u in members and v not in members:
                bridges[(u, v)] += d["trips"]
        bridge_txt = "; ".join(f"{name(u)} → {name(v)}" for (u, v), _ in bridges.most_common(4))
        text = (
            f"{title}. This part of the Berlin rail network has {len(members)} stations: "
            f"{', '.join(sorted(name(n) for n in members))}. "
            f"Main hubs: {', '.join(name(h) for h in hubs[:3])}. "
            f"Lines running inside it: {', '.join(ln for ln, _ in line_count.most_common(8))}. "
            f"Average punctuality of its stations: {_pct(punct)}"
            + (f"; least punctual is {name(worst)} ({_pct(G.nodes[worst]['punctuality'])})." if worst else ".")
            + (f" Main links to other areas: {bridge_txt}." if bridge_txt else "")
        )
        docs.append(Doc(cid, "community", title, text, {"stations": sorted(name(n) for n in members)}))

    for n, d in G.nodes(data=True):
        neigh = sorted({name(v) for v in G.successors(n)} | {name(u) for u in G.predecessors(n)})
        rel = (
            "No delay data (station reconstructed from planned paths)."
            if d.get("reconstructed")
            else f"Punctuality {_pct(d['punctuality'])} (arrivals under 6 min late), mean arrival delay "
            f"{d['mean_delay_min']} min, 90th percentile {d['p90_delay_min']} min, cancellations "
            f"{_pct(d['cancellation_rate'])}, about {d['stops_per_day']} train stops per day."
        )
        text = (
            f"Station {d['name']}"
            + (f" (official name {d['official_name']}, EVA {d['eva']})" if d.get("official_name") else "")
            + f". Served by lines {', '.join(d['lines'])}. Neighbouring stations: {', '.join(neigh)}. "
            + rel
            + f" Hub rank #{rank.get(n, '?')} of {G.number_of_nodes()} by betweenness centrality."
        )
        docs.append(Doc(f"station-{n}", "station", d["name"], text, {"slug": n, "community": community_of.get(n)}))

    for line, meta in G.graph.get("lines", {}).items():
        pats = meta["patterns"]
        if not pats:
            continue
        p = pats[0]
        stops = [name(x) for x in p["stops"]]
        text = (
            f"Line {line} ({meta['mode']}), {meta['trips']} trips observed. Most common route: "
            f"{stops[0]} to {stops[-1]} via {', '.join(stops[1:-1])}."
        )
        docs.append(Doc(f"line-{line}", "line", f"Line {line}", text, {"line": line}))
    return docs, community_of


class TfidfBackend:
    def __init__(self, texts: list[str]):
        self.vec = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
            ]
        )
        self.matrix = self.vec.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return (self.matrix @ q.T).toarray().ravel() / 2.0


class EmbeddingsBackend:
    """Wraps any LangChain ``Embeddings`` implementation (dense vector retrieval)."""

    def __init__(self, texts: list[str], embeddings):
        self.emb = embeddings
        m = np.asarray(embeddings.embed_documents(texts), dtype=np.float32)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def scores(self, query: str) -> np.ndarray:
        q = np.asarray(self.emb.embed_query(query), dtype=np.float32)
        return self.matrix @ (q / np.linalg.norm(q))


class KnowledgeBase:
    def __init__(self, docs: list[Doc], community_of: dict[str, str], embeddings=None):
        self.docs = docs
        self.community_of = community_of
        self.by_id = {d.id: d for d in docs}
        texts = [f"{d.title}. {d.text}" for d in docs]
        self.backend = EmbeddingsBackend(texts, embeddings) if embeddings is not None else TfidfBackend(texts)

    @classmethod
    def from_graph(cls, G: nx.DiGraph, betweenness: dict[str, float], embeddings=None) -> KnowledgeBase:
        docs, community_of = build_documents(G, betweenness)
        return cls(docs, community_of, embeddings)

    def search(self, query: str, k: int = 4, expand: bool = True) -> list[dict]:
        scores = self.backend.scores(query)
        order = np.argsort(-scores)[:k]
        results, seen = [], set()
        for i in order:
            d = self.docs[i]
            seen.add(d.id)
            results.append({"id": d.id, "kind": d.kind, "title": d.title, "score": round(float(scores[i]), 4), "text": d.text})
        if expand:  # graph-guided expansion: attach the community of each retrieved station
            for r in list(results):
                cid = self.by_id[r["id"]].meta.get("community")
                if cid and cid not in seen:
                    seen.add(cid)
                    c = self.by_id[cid]
                    results.append(
                        {"id": c.id, "kind": c.kind, "title": c.title, "score": None, "text": c.text, "expanded_from": r["id"]}
                    )
        return results
