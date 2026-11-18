"""Graph tools over the transit knowledge graph.

Every public method returns plain JSON-serialisable dicts so the same functions
can be exposed to the LangGraph agent, the MCP server and the REST API.
"""

from __future__ import annotations

import heapq
import itertools
import math
import re
import unicodedata
from functools import cached_property
from typing import Literal

import networkx as nx
from rapidfuzz import fuzz, process

from .knowledge import KnowledgeBase

CANCELLATION_REPLAN_MIN = 20  # expected cost of a cancelled train when optimising for reliability
ALIASES = {
    "hauptbahnhof": "Hbf",
    "berlin hbf": "Hbf",
    "central station": "Hbf",
    "main station": "Hbf",
    "alex": "Alexanderplatz",
    "zoo": "Zoologischer Garten",
    "bahnhof zoo": "Zoologischer Garten",
    "ostbf": "Ostbahnhof",
    "east station": "Ostbahnhof",
    "messe nord": "Messe Nord/ZOB (Witzleben)",
    "zob": "Messe Nord/ZOB (Witzleben)",
    "yorckstrasse": "Yorckstr.(S2)",
    "yorckstraße": "Yorckstr.(S2)",
    "grossgorschenstrasse": "Yorckstr.(S1)",
    "großgörschenstraße": "Yorckstr.(S1)",
}


class StationNotFound(ValueError):
    def __init__(self, query: str, suggestions: list[str]):
        self.query, self.suggestions = query, suggestions
        super().__init__(f"No station matches '{query}'. Did you mean: {', '.join(suggestions)}?")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("ß", "ss")
    text = re.sub(r"\b(berlin|s\+u|bhf|bahnhof|station)\b", " ", text)
    text = re.sub(r"\((s|s-bahn)\)", " ", text)
    text = text.replace("str.", "strasse").replace("str ", "strasse ")
    text = re.sub(r"[^\w/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class TransitService:
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self._names = {n: d.get("name", n) for n, d in graph.nodes(data=True)}
        self._choices: dict[str, str] = {}
        for n, d in graph.nodes(data=True):
            for label in filter(None, [d.get("name"), d.get("official_name"), n.replace("_", " ")]):
                self._choices[_norm(label)] = n
        for alias, target in ALIASES.items():
            slug = next((n for n, nm in self._names.items() if nm == target), None)
            if slug:
                self._choices[_norm(alias)] = slug

    # ------------------------------------------------------------------ lookup
    def resolve(self, query: str) -> str:
        if query in self.G:
            return query
        q = _norm(query)
        if q in self._choices:
            return self._choices[q]
        hits = process.extract(q, list(self._choices), scorer=fuzz.WRatio, limit=5)
        if hits and hits[0][1] >= 86:
            return self._choices[hits[0][0]]
        suggestions = list(dict.fromkeys(self._names[self._choices[h[0]]] for h in hits))[:3]
        raise StationNotFound(query, suggestions)

    def name(self, slug: str) -> str:
        return self._names.get(slug, slug)

    def find_station(self, query: str, limit: int = 5) -> dict:
        """Fuzzy-search stations by name."""
        q = _norm(query)
        hits = process.extract(q, list(self._choices), scorer=fuzz.WRatio, limit=limit * 3)
        seen, out = set(), []
        for label, score, _ in hits:
            slug = self._choices[label]
            if slug in seen:
                continue
            seen.add(slug)
            d = self.G.nodes[slug]
            out.append({"station": d["name"], "id": slug, "score": round(score, 1), "lines": d["lines"]})
            if len(out) == limit:
                break
        return {"query": query, "matches": out}

    # ------------------------------------------------------------------ station facts
    def station_info(self, station: str) -> dict:
        """Lines, neighbours, reliability statistics and graph role of a station."""
        s = self.resolve(station)
        d = self.G.nodes[s]
        neighbours = sorted(
            (
                {"station": self.name(v), "minutes": e["minutes"], "lines": sorted(e["lines"])}
                for v, e in ((v, self.G.edges[s, v]) for v in self.G.successors(s))
            ),
            key=lambda x: x["minutes"],
        )
        return {
            "station": d["name"],
            "official_name": d.get("official_name"),
            "eva": d.get("eva"),
            "location": {"lat": d.get("lat"), "lon": d.get("lon")},
            "lines": d["lines"],
            "neighbours": neighbours,
            "reliability": None
            if d.get("reconstructed")
            else {
                "punctuality_pct": round(100 * d["punctuality"], 1),
                "mean_arrival_delay_min": d["mean_delay_min"],
                "p90_arrival_delay_min": d["p90_delay_min"],
                "cancellation_rate_pct": round(100 * d["cancellation_rate"], 2),
                "stops_per_day": d["stops_per_day"],
            },
            "hub_rank_betweenness": self._rank("betweenness").get(s),
            "community": self.kb.community_of.get(s),
            "note": "Station reconstructed from planned paths; no delay data." if d.get("reconstructed") else None,
        }

    def line_info(self, line: str) -> dict:
        """Stops of a line (most common pattern per direction) and its reliability."""
        lines = self.G.graph["lines"]
        key = next((k for k in lines if k.casefold() == line.casefold().replace(" ", "")), None)
        key = key or next((k for k in lines if k.casefold() == line.casefold()), None)
        if key is None:
            return {"error": f"Unknown line '{line}'.", "known_lines": sorted(lines)[:40]}
        meta = lines[key]
        edges = [(u, v, e["lines"][key]) for u, v, e in self.G.edges(data=True) if key in e["lines"]]
        trips = sum(d["trips"] for *_, d in edges) or 1
        return {
            "line": key,
            "mode": meta["mode"],
            "trips_observed": meta["trips"],
            "patterns": [
                {
                    "from": self.name(p["stops"][0]),
                    "to": self.name(p["stops"][-1]),
                    "stops": [self.name(x) for x in p["stops"]],
                    "trips": p["trips"],
                }
                for p in meta["patterns"]
            ],
            "mean_added_delay_per_segment_min": round(sum(d["added_delay"] * d["trips"] for *_, d in edges) / trips, 3),
            "cancellation_rate_pct": round(100 * sum(d["cancellation_rate"] * d["trips"] for *_, d in edges) / trips, 2),
        }

    def nearest_station(self, lat: float, lon: float, k: int = 3) -> dict:
        """Closest stations to a coordinate (haversine distance)."""
        dist = []
        for n, d in self.G.nodes(data=True):
            if d.get("lat") is None:
                continue
            dist.append((_haversine_km(lat, lon, d["lat"], d["lon"]), n))
        dist.sort()
        return {
            "query": {"lat": lat, "lon": lon},
            "nearest": [{"station": self.name(n), "distance_km": round(km, 3)} for km, n in dist[:k]],
        }

    # ------------------------------------------------------------------ routing
    def _search(
        self,
        origin: str,
        target: str | None = None,
        *,
        cutoff: float | None = None,
        avoid: frozenset[str] = frozenset(),
        transfer_penalty: float = 5.0,
        optimize: str = "fastest",
        exclude_modes: frozenset[str] = frozenset(),
    ):
        """Dijkstra over (station, line) states so that line changes can be penalised."""
        tie = itertools.count()
        start = (origin, None)
        best = {start: 0.0}
        parent: dict = {start: None}
        heap = [(0.0, next(tie), start)]
        settled_station: dict[str, float] = {}
        while heap:
            cost, _, state = heapq.heappop(heap)
            if cost > best.get(state, math.inf):
                continue
            station, line = state
            settled_station.setdefault(station, cost)
            if station == target:
                return state, best, parent, settled_station
            for nxt in self.G.successors(station):
                if nxt in avoid:
                    continue
                for ln, d in self.G.edges[station, nxt]["lines"].items():
                    if d["mode"] in exclude_modes:
                        continue
                    w = d["minutes"]
                    if optimize == "reliable":
                        w += max(d["added_delay"], 0.0) + d["cancellation_rate"] * CANCELLATION_REPLAN_MIN
                    if line is not None and ln != line:
                        w += transfer_penalty
                    new = cost + w
                    if cutoff is not None and new > cutoff:
                        continue
                    ns = (nxt, ln)
                    if new < best.get(ns, math.inf):
                        best[ns], parent[ns] = new, state
                        heapq.heappush(heap, (new, next(tie), ns))
        return None, best, parent, settled_station

    def plan_route(
        self,
        origin: str,
        destination: str,
        optimize: Literal["fastest", "reliable"] = "fastest",
        avoid_stations: list[str] | None = None,
        transfer_penalty_min: float = 5.0,
        exclude_modes: list[str] | None = None,
    ) -> dict:
        """Best route between two stations, either fastest or most reliable (delay/cancellation aware)."""
        o, t = self.resolve(origin), self.resolve(destination)
        if o == t:
            return {"error": "Origin and destination are the same station."}
        avoid = frozenset(self.resolve(a) for a in (avoid_stations or [])) - {o, t}
        end, best, parent, _ = self._search(
            o,
            t,
            avoid=avoid,
            transfer_penalty=transfer_penalty_min,
            optimize=optimize,
            exclude_modes=frozenset(exclude_modes or []),
        )
        if end is None:
            return {
                "origin": self.name(o),
                "destination": self.name(t),
                "found": False,
                "message": "No connection in the observed network.",
            }
        states = []
        while end is not None:
            states.append(end)
            end = parent[end]
        states.reverse()
        return self._describe(states, optimize, transfer_penalty_min)

    def _describe(self, states, optimize, transfer_penalty) -> dict:
        legs, ride, delay, p_ok = [], 0.0, 0.0, 1.0
        for (a, _), (b, ln) in zip(states, states[1:]):
            d = self.G.edges[a, b]["lines"][ln]
            ride += d["minutes"]
            delay += max(d["added_delay"], 0.0)
            p_ok *= 1 - d["cancellation_rate"]
            if legs and legs[-1]["line"] == ln:
                legs[-1]["to"] = self.name(b)
                legs[-1]["stops"] += 1
                legs[-1]["minutes"] = round(legs[-1]["minutes"] + d["minutes"], 1)
            else:
                legs.append(
                    {
                        "line": ln,
                        "mode": d["mode"],
                        "from": self.name(a),
                        "to": self.name(b),
                        "stops": 1,
                        "minutes": round(d["minutes"], 1),
                    }
                )
        transfers = len(legs) - 1
        return {
            "found": True,
            "origin": self.name(states[0][0]),
            "destination": self.name(states[-1][0]),
            "optimize": optimize,
            "total_minutes": round(ride + transfers * transfer_penalty, 1),
            "in_vehicle_minutes": round(ride, 1),
            "transfers": transfers,
            "transfer_buffer_min": transfer_penalty,
            "expected_extra_delay_min": round(delay, 1),
            "probability_no_cancellation_pct": round(100 * p_ok, 1),
            "legs": legs,
            "path": [self.name(s) for s, _ in states],
        }

    def reachable_within(self, station: str, minutes: float = 20, transfer_penalty_min: float = 5.0) -> dict:
        """Isochrone: all stations reachable from a station within a time budget."""
        s = self.resolve(station)
        _, _, _, settled = self._search(s, cutoff=minutes, transfer_penalty=transfer_penalty_min)
        reach = sorted(((c, n) for n, c in settled.items() if n != s))
        return {
            "origin": self.name(s),
            "budget_min": minutes,
            "count": len(reach),
            "stations": [{"station": self.name(n), "minutes": round(c, 1)} for c, n in reach],
        }

    # ------------------------------------------------------------------ network analysis
    @cached_property
    def centrality(self) -> dict[str, dict[str, float]]:
        G = self.G
        weighted = nx.DiGraph()
        weighted.add_edges_from((u, v, {"w": d["minutes"]}) for u, v, d in G.edges(data=True))
        weighted.add_nodes_from(G)
        return {
            "betweenness": nx.betweenness_centrality(weighted, weight="w", normalized=True),
            "pagerank": nx.pagerank(G, weight="trips"),
            "degree": {n: float(len(set(G.successors(n)) | set(G.predecessors(n)))) for n in G},
            "lines": {n: float(len(G.nodes[n]["lines"])) for n in G},
        }

    def _rank(self, metric: str) -> dict[str, int]:
        scores = self.centrality[metric]
        return {n: i + 1 for i, n in enumerate(sorted(scores, key=scores.get, reverse=True))}

    def hub_ranking(self, metric: Literal["betweenness", "pagerank", "degree", "lines"] = "betweenness", top_k: int = 10) -> dict:
        """Most important stations by a graph-centrality metric."""
        if metric not in self.centrality:
            return {"error": f"metric must be one of {sorted(self.centrality)}"}
        scores = self.centrality[metric]
        top = sorted(scores, key=scores.get, reverse=True)[:top_k]
        explain = {
            "betweenness": "share of fastest paths between all station pairs that pass through the station",
            "pagerank": "PageRank on the train-frequency-weighted network",
            "degree": "number of directly connected neighbouring stations",
            "lines": "number of distinct lines serving the station",
        }[metric]
        return {
            "metric": metric,
            "meaning": explain,
            "ranking": [{"rank": i + 1, "station": self.name(n), "score": round(scores[n], 4)} for i, n in enumerate(top)],
        }

    def delay_hotspots(self, kind: Literal["stations", "segments"] = "stations", top_k: int = 10, min_trips: int = 100) -> dict:
        """Where delays happen: least punctual stations, or segments that add the most delay."""
        if kind == "stations":
            rows = [
                (d["punctuality"], n)
                for n, d in self.G.nodes(data=True)
                if not d.get("reconstructed") and d["stops_per_day"] and d["stops_per_day"] * 6 >= min_trips
            ]
            rows.sort()
            return {
                "kind": kind,
                "definition": "punctual = arrival less than 6 minutes late (DB standard)",
                "hotspots": [
                    {
                        "station": self.name(n),
                        "punctuality_pct": round(100 * p, 1),
                        "mean_arrival_delay_min": self.G.nodes[n]["mean_delay_min"],
                        "cancellation_rate_pct": round(100 * self.G.nodes[n]["cancellation_rate"], 2),
                    }
                    for p, n in rows[:top_k]
                ],
            }
        rows = [
            (d["added_delay"], u, v, d)
            for u, v, d in self.G.edges(data=True)
            if d["trips"] >= min_trips and not d.get("reconstructed")
        ]
        rows.sort(key=lambda r: r[0], reverse=True)
        return {
            "kind": kind,
            "definition": "mean minutes of delay gained between the two stops",
            "hotspots": [
                {
                    "from": self.name(u),
                    "to": self.name(v),
                    "added_delay_min": round(a, 2),
                    "trips": d["trips"],
                    "lines": sorted(d["lines"]),
                }
                for a, u, v, d in rows[:top_k]
            ],
        }

    def closure_impact(self, station: str, transfer_penalty_min: float = 5.0) -> dict:
        """What-if analysis: how journeys between a station's neighbours change if it closes."""
        s = self.resolve(station)
        preds, succs = set(self.G.predecessors(s)) - {s}, set(self.G.successors(s)) - {s}
        pairs, broken, deltas = [], 0, []
        for a, b in itertools.product(sorted(preds), sorted(succs)):
            if a == b:
                continue
            before = self.plan_route(a, b, transfer_penalty_min=transfer_penalty_min)
            if not before.get("found") or s not in [self.resolve(x) for x in before["path"]]:
                continue  # this pair never used the station
            after = self.plan_route(a, b, avoid_stations=[s], transfer_penalty_min=transfer_penalty_min)
            if not after.get("found"):
                broken += 1
                pairs.append({"from": self.name(a), "to": self.name(b), "before_min": before["total_minutes"], "after_min": None})
                continue
            delta = after["total_minutes"] - before["total_minutes"]
            deltas.append(delta)
            pairs.append(
                {
                    "from": self.name(a),
                    "to": self.name(b),
                    "before_min": before["total_minutes"],
                    "after_min": after["total_minutes"],
                    "extra_min": round(delta, 1),
                    "detour_via": after["path"][1:-1][:6],
                }
            )
        pairs.sort(key=lambda p: (p["after_min"] is not None, -(p.get("extra_min") or 0)))
        und = self.G.to_undirected()
        h = und.copy()
        h.remove_node(s)
        return {
            "station": self.name(s),
            "is_articulation_point": s in set(nx.articulation_points(und)),
            "components_after_closure": nx.number_connected_components(h),
            "affected_neighbour_pairs": len(pairs),
            "disconnected_pairs": broken,
            "mean_extra_minutes": round(sum(deltas) / len(deltas), 1) if deltas else 0.0,
            "worst_pairs": pairs[:8],
            "betweenness_rank": self._rank("betweenness")[s],
        }

    # ------------------------------------------------------------------ GraphRAG
    @cached_property
    def kb(self) -> KnowledgeBase:
        return KnowledgeBase.from_graph(self.G, self.centrality["betweenness"])

    def search_network_knowledge(self, query: str, k: int = 4) -> dict:
        """GraphRAG search over station profiles, line descriptions and network-community summaries."""
        return {"query": query, "results": self.kb.search(query, k=k)}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
