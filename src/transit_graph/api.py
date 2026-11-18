"""FastAPI service: a chat web page, REST endpoints for the graph tools, and a natural-language /ask endpoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import TransitAgent
from .config import get_service
from .tools import StationNotFound

STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = get_service()
    state["service"] = service
    state["agent"] = TransitAgent(service)
    provider = os.getenv("TRANSIT_LLM_PROVIDER", "offline")
    model = os.getenv("TRANSIT_LLM_MODEL", "")
    state["backend"] = f"{provider} ({model})" if model else provider
    yield
    state.clear()


app = FastAPI(title="Berlin Transit Graph Agent", version="0.2.0", lifespan=lifespan)


def _call(fn, **kwargs):
    try:
        return fn(**kwargs)
    except StationNotFound as exc:
        raise HTTPException(404, detail={"error": str(exc), "suggestions": exc.suggestions}) from exc


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["Fastest way from Spandau to Ostkreuz?"])


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[dict]
    tool_results: list[dict]
    latency_s: float


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    G = state["service"].G
    return {
        "status": "ok",
        "stations": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "period": G.graph.get("period"),
        "agent_backend": state["backend"],
    }


@app.get("/network")
def network():
    """Stations with coordinates and the edges between them, for drawing the map."""
    G = state["service"].G

    def coords(n):
        d = G.nodes[n]
        if d.get("lat") is not None:
            return d["lat"], d["lon"]
        # stations rebuilt from planned paths have no coordinates: place them between their S-Bahn neighbours
        nbrs = set(G.successors(n)) | set(G.predecessors(n))
        located = [m for m in nbrs if G.nodes[m].get("lat") is not None]
        sbahn = [
            m
            for m in located
            if any(ln.startswith("S") and not ln.startswith("SEV") for ln in (G.edges.get((n, m)) or G.edges[m, n])["lines"])
        ]
        pts = sbahn or located
        if not pts:
            return None
        return sum(G.nodes[m]["lat"] for m in pts) / len(pts), sum(G.nodes[m]["lon"] for m in pts) / len(pts)

    pos = {n: coords(n) for n in G}
    stations = [
        {
            "name": d["name"],
            "lat": pos[n][0],
            "lon": pos[n][1],
            "lines": d["lines"],
            "punctuality": d.get("punctuality"),
            "approximate": bool(d.get("reconstructed")),
        }
        for n, d in G.nodes(data=True)
        if pos[n] is not None
    ]
    edges, seen = [], set()
    for u, v, d in G.edges(data=True):
        key = frozenset((u, v))
        if key in seen or pos[u] is None or pos[v] is None:
            continue
        seen.add(key)
        lines = set(d["lines"]) | (set(G.edges[v, u]["lines"]) if G.has_edge(v, u) else set())
        edges.append({"from": G.nodes[u]["name"], "to": G.nodes[v]["name"], "lines": sorted(lines)})
    return {"stations": stations, "edges": edges}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    r = state["agent"].ask(req.question)
    return AskResponse(answer=r.answer, tool_calls=r.tool_calls, tool_results=r.tool_results, latency_s=r.latency_s)


@app.get("/stations/search")
def find_station(q: str, limit: int = Query(5, ge=1, le=20)):
    return state["service"].find_station(q, limit)


@app.get("/stations/{station}")
def station_info(station: str):
    return _call(state["service"].station_info, station=station)


@app.get("/route")
def route(
    origin: str,
    destination: str,
    optimize: Literal["fastest", "reliable"] = "fastest",
    avoid: list[str] = Query(default=[]),
):
    return _call(state["service"].plan_route, origin=origin, destination=destination, optimize=optimize, avoid_stations=avoid)


@app.get("/reachable/{station}")
def reachable(station: str, minutes: float = Query(20, gt=0, le=120)):
    return _call(state["service"].reachable_within, station=station, minutes=minutes)


@app.get("/hubs")
def hubs(metric: Literal["betweenness", "pagerank", "degree", "lines"] = "betweenness", top_k: int = 10):
    return state["service"].hub_ranking(metric, top_k)


@app.get("/delays")
def delays(kind: Literal["stations", "segments"] = "stations", top_k: int = 10):
    return state["service"].delay_hotspots(kind, top_k)


@app.get("/closure/{station}")
def closure(station: str):
    return _call(state["service"].closure_impact, station=station)


@app.get("/lines/{line}")
def line(line: str):
    return state["service"].line_info(line)
