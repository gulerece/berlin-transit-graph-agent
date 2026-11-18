"""FastAPI service: REST endpoints for the graph tools plus a natural-language /ask endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .agent import TransitAgent
from .config import get_service
from .tools import StationNotFound

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = get_service()
    state["service"] = service
    state["agent"] = TransitAgent(service)
    yield
    state.clear()


app = FastAPI(title="Berlin Transit Graph Agent", version="0.1.0", lifespan=lifespan)


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
    latency_s: float


@app.get("/health")
def health():
    G = state["service"].G
    return {"status": "ok", "stations": G.number_of_nodes(), "edges": G.number_of_edges(), "period": G.graph.get("period")}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    r = state["agent"].ask(req.question)
    return AskResponse(answer=r.answer, tool_calls=r.tool_calls, latency_s=r.latency_s)


@app.get("/stations/search")
def find_station(q: str, limit: int = Query(5, ge=1, le=20)):
    return state["service"].find_station(q, limit)


@app.get("/stations/{station}")
def station_info(station: str):
    return _call(state["service"].station_info, station=station)


@app.get("/route")
def route(
    origin: str, destination: str, optimize: Literal["fastest", "reliable"] = "fastest", avoid: list[str] = Query(default=[])
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
