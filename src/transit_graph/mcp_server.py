"""MCP server exposing the transit graph tools to any MCP client
(Claude Desktop, Claude Code, Cursor, ...).

    python -m transit_graph.mcp_server                    # stdio
    python -m transit_graph.mcp_server --http --port 8001 # streamable HTTP
"""

from __future__ import annotations

import argparse
from typing import Literal

from mcp.server.mcpserver import MCPServer

from .config import get_service
from .tools import StationNotFound

server = MCPServer(
    name="berlin-transit-graph",
    instructions=(
        "Tools over a knowledge graph of the Berlin S-Bahn and regional rail network, built from "
        "six weeks of real Deutsche Bahn timetable and delay data (Sep–Oct 2025). Station names are resolved fuzzily."
    ),
)


def _safe(fn, **kwargs) -> dict:
    try:
        return fn(**kwargs)
    except StationNotFound as exc:
        return {"error": str(exc), "suggestions": exc.suggestions}


@server.tool()
def plan_route(
    origin: str, destination: str, optimize: Literal["fastest", "reliable"] = "fastest", avoid_stations: list[str] | None = None
) -> dict:
    """Best route between two stations; optimize='reliable' accounts for observed delays and cancellations."""
    return _safe(
        get_service().plan_route, origin=origin, destination=destination, optimize=optimize, avoid_stations=avoid_stations
    )


@server.tool()
def station_info(station: str) -> dict:
    """Lines, neighbours, punctuality and hub rank of a station."""
    return _safe(get_service().station_info, station=station)


@server.tool()
def reachable_within(station: str, minutes: float = 20) -> dict:
    """Isochrone: stations reachable from a station within a time budget (minutes)."""
    return _safe(get_service().reachable_within, station=station, minutes=minutes)


@server.tool()
def hub_ranking(metric: Literal["betweenness", "pagerank", "degree", "lines"] = "betweenness", top_k: int = 10) -> dict:
    """Most important stations by graph centrality."""
    return get_service().hub_ranking(metric=metric, top_k=top_k)


@server.tool()
def delay_hotspots(kind: Literal["stations", "segments"] = "stations", top_k: int = 10) -> dict:
    """Least punctual stations, or track segments where trains gain the most delay."""
    return get_service().delay_hotspots(kind=kind, top_k=top_k)


@server.tool()
def closure_impact(station: str) -> dict:
    """What-if: effect of closing a station on journeys between its neighbours."""
    return _safe(get_service().closure_impact, station=station)


@server.tool()
def line_info(line: str) -> dict:
    """Stops and reliability of a line, e.g. 'S41' or 'RE1'."""
    return get_service().line_info(line=line)


@server.tool()
def nearest_station(lat: float, lon: float, k: int = 3) -> dict:
    """Closest stations to a latitude/longitude."""
    return get_service().nearest_station(lat=lat, lon=lon, k=k)


@server.tool()
def search_network_knowledge(query: str, k: int = 4) -> dict:
    """GraphRAG search over community summaries, station profiles and line descriptions."""
    return get_service().search_network_knowledge(query=query, k=k)


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args()
    get_service()  # fail fast if the graph is missing
    if a.http:
        server.run("streamable-http", host=a.host, port=a.port)
    else:
        server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
