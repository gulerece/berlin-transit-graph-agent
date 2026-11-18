from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .graph_builder import load_graph
from .tools import TransitService

DEFAULT_GRAPH = Path(os.getenv("TRANSIT_GRAPH_PATH", "data/graph.json.gz"))


@lru_cache(maxsize=1)
def get_service(path: str | None = None) -> TransitService:
    graph_path = Path(path) if path else DEFAULT_GRAPH
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found at {graph_path}. Build it first: `make data` (see README).")
    return TransitService(load_graph(graph_path))
