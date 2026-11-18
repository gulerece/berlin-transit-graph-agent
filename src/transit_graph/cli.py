"""Command line entry point: `transit-graph build|ask|chat`."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from pathlib import Path

from .agent import TransitAgent, make_llm
from .config import get_service
from .graph_builder import build_graph, save_graph
from .ingest import ingest


def main() -> None:
    p = argparse.ArgumentParser(prog="transit-graph")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="ingest raw DBahn-Berlin archives and build the graph")
    b.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/DBahn-berlin"),
        help="folder with timetables/, timetable_changes/ and station_data.json",
    )
    b.add_argument(
        "--weeks",
        nargs="*",
        default=["250902_250909"],
        help="archive names to use (default: the first week); 'all' for every week",
    )
    b.add_argument("--out", type=Path, default=Path("data"))

    for name in ("ask", "chat"):
        s = sub.add_parser(name)
        s.add_argument("question", nargs="?" if name == "chat" else None)
        s.add_argument("--provider", default=None, help="offline | ollama | openai | anthropic")
        s.add_argument("--model", default=None)
        s.add_argument("--graph", default=None)
        s.add_argument("--trace", action="store_true", help="print tool calls")

    a = p.parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")

    if a.cmd == "build":
        processed = a.out / "processed"
        tt_dir, ch_dir = a.raw / "timetables", a.raw / "timetable_changes"
        weeks = sorted(x.name.removesuffix(".tar.gz") for x in tt_dir.glob("*.tar.gz")) if a.weeks == ["all"] else a.weeks
        staging = a.out / "staging"
        for kind, src in (("timetables", tt_dir), ("timetable_changes", ch_dir)):
            (staging / kind).mkdir(parents=True, exist_ok=True)
            for old in (staging / kind).glob("*.tar.gz"):
                old.unlink()
            for w in weeks:
                (staging / kind / f"{w}.tar.gz").symlink_to((src / f"{w}.tar.gz").resolve())
        # run ingestion in a child process so its parsing memory is returned to the OS before graph building
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            counts = pool.apply(
                ingest, (staging / "timetables", staging / "timetable_changes", a.raw / "station_data.json", processed)
            )
        print(counts)
        G = build_graph(processed)
        save_graph(G, a.out / "graph.json.gz")
        print(f"graph saved: {G.number_of_nodes()} stations, {G.number_of_edges()} edges -> {a.out / 'graph.json.gz'}")
        return

    service = get_service(a.graph)
    agent = TransitAgent(service, make_llm(service, a.provider, a.model))

    def answer(q: str) -> None:
        r = agent.ask(q)
        if a.trace:
            print("tools:", json.dumps(r.tool_calls, ensure_ascii=False), f"({r.latency_s}s)")
        print(r.answer)

    if a.cmd == "ask":
        answer(a.question)
        return
    print("Ask about the Berlin rail network (Ctrl-D to quit).")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if q:
            answer(q)


if __name__ == "__main__":  # pragma: no cover
    main()
