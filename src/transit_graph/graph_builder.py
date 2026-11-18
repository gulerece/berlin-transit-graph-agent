"""Build a delay-aware transit knowledge graph from the ingested Parquet tables.

Nodes are stations (keyed by slug) with reliability statistics.
Directed edges connect consecutive stops of real trips and carry, per line,
the median planned travel time plus the mean delay *added* on that segment.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .ingest import display_name

log = logging.getLogger(__name__)

PUNCTUAL_THRESHOLD_MIN = 6  # DB counts a stop as punctual if it is < 6 minutes late
MIN_EDGE_MINUTES = 0.5  # planned times are minute-rounded; avoid zero-cost hops
MIN_LINE_TRIPS = 3  # a line must be observed on a segment at least this often


def stop_events(processed_dir: Path) -> pd.DataFrame:
    """Join planned stops with their latest real-time change and compute delays."""
    processed_dir = Path(processed_dir)
    stops = pd.read_parquet(processed_dir / "stops.parquet")
    changes = pd.read_parquet(processed_dir / "changes.parquet")
    ev = stops.merge(changes, on="id", how="left")
    del stops, changes
    for col in ("slug", "line", "mode"):
        ev[col] = ev[col].astype(str)
    ev["cancelled"] = (ev["cs_ar"] == "c") | (ev["cs_dp"] == "c")
    # no change record => the train ran as planned (the DB API only reports deviations)
    for kind in ("ar", "dp"):
        delay = (ev[f"ct_{kind}"] - ev[f"pt_{kind}"]).dt.total_seconds() / 60
        ev[f"delay_{kind}"] = delay.fillna(0).clip(lower=0).where(ev[f"pt_{kind}"].notna())
    return ev


def build_graph(processed_dir: Path, min_line_trips: int = MIN_LINE_TRIPS) -> nx.DiGraph:
    processed_dir = Path(processed_dir)
    ev = stop_events(processed_dir)
    stations = pd.read_parquet(processed_dir / "stations.parquet").set_index("slug")

    G = nx.DiGraph(name="Berlin S-Bahn + regional rail (DBahn-Berlin)")
    G.graph["period"] = [str(ev["pt_dp"].min()), str(ev["pt_dp"].max())]
    G.graph["n_stop_events"] = int(len(ev))

    _add_nodes(G, ev, stations)
    name_to_slug = {d["name"]: n for n, d in G.nodes(data=True)}
    seg = _segments(ev, name_to_slug)
    _add_edges(G, seg, min_line_trips)
    for n in G.nodes:  # stations served only via repaired segments still list their lines
        incident = {ln for _, _, d in G.in_edges(n, data=True) for ln in d["lines"]}
        incident |= {ln for _, _, d in G.out_edges(n, data=True) for ln in d["lines"]}
        G.nodes[n]["lines"] = sorted(set(G.nodes[n].get("lines", [])) | incident, key=_line_sort_key)
    G.graph["reconstructed_segment_share"] = round(float(seg["reconstructed"].mean()), 4)
    G.graph["lines"] = _line_patterns(ev, G)
    log.info("graph: %d stations, %d edges, %d lines", G.number_of_nodes(), G.number_of_edges(), len(G.graph["lines"]))
    return G


def _add_nodes(G: nx.DiGraph, ev: pd.DataFrame, stations: pd.DataFrame) -> None:
    n_days = max(1, ev["pt_dp"].dt.normalize().nunique())
    for slug, grp in ev.groupby("slug"):
        ran = grp[~grp["cancelled"]]
        arr = ran["delay_ar"].dropna()
        meta = stations.loc[slug] if slug in stations.index else None
        G.add_node(
            slug,
            name=meta["name"] if meta is not None else slug,
            official_name=meta["station"] if meta is not None else slug,
            eva=meta["eva"] if meta is not None else None,
            lat=float(meta["lat"]) if meta is not None and pd.notna(meta["lat"]) else None,
            lon=float(meta["lon"]) if meta is not None and pd.notna(meta["lon"]) else None,
            lines=sorted(grp["line"].unique()),
            modes=sorted(grp["mode"].unique()),
            stops_per_day=round(len(grp) / n_days, 1),
            mean_delay_min=round(float(arr.mean()), 2) if len(arr) else 0.0,
            p90_delay_min=round(float(arr.quantile(0.9)), 1) if len(arr) else 0.0,
            punctuality=round(float((arr < PUNCTUAL_THRESHOLD_MIN).mean()), 4) if len(arr) else 1.0,
            cancellation_rate=round(float(grp["cancelled"].mean()), 4),
            reconstructed=False,
        )


def _segments(ev: pd.DataFrame, name_to_slug: dict[str, str]) -> pd.DataFrame:
    """Consecutive stop pairs per trip, with gaps repaired from the planned path.

    Some station records (e.g. the S-Bahn platforms at Alexanderplatz) are missing
    from the dataset, so a trip's stop index jumps (Jannowitzbrücke -> Hackescher
    Markt). The planned path attribute (``ppth``) still names the skipped stations,
    so we split the segment through them and interpolate the travel time.
    """
    ev = ev.sort_values(["trip_key", "stop_idx"])
    nxt = ev.groupby("trip_key").shift(-1)
    seg = pd.DataFrame(
        {
            "u": ev["slug"],
            "v": nxt["slug"],
            "line": ev["line"],
            "mode": ev["mode"],
            "dep": ev["pt_dp"],
            "arr": nxt["pt_ar"].fillna(nxt["pt_dp"]),
            "dep_delay": ev["delay_dp"],
            "arr_delay": nxt["delay_ar"],
            "cancelled": ev["cancelled"] | nxt["cancelled"].fillna(False).astype(bool),
            "gap": nxt["stop_idx"] - ev["stop_idx"],
            "next_path": ev["next_path"],
        }
    ).dropna(subset=["v", "dep", "arr"])
    seg = seg[seg["u"] != seg["v"]]
    seg["minutes"] = (seg["arr"] - seg["dep"]).dt.total_seconds() / 60
    seg = seg[(seg["minutes"] >= 0) & (seg["minutes"] <= 90)]
    seg["added_delay"] = (seg["arr_delay"] - seg["dep_delay"]).where(~seg["cancelled"])

    ok = seg[seg["gap"] <= 1]
    rows, dropped = [], 0
    for r in seg[seg["gap"] > 1].itertuples(index=False):
        skipped = r.next_path.split("|")[: int(r.gap) - 1] if isinstance(r.next_path, str) else []
        chain = [name_to_slug.get(display_name(n)) or _reconstructed_slug(n) for n in skipped]
        if len(chain) != int(r.gap) - 1 or any(c is None for c in chain):
            dropped += 1  # a hole we cannot repair; keeping it would invent a non-existent express link
            continue
        hops = [r.u, *chain, r.v]
        k = len(hops) - 1
        for a, b in zip(hops, hops[1:]):
            if a != b:
                rows.append((a, b, r.line, r.mode, r.minutes / k, r.added_delay / k, r.cancelled, True))
    cols = ["u", "v", "line", "mode", "minutes", "added_delay", "cancelled", "reconstructed"]
    fixed = pd.DataFrame(rows, columns=cols)
    log.info("segments: %d direct, %d repaired hops, %d unrepairable gaps dropped", len(ok), len(fixed), dropped)
    ok = ok.assign(reconstructed=False)[cols]
    return pd.concat([ok, fixed], ignore_index=True)


def _reconstructed_slug(raw_name: str) -> str | None:
    """Stations named only in planned paths (e.g. 'Berlin Wuhletal'). Skip bus stops."""
    if "," in raw_name or "[" in raw_name or not raw_name.startswith("Berlin"):
        return None
    return "recon:" + display_name(raw_name)


def _add_edges(G: nx.DiGraph, seg: pd.DataFrame, min_line_trips: int = MIN_LINE_TRIPS) -> None:
    agg = (
        seg.groupby(["u", "v", "line"])
        .agg(
            minutes=("minutes", "median"),
            trips=("minutes", "size"),
            mode=("mode", "first"),
            added_delay=("added_delay", "mean"),
            cancellation_rate=("cancelled", "mean"),
            reconstructed=("reconstructed", "mean"),
        )
        .reset_index()
    )
    agg = agg[agg["trips"] >= min_line_trips]  # one-off observations are noise (diversions, data glitches)
    for (u, v), grp in agg.groupby(["u", "v"]):
        lines = {
            r.line: {
                "minutes": round(max(MIN_EDGE_MINUTES, float(r.minutes)), 2),
                "trips": int(r.trips),
                "mode": r.mode,
                "added_delay": round(float(np.nan_to_num(r.added_delay)), 3),
                "cancellation_rate": round(float(r.cancellation_rate), 4),
            }
            for r in grp.itertuples()
        }
        trips = sum(d["trips"] for d in lines.values())
        G.add_edge(
            u,
            v,
            lines=lines,
            minutes=min(d["minutes"] for d in lines.values()),
            trips=trips,
            added_delay=round(sum(d["added_delay"] * d["trips"] for d in lines.values()) / trips, 3),
            cancellation_rate=round(sum(d["cancellation_rate"] * d["trips"] for d in lines.values()) / trips, 4),
            reconstructed=bool((grp["reconstructed"] > 0.5).all()),
        )
    for n in G.nodes:
        if n.startswith("recon:") and "name" not in G.nodes[n]:
            G.nodes[n].update(
                name=n.split(":", 1)[1],
                official_name=None,
                eva=None,
                lat=None,
                lon=None,
                lines=sorted({ln for _, _, d in G.in_edges(n, data=True) for ln in d["lines"]}),
                modes=[],
                stops_per_day=None,
                mean_delay_min=None,
                p90_delay_min=None,
                punctuality=None,
                cancellation_rate=None,
                reconstructed=True,
            )


def _line_patterns(ev: pd.DataFrame, G: nx.DiGraph) -> dict[str, dict]:
    """Most frequent stop sequences per line (usually one per direction)."""
    seqs = ev.sort_values(["trip_key", "stop_idx"]).groupby(["line", "trip_key"])["slug"].agg(tuple)
    out = {}
    for line, trips in seqs.groupby(level=0):
        counts = Counter(t for t in trips if len(t) >= 2)
        patterns, seen_ends = [], set()
        for seq, n in counts.most_common():
            ends = frozenset((seq[0], seq[-1]))
            if ends in seen_ends and len(patterns) >= 1:
                continue
            seen_ends.add(ends)
            patterns.append({"stops": list(seq), "trips": n})
            if len(patterns) == 2:
                break
        mode = ev.loc[ev["line"] == line, "mode"].iat[0]
        out[line] = {"mode": mode, "trips": int(len(trips)), "patterns": patterns}
    return out


def _line_sort_key(line: str) -> tuple:
    head = "".join(ch for ch in line if not ch.isdigit())
    digits = "".join(ch for ch in line if ch.isdigit())
    return ({"S": 0, "RE": 1, "RB": 2}.get(head, 3), head, int(digits) if digits else 0, line)


def save_graph(G: nx.DiGraph, path: Path) -> None:
    """Save as gzipped node-link JSON (portable, diffable, no pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G, edges="edges")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))


def load_graph(path: Path) -> nx.DiGraph:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        return nx.node_link_graph(json.load(fh), directed=True, edges="edges")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level="INFO")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", default=Path("data/processed"), type=Path)
    p.add_argument("--out", default=Path("data/graph.json.gz"), type=Path)
    a = p.parse_args()
    save_graph(build_graph(a.processed), a.out)
