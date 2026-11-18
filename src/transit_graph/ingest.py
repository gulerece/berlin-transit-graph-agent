"""Ingest DBahn-Berlin timetable + change XML snapshots into tidy Parquet tables.

Reads the weekly ``.tar.gz`` archives directly (no extraction needed) or an
already-extracted directory tree. Produces:

* ``stops.parquet``    – one row per planned stop event (trip x station)
* ``changes.parquet``  – latest known actual times / cancellation flags per stop event
* ``stations.parquet`` – station slug, display name, EVA number, coordinates

The XML format is the DB Timetables API format::

    <timetable station="Berlin Alexanderplatz">
      <s id="<trip>-<start YYMMDDHHMM>-<stop index>">
        <tl c="S" n="1785" .../>              # category + train number
        <ar pt="2509081359" l="5" .../>       # planned arrival, line
        <dp pt="2509081403" l="5" .../>       # planned departure
      </s>
    </timetable>
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TIME_FMT = "%y%m%d%H%M"
PATH_KEEP = 12  # stations of the planned path kept per stop (gaps in the data are at most a few stops)
STOP_COLUMNS = [
    "id",
    "trip_key",
    "stop_idx",
    "slug",
    "station",
    "category",
    "train_number",
    "line_raw",
    "pt_ar",
    "pt_dp",
    "platform",
    "next_path",
]


def _iter_xml(source: Path) -> Iterator[tuple[str, str, bytes]]:
    """Yield (snapshot, file_name, xml_bytes) from a directory or .tar.gz archives."""
    source = Path(source)
    archives = sorted(source.glob("*.tar.gz")) if source.is_dir() else [source]
    if source.is_dir() and not archives:
        for path in sorted(source.rglob("*.xml")):
            yield path.parent.name, path.name, path.read_bytes()
        return
    for archive in archives:
        log.info("reading %s", archive.name)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".xml"):
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                parts = Path(member.name).parts
                yield parts[-2] if len(parts) > 1 else "", parts[-1], fh.read()


def _slug(file_name: str) -> str:
    return file_name.rsplit("_", 1)[0]


def parse_timetables(source: Path) -> pd.DataFrame:
    rows = []
    for _snapshot, name, payload in _iter_xml(source):
        root = ET.parse(io.BytesIO(payload)).getroot()
        station = root.get("station")
        if station is None:  # a handful of empty snapshot files exist
            continue
        slug = _slug(name)
        for s in root.findall("s"):
            sid = s.get("id")
            tl = s.find("tl")
            if sid is None or tl is None:
                continue
            ar, dp = s.find("ar"), s.find("dp")
            ev = dp if dp is not None else ar
            trip_key, _, idx = sid.rpartition("-")
            path = dp.get("ppth") if dp is not None else None
            rows.append(
                (
                    sid,
                    trip_key,
                    int(idx) if idx.isdigit() else -1,
                    slug,
                    station,
                    tl.get("c"),
                    tl.get("n"),
                    ev.get("l") if ev is not None else None,
                    ar.get("pt") if ar is not None else None,
                    dp.get("pt") if dp is not None else None,
                    ev.get("pp") if ev is not None else None,
                    # planned path = upcoming stations ("A|B|C"); only the first few are needed to repair gaps
                    "|".join(path.split("|")[:PATH_KEEP]) if path else None,
                )
            )
    df = pd.DataFrame(rows, columns=STOP_COLUMNS).drop_duplicates("id")
    del rows
    for col in ("pt_ar", "pt_dp"):
        df[col] = pd.to_datetime(df[col], format=TIME_FMT, errors="coerce")
    df["line"] = [_line_label(c, ln, n) for c, ln, n in zip(df.category, df.line_raw, df.train_number)]
    df["mode"] = df["category"].map(_mode)
    df = df.drop(columns=["line_raw"]).reset_index(drop=True)
    for col in ("slug", "station", "category", "line", "mode", "platform"):
        df[col] = df[col].astype("category")
    return df


def _line_label(category: str | None, line: str | None, number: str | None) -> str:
    category = category or "?"
    line = line if isinstance(line, str) and line else None
    number = number if isinstance(number, str) and number else None
    if category == "Bus":  # rail replacement bus (Schienenersatzverkehr)
        return f"SEV {line}" if line else "SEV"
    if category == "S":
        return f"S{line}" if line else "S"
    if line and _mode(category) == "regional" and category != "FEX":
        return f"{category}{line}"
    return category  # long-distance / unnumbered trains: one "line" per product


def _mode(category: str | None) -> str:
    if category == "S":
        return "S-Bahn"
    if category == "Bus":
        return "replacement bus"
    if category in {"ICE", "IC", "EC", "ECE", "NJ", "EN", "RJ", "RJX", "FLX", "TGV"}:
        return "long-distance"
    return "regional"


def parse_changes(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (latest change per stop id, station slug -> eva/name)."""
    rows, stations = [], {}
    for snapshot, name, payload in _iter_xml(source):
        root = ET.parse(io.BytesIO(payload)).getroot()
        if root.get("station"):
            stations[_slug(name)] = (root.get("station"), root.get("eva"))
        for s in root.findall("s"):
            ar, dp = s.find("ar"), s.find("dp")
            if ar is None and dp is None:
                continue
            rows.append(
                (
                    s.get("id"),
                    snapshot,
                    ar.get("ct") if ar is not None else None,
                    dp.get("ct") if dp is not None else None,
                    ar.get("cs") if ar is not None else None,
                    dp.get("cs") if dp is not None else None,
                )
            )
    ch = pd.DataFrame(rows, columns=["id", "snapshot", "ct_ar", "ct_dp", "cs_ar", "cs_dp"])
    del rows
    if ch.empty:
        ch = pd.DataFrame(columns=["id", "ct_ar", "ct_dp", "cs_ar", "cs_dp"])
    else:
        # later snapshots overwrite earlier ones; groupby.last() keeps the last non-null value
        ch = ch.sort_values("snapshot").groupby("id", as_index=False)[["ct_ar", "ct_dp", "cs_ar", "cs_dp"]].last()
    for col in ("ct_ar", "ct_dp"):
        ch[col] = pd.to_datetime(ch[col], format=TIME_FMT, errors="coerce")
    st = pd.DataFrame([{"slug": k, "change_name": v[0], "eva": v[1]} for k, v in stations.items()])
    return ch, st


def load_station_metadata(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for st in data.get("result", []):
        for eva in st.get("evaNumbers", [])[:1]:
            coords = (eva.get("geographicCoordinates") or {}).get("coordinates") or [None, None]
            rows.append({"eva": str(eva["number"]), "meta_name": st["name"], "lon": coords[0], "lat": coords[1]})
    return pd.DataFrame(rows)


def ingest(timetables: Path, changes: Path, station_json: Path, out_dir: Path) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stops = parse_timetables(timetables)
    stops.to_parquet(out_dir / "stops.parquet", index=False)
    ch, change_stations = parse_changes(changes)
    ch.to_parquet(out_dir / "changes.parquet", index=False)
    meta = load_station_metadata(station_json)

    names = stops.groupby("slug")["station"].agg(lambda s: s.mode().iat[0]).reset_index()
    stations = names.merge(change_stations, on="slug", how="left").merge(meta, on="eva", how="left")
    stations["name"] = stations["station"].map(display_name)
    stations = stations[["slug", "name", "station", "eva", "lat", "lon"]]

    stations.to_parquet(out_dir / "stations.parquet", index=False)
    counts = {"stops": len(stops), "changes": len(ch), "stations": len(stations)}
    log.info("ingested %s", counts)
    return counts


def display_name(raw: str) -> str:
    """'Berlin-Waidmannslust' -> 'Waidmannslust', 'Berlin Ostkreuz (S)' -> 'Ostkreuz'."""
    name = raw
    for prefix in ("Berlin-", "Berlin "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for suffix in (" (S-Bahn)", " (S)", "(S)"):
        name = name.removesuffix(suffix)
    return name.strip() or raw


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timetables", required=True, type=Path)
    p.add_argument("--changes", required=True, type=Path)
    p.add_argument("--stations", required=True, type=Path)
    p.add_argument("--out", default=Path("data/processed"), type=Path)
    a = p.parse_args()
    print(ingest(a.timetables, a.changes, a.stations, a.out))
