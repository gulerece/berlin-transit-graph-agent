from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from transit_graph.graph_builder import load_graph
from transit_graph.tools import TransitService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def service() -> TransitService:
    """Service over the committed real graph (one week of DBahn-Berlin data)."""
    return TransitService(load_graph(ROOT / "data" / "graph.json.gz"))


def _tar(path: Path, files: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture()
def tiny_raw(tmp_path: Path) -> Path:
    """A 4-station toy network in the DB XML format.

    Trip T1 runs A -> B -> C -> D but station C's record is missing (stop index jumps 2 -> 4);
    its planned path still names C, so the builder must reconstruct B -> C -> D.
    """

    def tt(station, stops):
        body = "".join(stops)
        return f'<?xml version="1.0"?><timetable station="{station}">{body}</timetable>'

    def s(sid, ar=None, dp=None, ppth=""):
        ar_x = f'<ar pt="{ar}" l="9"/>' if ar else ""
        dp_x = f'<dp pt="{dp}" l="9" ppth="{ppth}"/>' if dp else ""
        return f'<s id="{sid}"><tl c="S" n="1"/>{ar_x}{dp_x}</s>'

    timetables = {
        "2509021200/a_timetable.xml": tt("Berlin A", [s("T1-2509021200-1", dp="2509021200", ppth="Berlin B|Berlin C|Berlin D")]),
        "2509021200/b_timetable.xml": tt(
            "Berlin B", [s("T1-2509021200-2", ar="2509021203", dp="2509021204", ppth="Berlin C|Berlin D")]
        ),
        "2509021200/d_timetable.xml": tt("Berlin D", [s("T1-2509021200-4", ar="2509021210")]),
        "2509021200/c_timetable.xml": '<?xml version="1.0"?><timetable />',  # empty snapshot, like the real data
    }
    changes = {
        "2509021215/b_change.xml": '<?xml version="1.0"?><timetable station="Berlin B" eva="2">'
        '<s id="T1-2509021200-2"><ar ct="2509021205"/><dp ct="2509021206"/></s></timetable>',
        "2509021215/d_change.xml": '<?xml version="1.0"?><timetable station="Berlin D" eva="4">'
        '<s id="T1-2509021200-4"><ar ct="2509021214"/></s></timetable>',
        "2509021215/a_change.xml": '<?xml version="1.0"?><timetable station="Berlin A" eva="1"></timetable>',
    }
    raw = tmp_path / "raw"
    (raw / "timetables").mkdir(parents=True)
    (raw / "timetable_changes").mkdir()
    _tar(raw / "timetables" / "250902_250909.tar.gz", timetables)
    _tar(raw / "timetable_changes" / "250902_250909.tar.gz", changes)
    meta = {
        "result": [
            {"name": n, "evaNumbers": [{"number": e, "geographicCoordinates": {"coordinates": [13.4, 52.5]}}]}
            for n, e in [("A", 1), ("B", 2), ("D", 4)]
        ]
    }
    (raw / "station_data.json").write_text(json.dumps(meta))
    return raw
