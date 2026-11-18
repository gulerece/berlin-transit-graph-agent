import pandas as pd

from transit_graph.graph_builder import build_graph, load_graph, save_graph
from transit_graph.ingest import _line_label, display_name, ingest


def test_display_name():
    assert display_name("Berlin-Waidmannslust") == "Waidmannslust"
    assert display_name("Berlin Gesundbrunnen(S)") == "Gesundbrunnen"
    assert display_name("Berlin Hbf (S-Bahn)") == "Hbf"
    assert display_name("Ahrensfelde") == "Ahrensfelde"


def test_line_labels():
    assert _line_label("S", "41", "1") == "S41"
    assert _line_label("RE", "1", "2") == "RE1"
    assert _line_label("ICE", None, "643") == "ICE"
    assert _line_label("Bus", "S42", "9") == "SEV S42"


def test_ingest_and_build(tiny_raw, tmp_path):
    out = tmp_path / "processed"
    counts = ingest(tiny_raw / "timetables", tiny_raw / "timetable_changes", tiny_raw / "station_data.json", out)
    assert counts["stops"] == 3 and counts["stations"] == 3

    stops = pd.read_parquet(out / "stops.parquet")
    assert set(stops["line"]) == {"S9"}

    G = build_graph(out, min_line_trips=1)
    # C was missing from the dataset but named in the planned path -> reconstructed node + split edges
    assert "recon:C" in G
    assert G.has_edge("b", "recon:C") and G.has_edge("recon:C", "d")
    assert not G.has_edge("b", "d")
    # B -> D planned 6 min (12:04 -> 12:10), split evenly across the two hops
    assert G.edges["b", "recon:C"]["minutes"] == 3.0
    # delays: B arrived 2 min late (12:05 vs 12:03), D 4 min late
    assert G.nodes["b"]["mean_delay_min"] == 2.0
    assert G.nodes["d"]["mean_delay_min"] == 4.0
    # delay gained between B (dep +2) and D (arr +4) is 2 min, split over 2 hops
    assert G.edges["b", "recon:C"]["added_delay"] == 1.0

    path = tmp_path / "g.json.gz"
    save_graph(G, path)
    G2 = load_graph(path)
    assert G2.number_of_edges() == G.number_of_edges()
    assert G2.graph["lines"]["S9"]["patterns"][0]["stops"] == ["a", "b", "d"]


def test_real_graph_shape(service):
    G = service.G
    assert G.number_of_nodes() > 120
    assert {"S1", "S41", "S42", "RE1"} <= set(G.graph["lines"])
    # Alexanderplatz's S-Bahn platforms are missing in the raw data; reconstruction restores S-Bahn service
    assert "S5" in G.nodes["alexanderplatz"]["lines"]
