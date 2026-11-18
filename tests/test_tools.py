import pytest

from transit_graph.tools import StationNotFound


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Alex", "Alexanderplatz"),
        ("Hauptbahnhof", "Hbf"),
        ("zoo", "Zoologischer Garten"),
        ("S+U Gesundbrunnen Bhf (Berlin)", "Gesundbrunnen"),
        ("friedrichstr", "Friedrichstraße"),
        ("Warschauer Str", "Warschauer Straße"),
        ("Ostkreutz", "Ostkreuz"),
    ],
)
def test_resolve(service, query, expected):
    assert service.name(service.resolve(query)) == expected


def test_unknown_station_suggests(service):
    with pytest.raises(StationNotFound) as exc:
        service.resolve("Qwertzuiop")
    assert exc.value.suggestions


def test_route_is_consistent(service):
    r = service.plan_route("Spandau", "Grünau")
    assert r["found"] and r["path"][0] == "Spandau" and r["path"][-1] == "Grünau"
    assert r["transfers"] == len(r["legs"]) - 1
    assert r["total_minutes"] == pytest.approx(r["in_vehicle_minutes"] + 5 * r["transfers"], abs=0.11)
    for a, b in zip(r["legs"], r["legs"][1:]):
        assert a["to"] == b["from"]
    for x, y in zip(r["path"], r["path"][1:]):
        assert service.G.has_edge(service.resolve(x), service.resolve(y))


def test_fastest_is_never_slower_than_reliable(service):
    for o, d in [("Spandau", "Grünau"), ("Wannsee", "Ostkreuz"), ("Pankow", "Südkreuz")]:
        fast = service.plan_route(o, d)
        safe = service.plan_route(o, d, optimize="reliable")
        assert fast["total_minutes"] <= safe["total_minutes"] + 1e-9


def test_avoid_station(service):
    r = service.plan_route("Ostbahnhof", "Hbf", avoid_stations=["Friedrichstraße"])
    assert "Friedrichstraße" not in r["path"]


def test_isochrone_monotonic(service):
    small = service.reachable_within("Ostkreuz", 5)
    big = service.reachable_within("Ostkreuz", 15)
    assert small["count"] < big["count"]
    assert {s["station"] for s in small["stations"]} <= {s["station"] for s in big["stations"]}
    assert all(s["minutes"] <= 5 for s in small["stations"])


def test_hubs_and_delays(service):
    hubs = service.hub_ranking("betweenness", 5)["ranking"]
    assert len(hubs) == 5 and hubs[0]["score"] >= hubs[-1]["score"]
    worst = service.delay_hotspots("stations", 3)["hotspots"]
    assert worst[0]["punctuality_pct"] <= worst[-1]["punctuality_pct"]
    segs = service.delay_hotspots("segments", 3)["hotspots"]
    assert segs[0]["added_delay_min"] >= segs[-1]["added_delay_min"]


def test_closure_impact(service):
    r = service.closure_impact("Ostkreuz")
    assert r["affected_neighbour_pairs"] > 0
    for p in r["worst_pairs"]:
        if p["after_min"] is not None:
            assert p["after_min"] >= p["before_min"]


def test_station_and_line_info(service):
    info = service.station_info("Südkreuz")
    assert "S41" in info["lines"] and 0 < info["reliability"]["punctuality_pct"] <= 100
    line = service.line_info("s41")
    assert line["line"] == "S41" and len(line["patterns"][0]["stops"]) > 10
    assert "error" in service.line_info("U99")


def test_nearest_station(service):
    r = service.nearest_station(52.5215, 13.411)  # Alexanderplatz coordinates
    assert r["nearest"][0]["station"] == "Alexanderplatz"


def test_graphrag_search(service):
    res = service.search_network_knowledge("Ostkreuz station punctuality", k=3)["results"]
    assert any(r["title"] == "Ostkreuz" for r in res)
    assert any(r["kind"] == "community" and r.get("expanded_from") for r in res)
