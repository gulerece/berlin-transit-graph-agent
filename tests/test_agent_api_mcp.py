import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from transit_graph.agent import TransitAgent
from transit_graph.evals import generate_cases, run_eval


@pytest.fixture(scope="module")
def agent(service):
    return TransitAgent(service)  # offline router model


@pytest.mark.parametrize(
    "question,tool",
    [
        ("How do I get from Alexanderplatz to Wannsee?", "plan_route"),
        ("Wie komme ich von Hauptbahnhof nach Ostkreuz?", "plan_route"),
        ("What happens if Südkreuz is closed?", "closure_impact"),
        ("Which stations can I reach from Ostkreuz within 10 minutes?", "reachable_within"),
        ("Which stations are least punctual?", "delay_hotspots"),
        ("Which stations does the S41 serve?", "line_info"),
    ],
)
def test_agent_routes_to_tool(agent, question, tool):
    r = agent.ask(question)
    assert r.tool_calls and r.tool_calls[0]["name"] == tool
    assert r.answer


def test_agent_handles_unknown_station(agent):
    r = agent.ask("Tell me about Hogwarts station")
    assert r.answer  # no crash; tool error is turned into a message


def test_eval_harness(agent, service):
    cases = generate_cases(service, n_routes=5)
    res = run_eval(agent, cases)
    s = res["summary"]["overall"]
    assert s["n"] == len(cases)
    for k in ("tool_accuracy", "arg_accuracy", "fact_recall"):
        assert 0 <= s[k] <= 1


def test_api(monkeypatch):
    from transit_graph.api import app

    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.get("/route", params={"origin": "Spandau", "destination": "Ostkreuz"})
        assert r.status_code == 200 and r.json()["found"]
        assert c.get("/stations/Hogwarts").status_code == 404
        a = c.post("/ask", json={"question": "What if Ostkreuz closes?"}).json()
        assert a["tool_calls"][0]["name"] == "closure_impact"


def test_mcp_server():
    from transit_graph.mcp_server import server

    async def go():
        names = {t.name for t in await server.list_tools()}
        assert {"plan_route", "closure_impact", "search_network_knowledge"} <= names
        res = await server.call_tool("plan_route", {"origin": "Hbf", "destination": "Ostkreuz"})
        return json.loads(res.content[0].text)

    out = asyncio.run(go())
    assert out["found"] and out["destination"] == "Ostkreuz"
