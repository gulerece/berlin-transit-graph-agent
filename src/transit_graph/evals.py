"""Agent evaluation: generate data-grounded test cases, run an agent, score it.

Ground truth comes from the graph tools themselves, so every expected number is
verifiable. Cases mix English, German, aliases ("Alex", "Hauptbahnhof") and typos
to test robustness. Metrics:

* tool_accuracy   – the first tool called is an acceptable tool for the question
* arg_accuracy    – the stations passed to the tool resolve to the right stations
* fact_recall     – the final answer contains the expected facts (numbers within tolerance)
* latency p50/p95
"""

from __future__ import annotations

import json
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import TransitAgent
from .tools import StationNotFound, TransitService

ROUTE_TEMPLATES = [
    ("How do I get from {o} to {d}?", "fastest"),
    ("Fastest connection {o} -> {d}", "fastest"),
    ("Wie komme ich von {o} nach {d}?", "fastest"),
    ("I'm at {o} and need to get to {d}. Which trains should I take?", "fastest"),
    ("What is the most reliable way to travel from {o} to {d}? I can't be late.", "reliable"),
    ("from {o} to {d} please", "fastest"),
]
ALIAS_OF = {"Alexanderplatz": "Alex", "Hbf": "Hauptbahnhof", "Zoologischer Garten": "Zoo"}


@dataclass
class Case:
    id: str
    category: str
    question: str
    tools: list[str]
    stations: list[str] = field(default_factory=list)  # expected station slugs in tool args
    facts: list[dict] = field(default_factory=list)  # {"text": str} or {"number": x, "tol": t}


def _typo(name: str, rng: random.Random) -> str:
    if len(name) < 8:
        return name
    i = rng.randrange(2, len(name) - 2)
    return name[:i] + name[i + 1 :]


def generate_cases(svc: TransitService, n_routes: int = 30, seed: int = 7) -> list[Case]:
    rng = random.Random(seed)
    G = svc.G
    measured = sorted(n for n, d in G.nodes(data=True) if not d.get("reconstructed"))
    busy = sorted(measured, key=lambda n: -(G.nodes[n]["stops_per_day"] or 0))[:60]
    cases: list[Case] = []

    def surface(slug: str) -> str:
        name = svc.name(slug)
        r = rng.random()
        if name in ALIAS_OF and r < 0.5:
            return ALIAS_OF[name]
        if r < 0.15:
            return _typo(name, rng)
        return name

    while len([c for c in cases if c.category == "route"]) < n_routes:
        o, d = rng.sample(busy, 2)
        tmpl, opt = rng.choice(ROUTE_TEMPLATES)
        truth = svc.plan_route(o, d, optimize=opt)
        if not truth.get("found"):
            continue
        cases.append(
            Case(
                f"route-{len(cases)}",
                "route",
                tmpl.format(o=surface(o), d=surface(d)),
                ["plan_route"],
                [o, d],
                [{"text": svc.name(d)}, {"number": truth["total_minutes"], "tol": 1.5}],
            )
        )

    for i, s in enumerate(rng.sample(busy, 8)):
        q = rng.choice(
            ["What happens if {s} is closed?", "What if {s} station shuts down for a week?", "Impact of closing {s}?"]
        ).format(s=surface(s))
        t = svc.closure_impact(s)
        cases.append(
            Case(f"closure-{i}", "closure", q, ["closure_impact"], [s], [{"number": t["affected_neighbour_pairs"], "tol": 0}])
        )

    for i, s in enumerate(rng.sample(busy, 8)):
        m = rng.choice([10, 15, 20])
        q = rng.choice(
            ["Which stations can I reach from {s} within {m} minutes?", "Where can I get in {m} minutes from {s}?"]
        ).format(s=surface(s), m=m)
        t = svc.reachable_within(s, m)
        cases.append(Case(f"reach-{i}", "reachable", q, ["reachable_within"], [s], [{"number": t["count"], "tol": 0}]))

    for i, s in enumerate(rng.sample(busy, 8)):
        q = rng.choice(["How punctual is {s}?", "Tell me about {s} station.", "Is {s} a reliable station?"]).format(s=surface(s))
        t = svc.station_info(s)
        cases.append(
            Case(
                f"station-{i}", "station", q, ["station_info"], [s], [{"number": t["reliability"]["punctuality_pct"], "tol": 0.5}]
            )
        )

    hubs = svc.hub_ranking("betweenness")["ranking"]
    for i, q in enumerate(
        [
            "Which stations are the most important hubs in the network?",
            "What are the most central stations?",
            "Which station is the biggest bottleneck of the network?",
        ]
    ):
        cases.append(Case(f"hubs-{i}", "hubs", q, ["hub_ranking"], [], [{"text": hubs[0]["station"]}]))

    worst = svc.delay_hotspots("stations")["hotspots"][0]["station"]
    seg = svc.delay_hotspots("segments")["hotspots"][0]
    for i, q in enumerate(
        ["Which stations are the least punctual?", "Where are delays worst?", "Welche Bahnhöfe sind am unpünktlichsten?"]
    ):
        cases.append(Case(f"delays-{i}", "delays", q, ["delay_hotspots"], [], [{"text": worst}]))
    for i, q in enumerate(
        ["On which track segments do trains gain the most delay?", "Between which stations do trains lose the most time?"]
    ):
        cases.append(Case(f"segments-{i}", "delays", q, ["delay_hotspots"], [], [{"text": seg["from"]}]))

    for i, line in enumerate(["S41", "S1", "S5", "RE1"]):
        p = svc.line_info(line)["patterns"][0]
        q = rng.choice(["Which stations does the {line} serve?", "Where does {line} go?"]).format(line=line)
        cases.append(Case(f"line-{i}", "line", q, ["line_info"], [], [{"text": p["to"]}]))
    return cases


def save_cases(cases: list[Case], path: Path) -> None:
    Path(path).write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")


def load_cases(path: Path) -> list[Case]:
    return [Case(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _has_fact(answer: str, fact: dict) -> bool:
    if "text" in fact:
        return fact["text"].casefold() in answer.casefold()
    no_impact = r"\b(no|zero|keine?)\b.{0,40}\b(impact|effect|affected|journeys?|disruption|auswirkung)"
    if fact["number"] == 0 and re.search(no_impact, answer, re.I):
        return True
    nums = [float(x.replace(",", ".")) for x in _NUM.findall(answer)]
    return any(abs(x - fact["number"]) <= fact["tol"] for x in nums)


def _arg_stations(svc: TransitService, call: dict) -> list[str]:
    out = []
    for key in ("origin", "destination", "station"):
        if key in call["args"]:
            try:
                out.append(svc.resolve(str(call["args"][key])))
            except StationNotFound:
                out.append(None)
    return out


def run_eval(agent: TransitAgent, cases: list[Case]) -> dict:
    svc = agent.service
    rows = []
    for c in cases:
        try:
            r = agent.ask(c.question)
            answer, calls, latency, err = r.answer, r.tool_calls, r.latency_s, None
        except Exception as exc:  # an agent crash counts as a failure, not an eval crash
            answer, calls, latency, err = "", [], 0.0, repr(exc)
        first = calls[0] if calls else None
        tool_ok = bool(first and first["name"] in c.tools)
        args_ok = (not c.stations) or bool(first and _arg_stations(svc, first)[: len(c.stations)] == c.stations)
        facts = [_has_fact(answer, f) for f in c.facts]
        rows.append(
            {
                "id": c.id,
                "category": c.category,
                "question": c.question,
                "tool_ok": tool_ok,
                "args_ok": args_ok,
                "fact_recall": sum(facts) / len(facts) if facts else 1.0,
                "latency_s": latency,
                "tool_calls": calls,
                "answer": answer,
                "error": err,
            }
        )
    return {"summary": summarise(rows), "rows": rows}


def summarise(rows: list[dict]) -> dict:
    def agg(rs):
        lat = sorted(r["latency_s"] for r in rs)
        return {
            "n": len(rs),
            "tool_accuracy": round(statistics.mean(r["tool_ok"] for r in rs), 3),
            "arg_accuracy": round(statistics.mean(r["args_ok"] for r in rs), 3),
            "fact_recall": round(statistics.mean(r["fact_recall"] for r in rs), 3),
            "latency_p50_s": round(lat[len(lat) // 2], 3),
            "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
        }

    cats = sorted({r["category"] for r in rows})
    return {"overall": agg(rows), "by_category": {c: agg([r for r in rows if r["category"] == c]) for c in cats}}


def to_markdown(summary: dict, label: str) -> str:
    lines = [
        f"### {label}",
        "",
        "| category | n | tool acc. | arg acc. | fact recall | p50 latency |",
        "|---|---|---|---|---|---|",
    ]
    for cat, s in [("**overall**", summary["overall"]), *summary["by_category"].items()]:
        lines.append(
            f"| {cat} | {s['n']} | {s['tool_accuracy']:.0%} | {s['arg_accuracy']:.0%} | "
            f"{s['fact_recall']:.0%} | {s['latency_p50_s']:.2f}s |"
        )
    return "\n".join(lines) + "\n"
