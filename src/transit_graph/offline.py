"""A deterministic, rule-based chat model that speaks the LangChain tool-calling protocol.

It is the *baseline* the LLM agents are evaluated against, and it lets the whole
agent graph run in CI without an API key or a GPU. It routes a question to one
tool with keyword rules + fuzzy station extraction, then verbalises the tool
result with templates.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from rapidfuzz import fuzz

from .tools import ALIASES, _norm


class OfflineRouterModel(BaseChatModel):
    station_names: list[str] = Field(default_factory=list)
    line_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "offline-router"

    def bind_tools(self, tools: Any, **kwargs: Any):  # tools are chosen by rules
        return self

    # ------------------------------------------------------------------ routing
    def _stations_in(self, text: str) -> list[str]:
        """Stations mentioned in the text, in order of appearance."""
        norm = " " + _norm(text) + " "
        found: list[tuple[int, str]] = []
        candidates = {n: n for n in self.station_names}
        candidates.update({a: t for a, t in ALIASES.items() if t in self.station_names})
        for label, target in candidates.items():
            key = _norm(label)
            if len(key) < 3:
                continue
            pos = norm.find(" " + key + " ")
            if pos < 0:
                pos = norm.find(" " + key)
                if pos < 0 or len(key) < 6:
                    continue
            found.append((pos, target, len(key)))
        if not found:  # tolerate typos: fuzzy partial match of each station against the text
            for n in self.station_names:
                if len(n) >= 6 and fuzz.partial_ratio(_norm(n), norm) >= 90:
                    found.append((norm.find(_norm(n)[:4]), n, len(n)))
        # longest match wins at the same position (e.g. "Lichterfelde Ost" over "Ost...")
        found.sort(key=lambda f: (f[0], -f[2]))
        out, last_end = [], -1
        for pos, name, ln in found:
            if pos >= last_end and name not in out:
                out.append(name)
                last_end = pos + ln
        return out

    def _best_match(self, fragment: str) -> str | None:
        frag = _norm(fragment)
        scored = [(fuzz.WRatio(frag, _norm(n)), n) for n in self.station_names]
        scored += [(fuzz.WRatio(frag, _norm(a)), t) for a, t in ALIASES.items() if t in self.station_names]
        score, name = max(scored, default=(0, None))
        return name if score >= 80 else None

    def route(self, question: str) -> tuple[str, dict]:
        q = question.casefold()
        stations = self._stations_in(question)
        od = re.search(r"(?:from|von)\s+(.+?)\s+(?:to|nach|zum|zur)\s+(.+?)[?.!]*$", question, re.I)
        if od and len(stations) < 2:  # typo-tolerant origin/destination extraction
            pair = [self._best_match(od.group(1)), self._best_match(od.group(2))]
            if all(pair):
                stations = pair
        line = next(
            (ln for ln in sorted(self.line_names, key=len, reverse=True) if re.search(rf"\b{re.escape(ln.casefold())}\b", q)),
            None,
        )
        minutes = re.search(r"(\d+)\s*(min|minute)", q)

        if re.search(r"\b(clos|shut|what if|out of service|blocked)", q) and stations:
            return "closure_impact", {"station": stations[0]}
        if (re.search(r"within|reach|isochrone", q) or (minutes and len(stations) == 1)) and stations:
            return "reachable_within", {"station": stations[0], "minutes": float(minutes.group(1)) if minutes else 20}
        if len(stations) >= 2:
            args = {"origin": stations[0], "destination": stations[1]}
            if re.search(r"reliab|punctual|on time|safest|avoid delay|least delay", q):
                args["optimize"] = "reliable"
            return "plan_route", args
        if re.search(r"nearest|closest", q):
            m = re.search(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", q)
            if m:
                return "nearest_station", {"lat": float(m.group(1)), "lon": float(m.group(2))}
        if line and not stations:
            return "line_info", {"line": line}
        if stations:
            return "station_info", {"station": stations[0]}
        if re.search(r"\bhubs?\b|important|central|critical|bottleneck", q):
            return "hub_ranking", {"metric": "pagerank" if "busiest" in q else "betweenness"}
        if re.search(r"busiest", q):
            return "hub_ranking", {"metric": "pagerank"}
        if re.search(r"delay|punctual|late|unreliable|reliab", q):
            kind = "segments" if re.search(r"segment|stretch|between stations|track section|where .* gain", q) else "stations"
            return "delay_hotspots", {"kind": kind}
        return "search_network_knowledge", {"query": question}

    # ------------------------------------------------------------------ LangChain hooks
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            msg = AIMessage(content=verbalise(last.name or "", last.content))
        else:
            question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            tool, args = self.route(str(question))
            msg = AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def verbalise(tool: str, content: Any) -> str:
    try:
        r = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return str(content)
    if not isinstance(r, dict):
        return str(r)
    if "error" in r:
        return f"Sorry: {r['error']}"
    if tool == "plan_route":
        if not r.get("found"):
            return f"I found no connection from {r['origin']} to {r['destination']} in the observed network."
        legs = "; then ".join(
            f"{leg['line']} from {leg['from']} to {leg['to']} ({leg['stops']} stops, {leg['minutes']:g} min)" for leg in r["legs"]
        )
        return (
            f"From {r['origin']} to {r['destination']}: take {legs}. Total about {r['total_minutes']:g} min "
            f"with {r['transfers']} transfer(s); expect roughly {r['expected_extra_delay_min']:g} min of extra "
            f"delay, and a {r['probability_no_cancellation_pct']:g}% chance of no cancellation on the way."
        )
    if tool == "reachable_within":
        names = ", ".join(f"{s['station']} ({s['minutes']:g})" for s in r["stations"][:12])
        return f"{r['count']} stations are reachable from {r['origin']} within {r['budget_min']:g} min: {names}."
    if tool == "hub_ranking":
        top = ", ".join(f"{x['rank']}. {x['station']}" for x in r["ranking"][:5])
        return f"Top hubs by {r['metric']} ({r['meaning']}): {top}."
    if tool == "delay_hotspots":
        if r["kind"] == "stations":
            top = ", ".join(f"{h['station']} ({h['punctuality_pct']:g}% punctual)" for h in r["hotspots"][:5])
            return f"Least punctual stations ({r['definition']}): {top}."
        top = ", ".join(f"{h['from']} → {h['to']} (+{h['added_delay_min']:g} min)" for h in r["hotspots"][:5])
        return f"Segments where trains gain the most delay: {top}."
    if tool == "closure_impact":
        worst = r["worst_pairs"][0] if r["worst_pairs"] else None
        extra = ""
        if worst:
            after = "no alternative" if worst["after_min"] is None else f"{worst['after_min']:g} min"
            extra = f" Worst case: {worst['from']} → {worst['to']} goes from {worst['before_min']:g} min to {after}."
        return (
            f"If {r['station']} closed, {r['affected_neighbour_pairs']} neighbouring journeys would be affected, "
            f"{r['disconnected_pairs']} of them with no rail alternative; the others take "
            f"{r['mean_extra_minutes']:g} min longer on average."
            + (" It is an articulation point: closing it splits the network." if r["is_articulation_point"] else "")
            + extra
        )
    if tool == "station_info":
        rel = r.get("reliability")
        rel_txt = (
            f" Punctuality {rel['punctuality_pct']:g}%, mean arrival delay {rel['mean_arrival_delay_min']:g} min,"
            f" cancellations {rel['cancellation_rate_pct']:g}%."
            if rel
            else ""
        )
        return (
            f"{r['station']} is served by {', '.join(r['lines'])} and connects to "
            f"{', '.join(n['station'] for n in r['neighbours'])}.{rel_txt} "
            f"Hub rank #{r['hub_rank_betweenness']} by betweenness."
        )
    if tool == "line_info":
        p = r["patterns"][0]
        return (
            f"{r['line']} ({r['mode']}) runs {p['from']} → {p['to']} via {', '.join(p['stops'][1:-1])}. "
            f"Cancellation rate {r['cancellation_rate_pct']:g}%."
        )
    if tool == "nearest_station":
        return "Nearest stations: " + ", ".join(f"{n['station']} ({n['distance_km']:g} km)" for n in r["nearest"]) + "."
    if tool == "find_station":
        return "Matching stations: " + ", ".join(m["station"] for m in r["matches"]) + "."
    if tool == "search_network_knowledge":
        hits = r["results"]
        return hits[0]["text"] if hits else "I found nothing relevant."
    return json.dumps(r, ensure_ascii=False)[:1500]
