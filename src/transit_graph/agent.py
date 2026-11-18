"""LangGraph agent: an LLM that answers transit questions by calling graph tools.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

The LLM backend is swappable (Ollama, OpenAI, Anthropic, or the offline
rule-based router) via ``TRANSIT_LLM_PROVIDER`` / ``TRANSIT_LLM_MODEL``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import create_model

from .offline import OfflineRouterModel
from .tools import StationNotFound, TransitService

SYSTEM_PROMPT = """You are a Berlin rail network analyst. You answer questions about the S-Bahn and
regional rail network using ONLY the tools provided; they query a knowledge graph built from real
Deutsche Bahn timetable and delay data ({period}).

Rules:
- Always call a tool before answering factual questions; never invent stations, times or statistics.
- Pass station names as the user wrote them; the tools resolve fuzzy names.
- For "what if X closes" questions use closure_impact; for "how do I get from A to B" use plan_route
  with optimize="fastest" by default; use optimize="reliable" ONLY if the user explicitly mentions
  reliability, punctuality, delays or not being late.
- Always state the headline numbers from the tool result: total travel time and number of transfers
  for routes, the number of stations for reachability, affected/disconnected journeys for closures.
- Report numbers with units and say they are based on observed data, not live departures.
- To call a tool, use the tool-calling interface; never write a tool call as text.
- Answer in the user's language, concisely."""

TOOL_NAMES = [
    "plan_route",
    "station_info",
    "reachable_within",
    "hub_ranking",
    "delay_hotspots",
    "closure_impact",
    "line_info",
    "nearest_station",
    "find_station",
    "search_network_knowledge",
]


HIDDEN_ARGS = {"transfer_penalty_min", "exclude_modes", "min_trips"}  # internal knobs the LLM should not set


def make_tools(service: TransitService) -> list[StructuredTool]:
    tools = []
    for name in TOOL_NAMES:
        fn = getattr(service, name)

        def run(_fn=fn, **kwargs):
            try:
                return json.dumps(_fn(**kwargs), ensure_ascii=False)
            except StationNotFound as exc:
                return json.dumps({"error": str(exc), "suggestions": exc.suggestions}, ensure_ascii=False)

        tool = StructuredTool.from_function(fn, name=name, description=(fn.__doc__ or name).strip())
        tool.func = run
        visible = {k: (f.annotation, f) for k, f in tool.args_schema.model_fields.items() if k not in HIDDEN_ARGS}
        tool.args_schema = create_model(name, **visible)
        tools.append(tool)
    return tools


def make_llm(service: TransitService, provider: str | None = None, model: str | None = None):
    provider = (provider or os.getenv("TRANSIT_LLM_PROVIDER", "offline")).lower()
    model = model or os.getenv("TRANSIT_LLM_MODEL")
    if provider == "offline":
        return OfflineRouterModel(
            station_names=[d["name"] for _, d in service.G.nodes(data=True)],
            line_names=list(service.G.graph.get("lines", {})),
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or "qwen2.5:7b",
            temperature=0,
            num_ctx=8192,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model or "claude-haiku-5-5", temperature=0)
    raise ValueError(f"Unknown provider '{provider}' (use offline, ollama, openai or anthropic)")


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    steps: int = 0


class TransitAgent:
    def __init__(self, service: TransitService, llm=None, max_steps: int = 6):
        self.service = service
        self.tools = make_tools(service)
        self.llm = (llm or make_llm(service)).bind_tools(self.tools)
        self.max_steps = max_steps
        start, end = (p[:10] for p in service.G.graph.get("period", ["?", "?"]))
        self.system_prompt = SYSTEM_PROMPT.format(period=f"{start} to {end}")
        self.graph = self._build()

    def _build(self):
        def agent(state: MessagesState):
            msgs = [SystemMessage(self.system_prompt), *state["messages"]]
            return {"messages": [self.llm.invoke(msgs)]}

        g = StateGraph(MessagesState)
        g.add_node("agent", agent)
        g.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", tools_condition)
        g.add_edge("tools", "agent")
        return g.compile()

    def ask(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        out = self.graph.invoke({"messages": [HumanMessage(question)]}, config={"recursion_limit": 2 * self.max_steps + 1})
        msgs = out["messages"]
        calls = [{"name": c["name"], "args": c["args"]} for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
        final = next((m for m in reversed(msgs) if isinstance(m, AIMessage) and not m.tool_calls), None)
        answer = final.content if final else ""
        if isinstance(answer, list):  # some providers return content blocks
            answer = " ".join(b.get("text", "") for b in answer if isinstance(b, dict))
        results = []
        for m in msgs:
            if isinstance(m, ToolMessage):
                try:
                    results.append({"name": m.name, "result": json.loads(m.content)})
                except (TypeError, json.JSONDecodeError):
                    results.append({"name": m.name, "result": {"error": str(m.content)}})
        return AgentResult(
            answer=str(answer).strip(),
            tool_calls=calls,
            tool_results=results,
            latency_s=round(time.perf_counter() - t0, 3),
            steps=sum(isinstance(m, ToolMessage) for m in msgs),
        )
