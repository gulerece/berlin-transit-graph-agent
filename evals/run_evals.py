"""Run the agent benchmark.

python evals/run_evals.py --generate            # (re)create evals/cases.jsonl from the graph
python evals/run_evals.py --provider offline    # rule-based baseline
python evals/run_evals.py --provider ollama --model qwen2.5:7b
python evals/run_evals.py --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transit_graph.agent import TransitAgent, make_llm
from transit_graph.config import get_service
from transit_graph.evals import generate_cases, load_cases, run_eval, save_cases, to_markdown

HERE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    p.add_argument("--provider", default="offline")
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--graph", default=None)
    a = p.parse_args()

    svc = get_service(a.graph)
    if a.generate:
        cases = generate_cases(svc)
        save_cases(cases, a.cases)
        print(f"wrote {len(cases)} cases to {a.cases}")
        return

    cases = load_cases(a.cases)[: a.limit]
    agent = TransitAgent(svc, make_llm(svc, a.provider, a.model))
    result = run_eval(agent, cases)
    label = f"{a.provider}" + (f" ({a.model})" if a.model else "")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stem = label.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "-").replace("/", "-")
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(result["summary"], label)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    failures = [r for r in result["rows"] if not (r["tool_ok"] and r["args_ok"] and r["fact_recall"] == 1)]
    for r in failures[:10]:
        print("FAIL", r["id"], "|", r["question"], "|", r["tool_calls"], "|", r["answer"][:160])


if __name__ == "__main__":
    main()
